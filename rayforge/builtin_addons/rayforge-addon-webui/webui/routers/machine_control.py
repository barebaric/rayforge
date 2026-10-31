"""Machine control endpoints: home, jog, move, power, WCS, macros."""

from gettext import gettext as _

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field
from raygeo.ops.axis import Axis

from rayforge.machine.cmd import JobAlreadyRunningError

from ..coordinates import (
    jog_for_panel_direction,
    panel_point_to_machine,
)
from .deps import (
    bad_request,
    conflict,
    get_bridge,
    machine_unavailable,
    not_found,
)

router = APIRouter(tags=["machine-control"])

_AXIS_NAMES = {"x": Axis.X, "y": Axis.Y, "z": Axis.Z}


class HomeRequest(BaseModel):
    axes: list[str] = Field(default_factory=list)


class JogRequest(BaseModel):
    direction: str
    distance_mm: float = Field(gt=0)
    speed: int | None = None


class MoveToRequest(BaseModel):
    x: float
    y: float
    z: float | None = None
    speed: float | None = None


class PowerRequest(BaseModel):
    head: str | None = None
    percent: float = Field(ge=0, le=1)


class RunRawRequest(BaseModel):
    gcode: str = Field(min_length=1)


class SetWorkOriginHereRequest(BaseModel):
    axes: list[str] = Field(default_factory=list)


class SetWorkOriginRequest(BaseModel):
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0
    wcs_slot: str | None = None


class SelectWcsRequest(BaseModel):
    slot: str


class FrameRequest(BaseModel):
    include_overscan: bool = False


def _parse_axes(names: list[str]) -> Axis | None:
    flags: Axis | None = None
    for name in names:
        axis = _AXIS_NAMES.get(name.lower())
        if axis is None:
            raise bad_request(_("Unknown axis '{axis}'").format(axis=name))
        flags = axis if flags is None else flags | axis
    return flags


def _get_machine(editor):
    machine = editor.context.machine
    if machine is None:
        raise machine_unavailable()
    return machine


def _resolve_head(machine, head_uid: str | None):
    head = None
    if head_uid is not None:
        head = machine.get_head_by_uid(head_uid)
        if head is None:
            raise not_found(_("Head not found"))
        return head
    head = machine.get_default_laser_head()
    if head is None:
        raise bad_request(_("Machine has no laser heads configured"))
    return head


def _check_capability(machine, capability: str, axes: Axis | None = None):
    if capability == "home" and not machine.can_home(axes):
        raise conflict(_("Machine does not support homing"))
    if capability == "jog" and not machine.can_jog(axes):
        raise conflict(_("Machine does not support jogging"))


@router.post("/machine/home")
async def home(request: Request, payload: HomeRequest) -> dict:
    bridge = get_bridge(request)
    axes = _parse_axes(payload.axes) if payload.axes else None

    def _home(editor):
        machine = _get_machine(editor)
        _check_capability(machine, "home", axes)
        machine_cmd = request.app.state.machine_cmd
        machine_cmd.home(machine, axes)

    await bridge.run(_home, bridge.editor)
    return {"status": "accepted"}


@router.post("/machine/jog")
async def jog(request: Request, payload: JogRequest) -> dict:
    bridge = get_bridge(request)

    def _jog(editor):
        machine = _get_machine(editor)
        try:
            deltas = jog_for_panel_direction(
                editor.context, payload.direction, payload.distance_mm
            )
        except ValueError as e:
            raise bad_request(str(e)) from e
        if not deltas:
            raise bad_request(_("Direction maps to no movable axis"))
        flags = None
        for axis in deltas:
            flags = axis if flags is None else flags | axis
        _check_capability(machine, "jog", flags)
        speed = payload.speed or machine.jog_speed
        machine_cmd = request.app.state.machine_cmd
        machine_cmd.jog(machine, deltas, speed)
        return deltas

    deltas = await bridge.run(_jog, bridge.editor)
    named = {axis.name.lower(): float(v) for axis, v in deltas.items()}
    return {"status": "accepted", "deltas": named}


@router.post("/machine/move-to")
async def move_to(request: Request, payload: MoveToRequest) -> dict:
    bridge = get_bridge(request)

    def _move(editor):
        machine = _get_machine(editor)
        machine_x, machine_y = panel_point_to_machine(
            editor.context, payload.x, payload.y
        )
        machine_cmd = request.app.state.machine_cmd
        machine_cmd.move_to(
            machine, machine_x, machine_y, payload.z, payload.speed
        )

    await bridge.run(_move, bridge.editor)
    return {"status": "accepted"}


@router.post("/machine/set-power")
async def set_power(request: Request, payload: PowerRequest) -> dict:
    bridge = get_bridge(request)

    def _power(editor):
        machine = _get_machine(editor)
        head = _resolve_head(machine, payload.head)
        machine_cmd = request.app.state.machine_cmd
        machine_cmd.set_power(head, payload.percent, machine)

    await bridge.run(_power, bridge.editor)
    return {"status": "accepted"}


@router.post("/machine/set-focus-power")
async def set_focus_power(request: Request, payload: PowerRequest) -> dict:
    bridge = get_bridge(request)

    def _power(editor):
        machine = _get_machine(editor)
        head = _resolve_head(machine, payload.head)
        machine_cmd = request.app.state.machine_cmd
        machine_cmd.set_focus_power(head, payload.percent, machine)

    await bridge.run(_power, bridge.editor)
    return {"status": "accepted"}


@router.post("/machine/run-raw")
async def run_raw(request: Request, payload: RunRawRequest) -> dict:
    bridge = get_bridge(request)

    def _run(editor):
        machine = _get_machine(editor)
        return machine.run_raw(payload.gcode)

    await bridge.run(_run, bridge.editor)
    return {"status": "accepted"}


@router.post("/machine/clear-alarm")
async def clear_alarm(request: Request) -> dict:
    bridge = get_bridge(request)

    def _clear(editor):
        machine = _get_machine(editor)
        machine_cmd = request.app.state.machine_cmd
        machine_cmd.clear_alarm(machine)

    await bridge.run(_clear, bridge.editor)
    _publish_machine_update(request)
    return {"status": "accepted"}


def _publish_machine_update(request: Request) -> None:
    events = request.app.state.events
    events.publish_machine_event(force=True)
    events.publish_viewport_event()


@router.post("/machine/set-work-origin-here")
async def set_work_origin_here(
    request: Request, payload: SetWorkOriginHereRequest
) -> dict:
    bridge = get_bridge(request)
    axes = _parse_axes(payload.axes) if payload.axes else Axis.X | Axis.Y

    def _origin_here(editor):
        machine = _get_machine(editor)
        return machine.controller.set_work_origin_here(axes)

    await bridge.run(_origin_here, bridge.editor)
    _publish_machine_update(request)
    return {"status": "accepted"}


@router.post("/machine/set-work-origin")
async def set_work_origin(
    request: Request, payload: SetWorkOriginRequest
) -> dict:
    bridge = get_bridge(request)

    def _origin(editor):
        machine = _get_machine(editor)
        machine_x, machine_y = panel_point_to_machine(
            editor.context, payload.x, payload.y
        )
        return machine.controller.set_work_origin(
            machine_x, machine_y, payload.z, payload.wcs_slot
        )

    await bridge.run(_origin, bridge.editor)
    _publish_machine_update(request)
    return {"status": "accepted"}


@router.post("/machine/select-wcs")
async def select_wcs(request: Request, payload: SelectWcsRequest) -> dict:
    bridge = get_bridge(request)

    def _select(editor):
        machine = _get_machine(editor)
        if payload.slot not in machine.coordinate_systems:
            raise not_found(
                _("Unknown WCS slot '{slot}'").format(slot=payload.slot)
            )
        return machine.controller.select_wcs(payload.slot)

    await bridge.run(_select, bridge.editor)
    _publish_machine_update(request)
    return {"status": "accepted", "active_wcs": payload.slot}


@router.post("/machine/sync-wcs")
async def sync_wcs(request: Request) -> dict:
    bridge = get_bridge(request)

    def _sync(editor):
        machine = _get_machine(editor)
        return machine.controller.sync_wcs_from_device()

    await bridge.run(_sync, bridge.editor)
    _publish_machine_update(request)
    return {"status": "accepted"}


@router.post("/machine/macros/{macro_uid}/run")
async def run_macro(request: Request, macro_uid: str) -> dict:
    bridge = get_bridge(request)

    def _run_macro(editor):
        machine = _get_machine(editor)
        macro = machine.macros.get(macro_uid)
        if macro is None or not macro.enabled:
            raise not_found(_("Macro not found or disabled"))
        machine_cmd = request.app.state.machine_cmd
        machine_cmd.execute_macro_by_uid(machine, macro_uid)

    await bridge.run(_run_macro, bridge.editor)
    return {"status": "accepted"}


@router.post("/machine/frame")
async def frame(request: Request, payload: FrameRequest) -> dict:
    bridge = get_bridge(request)

    def _frame(editor):
        machine = _get_machine(editor)
        machine_cmd = request.app.state.machine_cmd
        events = request.app.state.events
        return machine_cmd.frame_job(
            machine,
            on_progress=events.on_job_progress,
            include_overscan=payload.include_overscan,
        )

    try:
        await bridge.run(_frame, bridge.editor)
    except JobAlreadyRunningError as e:
        raise conflict(str(e)) from e
    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=_("Framing failed: {error}").format(error=e),
        ) from e
    return {"status": "accepted"}
