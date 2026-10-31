"""Machine endpoints: listing, connect, disconnect, state, settings,
driver setup, heads, macros, work offsets, no-go zones and rotary."""

from gettext import gettext as _

from fastapi import APIRouter, Request
from pydantic import BaseModel

from rayforge.machine.driver import (
    driver_by_classname,
    drivers,
    get_driver_cls,
)
from rayforge.machine.driver.driver import DRIVER_MATURITY_LABELS
from rayforge.machine.models.laser import LaserHead
from rayforge.machine.models.machine import Machine, Origin
from rayforge.machine.models.machine_panel import PanelOrientation
from rayforge.machine.models.macro import Macro
from rayforge.machine.models.zone import Zone, ZoneShape

from ..serializers import machine_state
from ..serializers.varset_schema import serialize_var
from .deps import get_bridge, machine_unavailable, not_found

router = APIRouter(tags=["machines"])

ORIGINS = [origin.value for origin in Origin]

HEAD_SETTINGS = {
    "max_power": float,
    "frame_power": float,
    "focus_power": float,
    "frame_speed": float,
    "frame_repeat_count": float,
    "frame_corner_pause": float,
    "focal_distance": float,
    "pointer_offset_enabled": bool,
    "pointer_offset_x": float,
    "pointer_offset_y": float,
}

ROTARY_SETTINGS = {
    "name": str,
    "default_diameter": float,
    "mm_per_rotation": float,
    "reverse_axis": bool,
}


class MachineSettingsPatch(BaseModel):
    values: dict


class DriverSwitchRequest(BaseModel):
    driver: str


class DriverSetupPatch(BaseModel):
    values: dict


class HeadsPatch(BaseModel):
    heads: list[dict]


class MacroRequest(BaseModel):
    name: str = ""
    code: str = ""
    enabled: bool = True


class MacroPatch(BaseModel):
    name: str | None = None
    code: str | None = None
    enabled: bool | None = None


class WcsPatch(BaseModel):
    offset: list[float]
    label: str | None = None


class NogoZoneRequest(BaseModel):
    name: str | None = None
    shape: str = "rect"
    params: dict[str, float]
    enabled: bool = True


class RotaryModulePatch(BaseModel):
    name: str | None = None
    default_diameter: float | None = None
    mm_per_rotation: float | None = None
    reverse_axis: bool | None = None


def _get_machine(context, machine_id: str) -> Machine:
    machine = context.machine_mgr.get_machine_by_id(machine_id)
    if machine is None:
        raise not_found(_("Machine not found"))
    return machine


def _machine_summary(context, machine: Machine) -> dict:
    return {
        "id": machine.id,
        "name": machine.name,
        "driver": machine.driver_name,
        "connected": machine.is_connected(),
        "connection": machine.connection_status.name,
        "active": context.machine is machine,
        "placeholder": machine.placeholder,
    }


@router.get("/machines")
async def get_machines(request: Request) -> dict:
    bridge = get_bridge(request)

    def _machines(editor) -> dict:
        context = editor.context
        machines = [
            _machine_summary(context, machine)
            for machine in context.machine_mgr.get_machines()
        ]
        return {"machines": machines}

    return await bridge.run(_machines, bridge.editor)


@router.post("/machines/{machine_id}/connect")
async def connect_machine(request: Request, machine_id: str) -> dict:
    bridge = get_bridge(request)

    def _connect(editor) -> dict:
        context = editor.context
        machine = _get_machine(context, machine_id)
        return {"machine": _machine_summary(context, machine)}

    summary = await bridge.run(_connect, bridge.editor)

    def _do_connect(editor) -> None:
        context = editor.context
        machine = _get_machine(context, machine_id)
        return context.machine_mgr.get_controller(machine.id).connect()

    await bridge.run(_do_connect, bridge.editor)
    return summary


@router.post("/machines/{machine_id}/disconnect")
async def disconnect_machine(request: Request, machine_id: str) -> dict:
    bridge = get_bridge(request)

    def _disconnect(editor) -> dict:
        context = editor.context
        machine = _get_machine(context, machine_id)
        return context.machine_mgr.get_controller(machine.id).disconnect()

    await bridge.run(_disconnect, bridge.editor)

    def _summary(editor) -> dict:
        context = editor.context
        machine = _get_machine(context, machine_id)
        return _machine_summary(context, machine)

    return await bridge.run(_summary, bridge.editor)


@router.get("/machine/state")
async def get_machine_state(request: Request) -> dict:
    bridge = get_bridge(request)

    def _state(editor):
        machine = editor.context.machine
        if machine is None:
            raise machine_unavailable()
        return machine_state(machine)

    return await bridge.run(_state, bridge.editor)


def _heads_payload(machine: Machine) -> list[dict]:
    from rayforge.machine.models.laser import LaserHead

    heads = []
    for head in machine.heads:
        entry: dict = {
            "uid": head.uid,
            "name": head.name,
            "type": type(head).__name__,
        }
        if isinstance(head, LaserHead):
            entry.update(
                {
                    "max_power": float(head.max_power),
                    "frame_power": float(head.frame_power_percent),
                    "focus_power": float(head.focus_power_percent),
                    "frame_speed": float(head.frame_speed),
                    "frame_repeat_count": int(head.frame_repeat_count),
                    "frame_corner_pause": float(head.frame_corner_pause),
                    "spot_size_mm": [float(v) for v in head.spot_size_mm],
                    "focal_distance": float(head.focal_distance),
                    "pointer_offset_enabled": bool(
                        head.pointer_offset_enabled
                    ),
                    "pointer_offset_x": float(head.pointer_offset_x),
                    "pointer_offset_y": float(head.pointer_offset_y),
                }
            )
        heads.append(entry)
    return heads


def _rotary_payload(machine: Machine) -> list[dict]:
    return [
        {
            "uid": module.uid,
            "name": module.name,
            "axis": module.axis.name,
            "mode": module.mode.name,
            "mm_per_rotation": float(module.mm_per_rotation),
            "default_diameter": float(module.default_diameter),
            "max_workpiece_length": float(module.max_workpiece_length),
            "reverse_axis": bool(module.reverse_axis),
        }
        for module in machine.rotary_modules.values()
    ]


def _nogo_zones_payload(machine: Machine) -> list[dict]:
    return [zone.to_dict() for zone in machine.nogo_zones.values()]


def _wcs_payload(machine: Machine) -> list[dict]:
    return [
        {
            "name": cs.name,
            "label": cs.label,
            "offset": [float(v) for v in cs.offset],
        }
        for cs in machine.get_wcs_list()
    ]


def _machine_settings_payload(machine: Machine) -> dict:
    orientation = machine.panel.orientation
    return {
        "name": machine.name,
        "axis_extents": [float(v) for v in machine.axis_extents],
        "origin": machine.origin.value,
        "panel_orientation": (
            orientation.value
            if isinstance(orientation, PanelOrientation)
            else str(orientation)
        ),
        "work_margins": [float(v) for v in machine.work_margins],
        "reverse_x_axis": machine.reverse_x_axis,
        "reverse_y_axis": machine.reverse_y_axis,
        "reverse_z_axis": machine.reverse_z_axis,
        "soft_limits_enabled": machine.soft_limits_enabled,
        "max_travel_speed": machine.max_travel_speed,
        "max_cut_speed": machine.max_cut_speed,
        "acceleration": machine.acceleration,
        "jog_speed": machine.jog_speed,
        "dialect_uid": machine.dialect_uid,
        "heads": _heads_payload(machine),
        "rotary_modules": _rotary_payload(machine),
        "rotary_enabled_default": machine.rotary_enabled_default,
        "nogo_zones": _nogo_zones_payload(machine),
        "coordinate_systems": _wcs_payload(machine),
    }


def _require_float_list(value, length: int, message: str) -> list[float]:
    if (
        not isinstance(value, list)
        or len(value) != length
        or any(float(v) < 0 for v in value)
    ):
        raise not_found(message)
    return [float(v) for v in value]


def _apply_machine_setting(machine: Machine, key: str, value) -> None:
    if key == "name":
        text = str(value).strip()
        if not text:
            raise not_found(_("Machine name cannot be empty"))
        machine.set_name(text)
    elif key == "axis_extents":
        extents = _require_float_list(
            value,
            2,
            _("axis_extents must be two non-negative numbers"),
        )
        if extents[0] <= 0 or extents[1] <= 0:
            raise not_found(_("axis_extents must be positive"))
        machine.set_axis_extents(extents[0], extents[1])
    elif key == "origin":
        if value not in ORIGINS:
            raise not_found(
                _("Unknown origin corner: {value}").format(value=value)
            )
        machine.set_origin(Origin(value))
    elif key == "work_margins":
        margins = _require_float_list(
            value, 4, _("work_margins must contain four values")
        )
        machine.set_work_margins(*margins)
    elif key == "reverse_x_axis":
        machine.set_reverse_x_axis(bool(value))
    elif key == "reverse_y_axis":
        machine.set_reverse_y_axis(bool(value))
    elif key == "reverse_z_axis":
        machine.set_reverse_z_axis(bool(value))
    elif key == "soft_limits_enabled":
        machine.set_soft_limits_enabled(bool(value))
    elif key == "max_travel_speed":
        machine.set_max_travel_speed(_require_speed(value))
    elif key == "max_cut_speed":
        machine.set_max_cut_speed(_require_speed(value))
    elif key == "acceleration":
        machine.set_acceleration(_require_speed(value))
    elif key == "jog_speed":
        machine.set_jog_speed(_require_speed(value))
    else:
        raise not_found(_("Unknown machine setting: {key}").format(key=key))


def _require_speed(value) -> int:
    number = int(value)
    if number < 0:
        raise not_found(_("Speeds cannot be negative"))
    return number


def _apply_machine_settings(machine: Machine, values: dict) -> None:
    for key, value in values.items():
        _apply_machine_setting(machine, key, value)


@router.get("/machines/{machine_id}/settings")
async def get_machine_settings(request: Request, machine_id: str) -> dict:
    bridge = get_bridge(request)

    def _settings(editor) -> dict:
        machine = _get_machine(editor.context, machine_id)
        return _machine_settings_payload(machine)

    return await bridge.run(_settings, bridge.editor)


@router.put("/machines/{machine_id}/settings")
async def put_machine_settings(
    request: Request, machine_id: str, patch: MachineSettingsPatch
) -> dict:
    bridge = get_bridge(request)

    def _apply(editor) -> dict:
        machine = _get_machine(editor.context, machine_id)
        _apply_machine_settings(machine, patch.values)
        return _machine_settings_payload(machine)

    return await bridge.run(_apply, bridge.editor)


def _driver_setup_payload(machine: Machine) -> dict:
    driver_cls = get_driver_cls(machine.driver_name or "")
    setup_varset = driver_cls.get_setup_vars()
    values = dict(machine.driver_args)
    for var in setup_varset:
        values.setdefault(var.key, var.default)
    device_settings = None
    if getattr(driver_cls, "supports_settings", False):
        instance = machine.driver
        if not isinstance(instance, driver_cls):
            instance = driver_cls.__new__(driver_cls)
        try:
            device_settings = [
                {
                    "title": varset.title,
                    "description": varset.description,
                    "fields": [serialize_var(var) for var in varset],
                }
                for varset in instance.get_setting_vars()
            ]
        except Exception:  # noqa: BLE001 - drivers may not initialize
            device_settings = None
    return {
        "driver": driver_cls.__name__,
        "setup": {
            "title": setup_varset.title,
            "fields": [serialize_var(var) for var in setup_varset],
        },
        "values": values,
        "precheck_error": machine.precheck_error,
        "device_settings": device_settings,
    }


@router.get("/drivers")
async def get_drivers() -> list[dict]:
    return [
        {
            "name": driver_cls.__name__,
            "maturity": driver_cls.maturity.name.lower(),
            "maturity_warning": DRIVER_MATURITY_LABELS.get(
                driver_cls.maturity, ""
            ),
        }
        for driver_cls in drivers
    ]


@router.get("/machines/{machine_id}/driver-setup")
async def get_driver_setup(request: Request, machine_id: str) -> dict:
    bridge = get_bridge(request)

    def _payload(editor) -> dict:
        machine = _get_machine(editor.context, machine_id)
        return _driver_setup_payload(machine)

    return await bridge.run(_payload, bridge.editor)


def _coerce_setup_value(var, value):
    if value is None:
        return None
    if var.var_type is bool:
        return bool(value)
    if var.var_type is int:
        return int(value)
    if var.var_type is float:
        return float(value)
    return str(value)


@router.put("/machines/{machine_id}/driver-setup")
async def put_driver_setup(
    request: Request, machine_id: str, patch: DriverSetupPatch
) -> dict:
    bridge = get_bridge(request)

    def _apply(editor) -> dict:
        machine = _get_machine(editor.context, machine_id)
        setup_vars = get_driver_cls(machine.driver_name or "").get_setup_vars()
        unknown = set(patch.values) - set(setup_vars.keys())
        if unknown:
            raise not_found(
                _("Unknown driver setting: {key}").format(key=min(unknown))
            )
        merged = dict(machine.driver_args)
        for key, value in patch.values.items():
            merged[key] = _coerce_setup_value(setup_vars[key], value)
        machine.set_driver_args(merged)
        return machine.driver_args

    await bridge.run(_apply, bridge.editor)

    def _rebuild(editor) -> None:
        controller = editor.context.machine_mgr.get_controller(machine_id)
        return controller.rebuild_driver()

    await bridge.run(_rebuild, bridge.editor)

    def _payload(editor) -> dict:
        machine = _get_machine(editor.context, machine_id)
        return _driver_setup_payload(machine)

    return await bridge.run(_payload, bridge.editor)


@router.put("/machines/{machine_id}/driver")
async def put_driver(
    request: Request, machine_id: str, payload: DriverSwitchRequest
) -> dict:
    bridge = get_bridge(request)
    driver_cls = driver_by_classname.get(payload.driver)
    if driver_cls is None:
        raise not_found(
            _("Unknown driver: {name}").format(name=payload.driver)
        )

    def _apply(editor) -> None:
        machine = _get_machine(editor.context, machine_id)
        machine.set_driver(driver_cls, {})

    await bridge.run(_apply, bridge.editor)

    def _rebuild(editor) -> None:
        controller = editor.context.machine_mgr.get_controller(machine_id)
        return controller.rebuild_driver()

    await bridge.run(_rebuild, bridge.editor)

    def _payload(editor) -> dict:
        machine = _get_machine(editor.context, machine_id)
        return _driver_setup_payload(machine)

    return await bridge.run(_payload, bridge.editor)


def _apply_head_patch(machine: Machine, patch: dict) -> None:
    head = next(
        (
            candidate
            for candidate in machine.heads
            if candidate.uid == patch.get("uid")
        ),
        None,
    )
    if head is None:
        raise not_found(_("Laser head not found"))
    if not isinstance(head, LaserHead):
        raise not_found(_("Only laser heads have editable settings"))
    attr_map = {
        "frame_power": "frame_power_percent",
        "focus_power": "focus_power_percent",
    }
    for key, kind in HEAD_SETTINGS.items():
        if key not in patch:
            continue
        attr = attr_map.get(key, key)
        if not hasattr(head, attr):
            raise not_found(_("Unknown head setting: {key}").format(key=key))
        value = patch[key]
        setattr(head, attr, bool(value) if kind is bool else float(value))
    if "spot_size_mm" in patch:
        spot = patch["spot_size_mm"]
        if not isinstance(spot, list) or len(spot) != 2:
            raise not_found(_("spot_size_mm must contain two values"))
        head.spot_size_mm = (float(spot[0]), float(spot[1]))


@router.put("/machines/{machine_id}/heads")
async def put_heads(
    request: Request, machine_id: str, patch: HeadsPatch
) -> list[dict]:
    bridge = get_bridge(request)

    def _apply(editor) -> list[dict]:
        machine = _get_machine(editor.context, machine_id)
        for head_patch in patch.heads:
            _apply_head_patch(machine, head_patch)
        machine.changed.send(machine)
        return _heads_payload(machine)

    return await bridge.run(_apply, bridge.editor)


def _macro_payload(macro: Macro) -> dict:
    return {
        "uid": macro.uid,
        "name": macro.name,
        "code": macro.code,
        "enabled": macro.enabled,
    }


def _code_lines(code: str) -> list[str]:
    return [line for line in code.replace("\r", "").split("\n")]


@router.get("/machines/{machine_id}/macros")
async def get_macros(request: Request, machine_id: str) -> dict:
    bridge = get_bridge(request)

    def _list(editor) -> dict:
        machine = _get_machine(editor.context, machine_id)
        return {
            "macros": [
                _macro_payload(macro) for macro in machine.macros.values()
            ]
        }

    return await bridge.run(_list, bridge.editor)


@router.post("/machines/{machine_id}/macros")
async def create_macro(
    request: Request, machine_id: str, payload: MacroRequest
) -> dict:
    bridge = get_bridge(request)

    def _create(editor) -> dict:
        machine = _get_machine(editor.context, machine_id)
        macro = Macro(
            name=payload.name or _("Unnamed Macro"),
            code=_code_lines(payload.code),
            enabled=payload.enabled,
        )
        machine.add_macro(macro)
        return _macro_payload(macro)

    return await bridge.run(_create, bridge.editor)


@router.put("/machines/{machine_id}/macros/{macro_uid}")
async def update_macro(
    request: Request, machine_id: str, macro_uid: str, patch: MacroPatch
) -> dict:
    bridge = get_bridge(request)

    def _update(editor) -> dict:
        machine = _get_machine(editor.context, machine_id)
        macro = machine.macros.get(macro_uid)
        if macro is None:
            raise not_found(_("Macro not found"))
        if patch.name is not None:
            macro.name = patch.name
        if patch.code is not None:
            macro.code = _code_lines(patch.code)
        if patch.enabled is not None:
            macro.enabled = patch.enabled
        machine.changed.send(machine)
        return _macro_payload(macro)

    return await bridge.run(_update, bridge.editor)


@router.delete("/machines/{machine_id}/macros/{macro_uid}")
async def delete_macro(
    request: Request, machine_id: str, macro_uid: str
) -> dict:
    bridge = get_bridge(request)

    def _delete(editor) -> dict:
        machine = _get_machine(editor.context, machine_id)
        if macro_uid not in machine.macros:
            raise not_found(_("Macro not found"))
        machine.remove_macro(macro_uid)
        return {"deleted": macro_uid}

    return await bridge.run(_delete, bridge.editor)


@router.put("/machines/{machine_id}/wcs/{slot}")
async def put_wcs(
    request: Request, machine_id: str, slot: str, patch: WcsPatch
) -> dict:
    bridge = get_bridge(request)

    def _apply(editor) -> dict:
        machine = _get_machine(editor.context, machine_id)
        if slot not in machine.coordinate_systems:
            raise not_found(_("Unknown work offset: {slot}").format(slot=slot))
        if len(patch.offset) != 3:
            raise not_found(_("offset must contain three values"))
        offset = (
            float(patch.offset[0]),
            float(patch.offset[1]),
            float(patch.offset[2]),
        )
        machine.update_wcs_offset(slot, offset)
        if patch.label is not None:
            machine.set_wcs_label(slot, patch.label)
        else:
            machine.changed.send(machine)
        cs = machine.coordinate_systems[slot]
        return {
            "name": cs.name,
            "label": cs.label,
            "offset": [float(v) for v in cs.offset],
        }

    return await bridge.run(_apply, bridge.editor)


@router.post("/machines/{machine_id}/nogo-zones")
async def create_nogo_zone(
    request: Request, machine_id: str, payload: NogoZoneRequest
) -> dict:
    bridge = get_bridge(request)

    def _create(editor) -> dict:
        machine = _get_machine(editor.context, machine_id)
        try:
            shape = ZoneShape(payload.shape)
        except ValueError:
            raise not_found(
                _("Unknown zone shape: {shape}").format(shape=payload.shape)
            ) from None
        zone = Zone()
        if payload.name:
            zone.set_name(payload.name)
        zone.set_shape(shape)
        for key, value in payload.params.items():
            zone.set_param(key, float(value))
        zone.set_enabled(payload.enabled)
        machine.add_nogo_zone(zone)
        return zone.to_dict()

    return await bridge.run(_create, bridge.editor)


@router.delete("/machines/{machine_id}/nogo-zones/{zone_uid}")
async def delete_nogo_zone(
    request: Request, machine_id: str, zone_uid: str
) -> dict:
    bridge = get_bridge(request)

    def _delete(editor) -> dict:
        machine = _get_machine(editor.context, machine_id)
        zone = machine.get_nogo_zone_by_uid(zone_uid)
        if zone is None:
            raise not_found(_("No-go zone not found"))
        machine.remove_nogo_zone(zone)
        return {"deleted": zone_uid}

    return await bridge.run(_delete, bridge.editor)


@router.put("/machines/{machine_id}/rotary-modules/{module_uid}")
async def put_rotary_module(
    request: Request,
    machine_id: str,
    module_uid: str,
    patch: RotaryModulePatch,
) -> list[dict]:
    bridge = get_bridge(request)

    def _apply(editor) -> list[dict]:
        machine = _get_machine(editor.context, machine_id)
        module = machine.rotary_modules.get(module_uid)
        if module is None:
            raise not_found(_("Rotary module not found"))
        if patch.name is not None:
            module.set_name(patch.name)
        if patch.default_diameter is not None:
            if patch.default_diameter <= 0:
                raise not_found(_("Diameter must be positive"))
            module.default_diameter = float(patch.default_diameter)
        if patch.mm_per_rotation is not None:
            module.set_mm_per_rotation(float(patch.mm_per_rotation))
        if patch.reverse_axis is not None:
            module.reverse_axis = bool(patch.reverse_axis)
        machine.changed.send(machine)
        return _rotary_payload(machine)

    return await bridge.run(_apply, bridge.editor)
