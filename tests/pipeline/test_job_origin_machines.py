"""
Start From on machine setups other than a plain bottom-left GRBL bed.

Each case pins what reaches the controller for one of the special cases
of the coordinate model: every origin corner, negative (reversed) axes,
"machine origin is work area origin" with margins, a rotated panel,
layers with their own WCS and rotary layers.

A 10 x 10 mm square is used throughout. The anchor is chosen as it is
seen on the canvas, so "bottom left" means the bottom-left corner of
the design on screen, whatever the machine calls its axes.
"""

import re
from pathlib import Path

import pytest
import pytest_asyncio

from rayforge.context import get_context
from rayforge.core.job_origin import JobAnchor, JobOrigin, StartFrom
from rayforge.core.vectorization_spec import PassthroughSpec
from rayforge.machine.cmd import MachineCmd
from rayforge.machine.driver.driver import DeviceState, DeviceStatus
from rayforge.machine.job_placement import JobPlacementError
from rayforge.machine.models.machine import Origin
from rayforge.machine.models.machine_panel import PanelOrientation
from rayforge.machine.transport import TransportStatus
from rayforge.pipeline.artifact import JobArtifact

ASSETS = Path(__file__).parent.parent / "doceditor" / "assets"
MARGINS = (10.0, 20.0, 30.0, 40.0)
WCS_OFFSET = (100.0, 50.0)
NEGATIVE_WCS_OFFSET = (-120.0, -80.0)
HEAD = (200.0, 150.0)
NEGATIVE_HEAD = (-200.0, -150.0)

# Command-space extents of the square for User Origin with the
# bottom-left anchor, per (origin, reversed axes). The anchor always
# lands on X0 Y0; the square extends in the direction in which the
# machine's coordinates grow towards the top right of the canvas.
USER_ORIGIN_BOTTOM_LEFT = {
    (Origin.BOTTOM_LEFT, False): (0.0, 0.0, 10.0, 10.0),
    (Origin.BOTTOM_LEFT, True): (-10.0, -10.0, 0.0, 0.0),
    (Origin.TOP_LEFT, False): (0.0, -10.0, 10.0, 0.0),
    (Origin.TOP_LEFT, True): (-10.0, 0.0, 0.0, 10.0),
    (Origin.BOTTOM_RIGHT, False): (-10.0, 0.0, 0.0, 10.0),
    (Origin.BOTTOM_RIGHT, True): (0.0, -10.0, 10.0, 0.0),
    (Origin.TOP_RIGHT, False): (-10.0, -10.0, 0.0, 0.0),
    (Origin.TOP_RIGHT, True): (0.0, 0.0, 10.0, 10.0),
}

# The command offset (the zero point in machine coordinates) when the
# machine origin is the work area origin, per origin corner, for the
# margins (left, top, right, bottom) = MARGINS. Reversed axes negate it.
WORKAREA_ZERO = {
    Origin.BOTTOM_LEFT: (10.0, 40.0),
    Origin.TOP_LEFT: (10.0, 20.0),
    Origin.BOTTOM_RIGHT: (30.0, 40.0),
    Origin.TOP_RIGHT: (30.0, 20.0),
}

ORIGINS = list(WORKAREA_ZERO)


def _configure(machine, origin, reversed_axes, workarea_origin):
    machine.set_origin(origin)
    machine.set_reverse_x_axis(reversed_axes)
    machine.set_reverse_y_axis(reversed_axes)
    machine.set_work_margins(*MARGINS)
    machine.set_wcs_origin_is_workarea_origin(workarea_origin)
    off_x, off_y = NEGATIVE_WCS_OFFSET if reversed_axes else WCS_OFFSET
    machine.update_wcs_offset("G54", (off_x, off_y, 0.0))
    machine.changed.send(machine)


def _zero_point(origin, reversed_axes, workarea_origin):
    """The active zero point in (signed) machine coordinates."""
    if workarea_origin:
        x, y = WORKAREA_ZERO[origin]
        return (-x, -y) if reversed_axes else (x, y)
    return NEGATIVE_WCS_OFFSET if reversed_axes else WCS_OFFSET


def _set_head(machine, x, y):
    machine.set_connection_status(TransportStatus.CONNECTED)
    machine.set_device_state(
        DeviceState(status=DeviceStatus.IDLE, machine_pos=(x, y, 0.0))
    )


def _motion_extents(gcode: str) -> tuple[float, float, float, float]:
    xs: list[float] = []
    ys: list[float] = []
    for line in gcode.splitlines():
        if "Return to origin" in line:
            continue
        code = line.split(";")[0]
        if not re.match(r"\s*G[01]\b", code):
            continue
        for axis, value in re.findall(r"([XY])([-+]?\d*\.?\d+)", code):
            (xs if axis == "X" else ys).append(float(value))
    return min(xs), min(ys), max(xs), max(ys)


def _center(rect):
    """The center of a world-space (min_x, min_y, max_x, max_y) box."""
    return ((rect[0] + rect[2]) / 2, (rect[1] + rect[3]) / 2)


@pytest_asyncio.fixture
async def editor(doc_editor, context_initializer, contour_step_class):
    machine = get_context().machine
    assert machine is not None
    machine.set_dialect_uid("grbl")
    machine.set_axis_extents(400, 400)
    machine.set_active_wcs("G54")

    step = contour_step_class.create(
        context_initializer, name="Contour", optimize=False
    )
    step.set_power(0.5)
    step.set_cut_speed(3000)
    workflow = doc_editor.doc.active_layer.workflow
    assert workflow is not None
    workflow.add_step(step)

    await doc_editor.import_file_from_path(
        ASSETS / "10x10_square.svg",
        mime_type="image/svg+xml",
        vectorization_spec=PassthroughSpec(),
    )
    await doc_editor.wait_until_settled()
    yield doc_editor, machine
    await doc_editor.wait_until_settled()


async def _generate(doc_editor) -> tuple[str, tuple]:
    await doc_editor.wait_until_settled()
    pipeline = doc_editor.pipeline
    pipeline.invalidate_job_output()
    handle = await pipeline.generate_job_artifact_async()
    assert handle is not None
    with pipeline.artifact_store.checkout_handle(handle) as artifact:
        assert isinstance(artifact, JobArtifact)
        assert artifact.machine_code is not None
        result = artifact.machine_code, artifact.ops.rect()
    await doc_editor.wait_until_settled()
    return result


# ----------------------------------------------------------------------
# User Origin
# ----------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.parametrize("workarea_origin", [False, True])
@pytest.mark.parametrize("reversed_axes", [False, True])
@pytest.mark.parametrize("origin", ORIGINS)
async def test_user_origin_bottom_left_on_every_setup(
    editor, origin, reversed_axes, workarea_origin
):
    doc_editor, machine = editor
    _configure(machine, origin, reversed_axes, workarea_origin)
    doc_editor.doc.set_job_origin(
        JobOrigin(StartFrom.USER_ORIGIN, JobAnchor.BOTTOM_LEFT)
    )
    gcode, _rect = await _generate(doc_editor)
    expected = USER_ORIGIN_BOTTOM_LEFT[(origin, reversed_axes)]
    assert _motion_extents(gcode) == pytest.approx(expected, abs=1e-3)


@pytest.mark.asyncio
@pytest.mark.parametrize("workarea_origin", [False, True])
@pytest.mark.parametrize("reversed_axes", [False, True])
@pytest.mark.parametrize("origin", ORIGINS)
async def test_user_origin_center_lands_on_zero_point(
    editor, origin, reversed_axes, workarea_origin
):
    doc_editor, machine = editor
    _configure(machine, origin, reversed_axes, workarea_origin)
    doc_editor.doc.set_job_origin(
        JobOrigin(StartFrom.USER_ORIGIN, JobAnchor.CENTER)
    )
    gcode, rect = await _generate(doc_editor)
    assert _motion_extents(gcode) == pytest.approx(
        (-5.0, -5.0, 5.0, 5.0), abs=1e-3
    )
    zero = _zero_point(origin, reversed_axes, workarea_origin)
    preview_center = machine.panel.world_point_to_machine(*_center(rect))
    assert preview_center == pytest.approx(zero, abs=1e-3)


# ----------------------------------------------------------------------
# Current Position
# ----------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.parametrize("workarea_origin", [False, True])
@pytest.mark.parametrize("reversed_axes", [False, True])
@pytest.mark.parametrize("origin", ORIGINS)
async def test_current_position_center_lands_on_head(
    editor, origin, reversed_axes, workarea_origin
):
    doc_editor, machine = editor
    _configure(machine, origin, reversed_axes, workarea_origin)
    head = NEGATIVE_HEAD if reversed_axes else HEAD
    _set_head(machine, *head)
    doc_editor.doc.set_job_origin(
        JobOrigin(StartFrom.CURRENT_POSITION, JobAnchor.CENTER)
    )
    gcode, rect = await _generate(doc_editor)
    preview_center = machine.panel.world_point_to_machine(*_center(rect))
    assert preview_center == pytest.approx(head, abs=1e-3)
    zero = _zero_point(origin, reversed_axes, workarea_origin)
    cx, cy = head[0] - zero[0], head[1] - zero[1]
    assert _motion_extents(gcode) == pytest.approx(
        (cx - 5.0, cy - 5.0, cx + 5.0, cy + 5.0), abs=1e-3
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("workarea_origin", [False, True])
@pytest.mark.parametrize("origin", ORIGINS)
async def test_frame_and_return_on_negative_axes(
    editor, mocker, origin, workarea_origin
):
    doc_editor, machine = editor
    _configure(machine, origin, True, workarea_origin)
    _set_head(machine, *NEGATIVE_HEAD)
    doc_editor.doc.set_job_origin(
        JobOrigin(StartFrom.CURRENT_POSITION, JobAnchor.CENTER)
    )
    await doc_editor.wait_until_settled()
    frame_spy = mocker.spy(machine.driver, "frame")
    move_spy = mocker.spy(machine.driver, "move_to")

    await MachineCmd(doc_editor).frame_job(machine)
    await doc_editor.wait_until_settled()

    zero = _zero_point(origin, True, workarea_origin)
    cx = NEGATIVE_HEAD[0] - zero[0]
    cy = NEGATIVE_HEAD[1] - zero[1]
    corners = frame_spy.call_args.args[0]
    xs = [c[0] for c in corners]
    ys = [c[1] for c in corners]
    assert (min(xs), min(ys), max(xs), max(ys)) == pytest.approx(
        (cx - 5.0, cy - 5.0, cx + 5.0, cy + 5.0), abs=1e-3
    )
    move_spy.assert_called_once()
    assert move_spy.call_args.args[:2] == pytest.approx((cx, cy))


# ----------------------------------------------------------------------
# Rotated panel: the anchor is the one seen on screen
# ----------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "orientation, anchor, expected",
    [
        (PanelOrientation.NATIVE, JobAnchor.TOP_LEFT, (0, -10, 10, 0)),
        (PanelOrientation.ROTATED_LEFT, JobAnchor.TOP_LEFT, (-10, -10, 0, 0)),
        (
            PanelOrientation.ROTATED_LEFT,
            JobAnchor.BOTTOM_LEFT,
            (0, -10, 10, 0),
        ),
        (PanelOrientation.ROTATED_RIGHT, JobAnchor.TOP_LEFT, (0, 0, 10, 10)),
        (
            PanelOrientation.ROTATED_RIGHT,
            JobAnchor.BOTTOM_RIGHT,
            (-10, -10, 0, 0),
        ),
    ],
)
async def test_rotated_panel_uses_anchor_as_seen_on_screen(
    editor, orientation, anchor, expected
):
    doc_editor, machine = editor
    _configure(machine, Origin.BOTTOM_LEFT, False, False)
    machine.set_panel_orientation(orientation, align_cameras=False)
    doc_editor.doc.set_job_origin(JobOrigin(StartFrom.USER_ORIGIN, anchor))
    gcode, _rect = await _generate(doc_editor)
    assert _motion_extents(gcode) == pytest.approx(expected, abs=1e-3)


# ----------------------------------------------------------------------
# Combinations that are refused instead of guessed
# ----------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "start_from", [StartFrom.USER_ORIGIN, StartFrom.CURRENT_POSITION]
)
async def test_layer_with_own_wcs_is_refused(editor, start_from):
    doc_editor, machine = editor
    _configure(machine, Origin.BOTTOM_LEFT, False, False)
    _set_head(machine, *HEAD)
    machine.update_wcs_offset("G55", (30.0, 20.0, 0.0))
    doc_editor.doc.active_layer.set_wcs("G55")
    doc_editor.doc.set_job_origin(JobOrigin(start_from, JobAnchor.CENTER))
    await doc_editor.wait_until_settled()
    with pytest.raises(JobPlacementError, match="G55"):
        await doc_editor.pipeline.generate_job_artifact_async()


@pytest.mark.asyncio
async def test_layer_naming_the_active_wcs_is_accepted(editor):
    doc_editor, machine = editor
    _configure(machine, Origin.BOTTOM_LEFT, False, False)
    doc_editor.doc.active_layer.set_wcs("G54")
    doc_editor.doc.set_job_origin(
        JobOrigin(StartFrom.USER_ORIGIN, JobAnchor.CENTER)
    )
    gcode, _rect = await _generate(doc_editor)
    assert _motion_extents(gcode) == pytest.approx(
        (-5.0, -5.0, 5.0, 5.0), abs=1e-3
    )


@pytest.mark.asyncio
async def test_layer_wcs_is_ignored_when_workarea_is_origin(editor):
    doc_editor, machine = editor
    _configure(machine, Origin.BOTTOM_LEFT, False, True)
    machine.update_wcs_offset("G55", (30.0, 20.0, 0.0))
    doc_editor.doc.active_layer.set_wcs("G55")
    doc_editor.doc.set_job_origin(
        JobOrigin(StartFrom.USER_ORIGIN, JobAnchor.CENTER)
    )
    gcode, _rect = await _generate(doc_editor)
    assert _motion_extents(gcode) == pytest.approx(
        (-5.0, -5.0, 5.0, 5.0), abs=1e-3
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "start_from", [StartFrom.USER_ORIGIN, StartFrom.CURRENT_POSITION]
)
async def test_rotary_layer_is_refused_in_every_mode(editor, start_from):
    doc_editor, machine = editor
    _configure(machine, Origin.BOTTOM_LEFT, False, False)
    _set_head(machine, *HEAD)
    doc_editor.doc.active_layer.set_rotary_enabled(True)
    doc_editor.doc.set_job_origin(JobOrigin(start_from, JobAnchor.CENTER))
    await doc_editor.wait_until_settled()
    with pytest.raises(JobPlacementError, match="rotary"):
        await doc_editor.pipeline.generate_job_artifact_async()
