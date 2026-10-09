"""
Golden G-code tests for the job origin ("Start From") setting.

A 10 x 10 mm square is imported at world (0, 0) on a bottom-left GRBL
machine whose G54 zero sits at machine (100, 50). The same document is
exported in every Start From mode; the expected files show exactly
which coordinates reach the controller.
"""

import asyncio
import re
from pathlib import Path

import pytest
import pytest_asyncio

from rayforge.context import get_context
from rayforge.core.job_origin import JobAnchor, JobOrigin, StartFrom
from rayforge.core.vectorization_spec import TraceSpec
from rayforge.machine.cmd import MachineCmd
from rayforge.machine.driver.driver import DeviceState, DeviceStatus
from rayforge.machine.job_placement import JobPlacementError
from rayforge.machine.models.machine import Origin
from rayforge.machine.models.zone import Zone
from rayforge.machine.transport import TransportStatus
from rayforge.pipeline.artifact import JobArtifact

ASSETS = Path(__file__).parent.parent / "doceditor" / "assets"
DATA = Path(__file__).parent / "data" / "job_origin"
G54_OFFSET = (100.0, 50.0, 0.0)
HEAD_POS = (200.0, 150.0, 0.0)


def _motion_extents(gcode: str) -> tuple[float, float, float, float]:
    xs: list[float] = []
    ys: list[float] = []
    for line in gcode.splitlines():
        code = line.split(";")[0]
        if not re.match(r"\s*G[01]\b", code):
            continue
        for axis, value in re.findall(r"([XY])([-+]?\d*\.?\d+)", code):
            (xs if axis == "X" else ys).append(float(value))
    return min(xs), min(ys), max(xs), max(ys)


def _job_motion(gcode: str) -> str:
    """The G-code without the dialect postscript's return move."""
    return "\n".join(
        line for line in gcode.splitlines() if "Return to origin" not in line
    )


@pytest_asyncio.fixture
async def square_doc(doc_editor, context_initializer, contour_step_class):
    machine = get_context().machine
    assert machine is not None
    machine.set_dialect_uid("grbl")
    machine.set_origin(Origin.BOTTOM_LEFT)
    machine.set_axis_extents(400, 400)
    machine.set_active_wcs("G54")
    machine.update_wcs_offset("G54", G54_OFFSET)

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
        vectorization_spec=TraceSpec(),
    )
    await doc_editor.wait_until_settled()
    return doc_editor, machine


def _set_head(machine, x, y):
    machine.set_connection_status(TransportStatus.CONNECTED)
    machine.set_device_state(
        DeviceState(status=DeviceStatus.IDLE, machine_pos=(x, y, 0.0))
    )


async def _generate(editor) -> tuple[str, tuple]:
    await editor.wait_until_settled()
    pipeline = editor.pipeline
    handle = await pipeline.generate_job_artifact_async()
    assert handle is not None
    with pipeline.artifact_store.checkout_handle(handle) as artifact:
        assert isinstance(artifact, JobArtifact)
        assert artifact.machine_code is not None
        result = artifact.machine_code, artifact.ops.rect()
    await editor.wait_until_settled()
    return result


def _assert_golden(gcode: str, name: str):
    expected = (DATA / name).read_text(encoding="utf-8")
    assert gcode.strip().splitlines() == expected.strip().splitlines()


@pytest.mark.asyncio
async def test_absolute_is_unchanged(square_doc):
    editor, _machine = square_doc
    gcode, rect = await _generate(editor)
    # Imported items land on the G54 zero; absolute output is relative
    # to G54 as before.
    assert _motion_extents(_job_motion(gcode)) == pytest.approx(
        (0.0, 0.0, 10.0, 10.0)
    )
    assert rect == pytest.approx((100.0, 50.0, 110.0, 60.0), abs=0.06)
    _assert_golden(gcode, "absolute.gcode")


@pytest.mark.asyncio
async def test_user_origin_center(square_doc):
    editor, _machine = square_doc
    editor.doc.set_job_origin(
        JobOrigin(StartFrom.USER_ORIGIN, JobAnchor.CENTER)
    )
    gcode, rect = await _generate(editor)
    # Centred on the G54 zero: the coordinates are WCS-relative and do
    # not depend on the stored offset value.
    assert _motion_extents(_job_motion(gcode)) == pytest.approx(
        (-5.0, -5.0, 5.0, 5.0)
    )
    assert rect == pytest.approx((95.0, 45.0, 105.0, 55.0), abs=0.06)
    _assert_golden(gcode, "user_origin_center.gcode")


@pytest.mark.asyncio
async def test_user_origin_output_is_independent_of_offset(square_doc):
    editor, machine = square_doc
    editor.doc.set_job_origin(
        JobOrigin(StartFrom.USER_ORIGIN, JobAnchor.TOP_RIGHT)
    )
    first, _rect = await _generate(editor)
    machine.update_wcs_offset("G54", (250.0, 300.0, 0.0))
    machine.changed.send(machine)
    editor.pipeline.invalidate_job_output()
    second, rect = await _generate(editor)
    assert first == second
    assert rect == pytest.approx((240.0, 290.0, 250.0, 300.0), abs=0.06)
    _assert_golden(first, "user_origin_top_right.gcode")


@pytest.mark.asyncio
async def test_current_position_bottom_left(square_doc):
    editor, machine = square_doc
    _set_head(machine, *HEAD_POS[:2])
    editor.doc.set_job_origin(
        JobOrigin(StartFrom.CURRENT_POSITION, JobAnchor.BOTTOM_LEFT)
    )
    gcode, rect = await _generate(editor)
    assert _motion_extents(_job_motion(gcode)) == pytest.approx(
        (100.0, 100.0, 110.0, 110.0)
    )
    assert rect == pytest.approx((200.0, 150.0, 210.0, 160.0), abs=0.06)
    _assert_golden(gcode, "current_position_bottom_left.gcode")


@pytest.mark.asyncio
async def test_current_position_reads_head_at_generation(square_doc):
    editor, machine = square_doc
    editor.doc.set_job_origin(
        JobOrigin(StartFrom.CURRENT_POSITION, JobAnchor.CENTER)
    )
    _set_head(machine, 150.0, 150.0)
    _first, rect_a = await _generate(editor)
    _set_head(machine, 300.0, 200.0)
    _second, rect_b = await _generate(editor)
    assert rect_a == pytest.approx((145.0, 145.0, 155.0, 155.0), abs=0.06)
    assert rect_b == pytest.approx((295.0, 195.0, 305.0, 205.0), abs=0.06)


@pytest.mark.asyncio
async def test_current_position_without_machine_is_refused(square_doc):
    editor, _machine = square_doc
    editor.doc.set_job_origin(
        JobOrigin(StartFrom.CURRENT_POSITION, JobAnchor.CENTER)
    )
    await editor.wait_until_settled()
    with pytest.raises(JobPlacementError):
        await editor.pipeline.generate_job_artifact_async()


@pytest.mark.asyncio
async def test_back_to_absolute_restores_output(square_doc):
    editor, _machine = square_doc
    editor.doc.set_job_origin(
        JobOrigin(StartFrom.USER_ORIGIN, JobAnchor.CENTER)
    )
    await _generate(editor)
    editor.doc.set_job_origin(JobOrigin())
    gcode, _rect = await _generate(editor)
    _assert_golden(gcode, "absolute.gcode")


@pytest.mark.asyncio
async def test_rotary_layer_refuses_relative_start(square_doc):
    editor, _machine = square_doc
    editor.doc.active_layer.set_rotary_enabled(True)
    editor.doc.set_job_origin(
        JobOrigin(StartFrom.USER_ORIGIN, JobAnchor.CENTER)
    )
    await editor.wait_until_settled()
    with pytest.raises(JobPlacementError):
        await editor.pipeline.generate_job_artifact_async()


# ----------------------------------------------------------------------
# Framing and sending
# ----------------------------------------------------------------------


def _frame_extents(corners) -> tuple[float, float, float, float]:
    xs = [c[0] for c in corners]
    ys = [c[1] for c in corners]
    return min(xs), min(ys), max(xs), max(ys)


@pytest.mark.asyncio
async def test_frame_current_position_and_return_to_start(square_doc, mocker):
    editor, machine = square_doc
    _set_head(machine, *HEAD_POS[:2])
    editor.doc.set_job_origin(
        JobOrigin(StartFrom.CURRENT_POSITION, JobAnchor.CENTER)
    )
    await editor.wait_until_settled()
    frame_spy = mocker.spy(machine.driver, "frame")
    move_spy = mocker.spy(machine.driver, "move_to")

    await MachineCmd(editor).frame_job(machine)
    await editor.wait_until_settled()

    frame_spy.assert_called_once()
    corners = frame_spy.call_args.args[0]
    # Centred on the head (200, 150), in G54 command coordinates.
    assert _frame_extents(corners) == pytest.approx(
        (95.0, 95.0, 105.0, 105.0), abs=0.06
    )
    # The postscript returns to X0 Y0; the head is sent back to where
    # it started, so a following Start lands in the same place.
    move_spy.assert_called_once()
    assert move_spy.call_args.args[:2] == pytest.approx((100.0, 100.0))


@pytest.mark.asyncio
async def test_frame_user_origin_does_not_move_head_back(square_doc, mocker):
    editor, machine = square_doc
    _set_head(machine, *HEAD_POS[:2])
    editor.doc.set_job_origin(
        JobOrigin(StartFrom.USER_ORIGIN, JobAnchor.BOTTOM_LEFT)
    )
    await editor.wait_until_settled()
    frame_spy = mocker.spy(machine.driver, "frame")
    move_spy = mocker.spy(machine.driver, "move_to")

    await MachineCmd(editor).frame_job(machine)
    await editor.wait_until_settled()

    corners = frame_spy.call_args.args[0]
    assert _frame_extents(corners) == pytest.approx(
        (0.0, 0.0, 10.0, 10.0), abs=0.06
    )
    move_spy.assert_not_called()


@pytest.mark.asyncio
async def test_send_current_position_runs_placed_job(square_doc, mocker):
    editor, machine = square_doc
    _set_head(machine, *HEAD_POS[:2])
    editor.doc.set_job_origin(
        JobOrigin(StartFrom.CURRENT_POSITION, JobAnchor.BOTTOM_LEFT)
    )
    await editor.wait_until_settled()
    run_spy = mocker.spy(machine.driver, "run")
    move_spy = mocker.spy(machine.driver, "move_to")

    await MachineCmd(editor).send_job(machine)
    await editor.wait_until_settled()

    run_spy.assert_called_once()
    _assert_golden(
        run_spy.call_args.args[0].text, "current_position_bottom_left.gcode"
    )
    move_spy.assert_called_once()
    assert move_spy.call_args.args[:2] == pytest.approx((100.0, 100.0))


@pytest.mark.asyncio
async def test_cancelled_job_does_not_return_to_start(square_doc, mocker):
    editor, machine = square_doc
    _set_head(machine, *HEAD_POS[:2])
    editor.doc.set_job_origin(
        JobOrigin(StartFrom.CURRENT_POSITION, JobAnchor.BOTTOM_LEFT)
    )
    await editor.wait_until_settled()
    cmd = MachineCmd(editor)
    original_run = machine.driver.run

    async def run_and_cancel(*args, **kwargs):
        cmd.cancel_job(machine)
        await original_run(*args, **kwargs)

    mocker.patch.object(machine.driver, "run", side_effect=run_and_cancel)
    move_spy = mocker.spy(machine.driver, "move_to")

    await cmd.send_job(machine)
    await editor.wait_until_settled()

    move_spy.assert_not_called()


@pytest.mark.asyncio
async def test_send_refuses_job_beyond_machine_extents(square_doc, mocker):
    editor, machine = square_doc
    _set_head(machine, 395.0, 395.0)
    editor.doc.set_job_origin(
        JobOrigin(StartFrom.CURRENT_POSITION, JobAnchor.BOTTOM_LEFT)
    )
    await editor.wait_until_settled()
    run_spy = mocker.spy(machine.driver, "run")
    messages: list[str] = []
    editor.notification_requested.connect(
        lambda sender, message, **kw: messages.append(message), weak=False
    )

    with pytest.raises(JobPlacementError):
        await MachineCmd(editor).send_job(machine)
    await editor.wait_until_settled()

    run_spy.assert_not_called()
    assert messages and "X=404.9 > 400.0" in messages[-1]


@pytest.mark.asyncio
async def test_frame_refuses_job_in_nogo_zone(square_doc, mocker):
    editor, machine = square_doc
    zone = Zone()
    zone.set_name("Clamp")
    zone.params.update({"x": 195.0, "y": 145.0, "w": 10.0, "h": 10.0})
    machine.add_nogo_zone(zone)
    _set_head(machine, *HEAD_POS[:2])
    editor.doc.set_job_origin(
        JobOrigin(StartFrom.CURRENT_POSITION, JobAnchor.CENTER)
    )
    await editor.wait_until_settled()
    frame_spy = mocker.spy(machine.driver, "frame")

    with pytest.raises(JobPlacementError):
        await MachineCmd(editor).frame_job(machine)
    await editor.wait_until_settled()

    frame_spy.assert_not_called()


@pytest.mark.asyncio
async def test_no_new_job_while_returning_to_start(square_doc, mocker):
    editor, machine = square_doc
    _set_head(machine, *HEAD_POS[:2])
    editor.doc.set_job_origin(
        JobOrigin(StartFrom.CURRENT_POSITION, JobAnchor.BOTTOM_LEFT)
    )
    await editor.wait_until_settled()
    cmd = MachineCmd(editor)
    original_run = machine.driver.run

    async def run_and_keep_busy(*args, **kwargs):
        await original_run(*args, **kwargs)
        # The controller still works through its buffer.
        machine.device_state.status = DeviceStatus.RUN

    mocker.patch.object(machine.driver, "run", side_effect=run_and_keep_busy)
    move_spy = mocker.spy(machine.driver, "move_to")
    messages: list[str] = []
    editor.notification_requested.connect(
        lambda sender, message, **kw: messages.append(message), weak=False
    )

    first = asyncio.create_task(cmd.send_job(machine))
    while machine.device_state.status != DeviceStatus.RUN:
        await asyncio.sleep(0.01)
    # While the head has not returned yet, a new Current Position job
    # is refused: the machine is not idle.
    with pytest.raises(JobPlacementError):
        await cmd.send_job(machine)
    assert messages and "idle" in messages[-1]

    machine.device_state.status = DeviceStatus.IDLE
    await first
    await editor.wait_until_settled()
    move_spy.assert_called_once()
