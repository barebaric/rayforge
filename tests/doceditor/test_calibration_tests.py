"""
End-to-end tests for the Interval Test and Focus Test generators:
the generated layer, its steps, and the G-code it produces on GRBL.
"""

import importlib
import re
from itertools import pairwise
from pathlib import Path

import pytest

from rayforge.context import get_context
from rayforge.core.workpiece import WorkPiece
from rayforge.machine.models.machine import Origin
from rayforge.pipeline.artifact import JobArtifact

DATA = Path(__file__).parent / "data" / "calibration_tests"

# The laser addon is importable through the sys.path setup of the root
# conftest; static analysis does not see that path.
_calibration = importlib.import_module("laser_essentials.calibration_tests")
_command = importlib.import_module(
    "laser_essentials.commands.calibration_test_cmd"
)
FocusMode = _calibration.FocusMode
FocusTestParams = _calibration.FocusTestParams
IntervalTestParams = _calibration.IntervalTestParams
CalibrationTestCmd = _command.CalibrationTestCmd


@pytest.fixture
def machine(context_initializer):
    machine = get_context().machine
    assert machine is not None
    machine.set_dialect_uid("grbl")
    machine.set_origin(Origin.BOTTOM_LEFT)
    machine.set_axis_extents(400, 400)
    return machine


async def _gcode(editor) -> str:
    await editor.wait_until_settled()
    pipeline = editor.pipeline
    handle = await pipeline.generate_job_artifact_async()
    assert handle is not None
    with pipeline.artifact_store.checkout_handle(handle) as artifact:
        assert isinstance(artifact, JobArtifact)
        assert artifact.machine_code is not None
        text = artifact.machine_code
    await editor.wait_until_settled()
    return text


def _assert_golden(gcode: str, name: str):
    expected = (DATA / name).read_text(encoding="utf-8")
    assert gcode.strip().splitlines() == expected.strip().splitlines()


def _steps(layer):
    assert layer.workflow is not None
    return list(layer.workflow.steps)


def _cut_moves(gcode: str) -> list[tuple[float, float]]:
    """End points of all G1 moves, tracking modal X/Y."""
    x = y = 0.0
    points = []
    for line in gcode.splitlines():
        code = line.split(";")[0]
        words = dict(re.findall(r"([GXY])([-+]?\d*\.?\d+)", code))
        if "X" in words:
            x = float(words["X"])
        if "Y" in words:
            y = float(words["Y"])
        if words.get("G") == "1":
            points.append((x, y))
    return points


# ----------------------------------------------------------------------
# Interval Test
# ----------------------------------------------------------------------


INTERVAL_PARAMS = IntervalTestParams(
    min_interval=0.5,
    max_interval=1.0,
    count=2,
    cell_size=4.0,
    spacing=2.0,
    power_percent=40.0,
    speed=2000.0,
    include_labels=False,
)


@pytest.mark.asyncio
async def test_interval_test_layer(doc_editor, machine):
    cmd = CalibrationTestCmd(doc_editor)
    params = IntervalTestParams(count=3, include_labels=True)
    layer = cmd.create_interval_test(params)

    assert layer in doc_editor.doc.layers
    steps = _steps(layer)
    # One label step, then one engrave step per cell.
    assert [s.typelabel for s in steps] == [
        "Contour",
        "Engrave",
        "Engrave",
        "Engrave",
    ]
    workpieces = {wp.uid: wp for wp in layer.all_workpieces}
    assert len(workpieces) == 4
    for step in steps:
        assert step.generated_workpiece_uid in workpieces
    intervals = [s.line_interval_mm for s in steps[1:]]
    assert intervals == pytest.approx([0.05, 0.15, 0.25])
    assert [s.power for s in steps[1:]] == pytest.approx([0.3] * 3)
    assert steps[0].power == pytest.approx(0.1)


@pytest.mark.asyncio
async def test_interval_test_is_centered_and_undoable(doc_editor, machine):
    cmd = CalibrationTestCmd(doc_editor)
    layer = cmd.create_interval_test(IntervalTestParams(count=4))
    rects = [
        wp.get_world_geometry().rect()  # type: ignore[union-attr]
        for wp in layer.all_workpieces
    ]
    center_x = (min(r[0] for r in rects) + max(r[2] for r in rects)) / 2
    center_y = (min(r[1] for r in rects) + max(r[3] for r in rects)) / 2
    assert (center_x, center_y) == pytest.approx((200.0, 200.0), abs=0.01)

    doc_editor.history_manager.undo()
    assert layer not in doc_editor.doc.layers


@pytest.mark.asyncio
async def test_interval_test_scan_line_pitch(doc_editor, machine):
    cmd = CalibrationTestCmd(doc_editor)
    layer = cmd.create_interval_test(INTERVAL_PARAMS)
    cells = sorted(
        (wp for wp in layer.all_workpieces if isinstance(wp, WorkPiece)),
        key=lambda wp: wp.pos[0],
    )
    gcode = await _gcode(doc_editor)

    for cell, interval in zip(cells, (0.5, 1.0)):
        x0, y0, x1, y1 = cell.get_world_geometry().rect()  # type: ignore
        ys = sorted(
            {
                round(y, 3)
                for x, y in _cut_moves(gcode)
                if x0 - 0.01 <= x <= x1 + 0.01 and y0 <= y <= y1
            }
        )
        assert len(ys) >= 3
        pitches = [b - a for a, b in pairwise(ys)]
        assert pitches == pytest.approx([interval] * len(pitches), abs=0.01)
    # Every cell is filled at the same, constant power (40 % of S1000).
    powers = set(re.findall(r"^M4 S(\d+)$", gcode, re.MULTILINE))
    assert powers == {"400"}
    _assert_golden(gcode, "interval_test.gcode")


# ----------------------------------------------------------------------
# Focus Test
# ----------------------------------------------------------------------


def _focus(mode, **kwargs) -> FocusTestParams:
    values = {
        "mode": mode,
        "start_offset": -0.5,
        "step": 0.5,
        "count": 3,
        "line_length": 5.0,
        "spacing": 3.0,
        "power_percent": 20.0,
        "speed": 1200.0,
        "include_labels": False,
    }
    values.update(kwargs)
    return FocusTestParams(**values)


@pytest.mark.asyncio
async def test_focus_modes_without_z_axis(doc_editor, machine):
    cmd = CalibrationTestCmd(doc_editor)
    modes = cmd.available_focus_modes()
    assert FocusMode.MANUAL in modes
    assert FocusMode.RAMP in modes
    if not machine.has_z_axis:
        assert FocusMode.Z_AXIS not in modes
        with pytest.raises(ValueError):
            cmd.create_focus_test(_focus(FocusMode.Z_AXIS))


@pytest.mark.asyncio
async def test_focus_manual_pauses_between_lines(doc_editor, machine):
    cmd = CalibrationTestCmd(doc_editor)
    layer = cmd.create_focus_test(_focus(FocusMode.MANUAL))
    names = [s.name for s in _steps(layer)]
    assert len(names) == 6
    assert names[0].startswith("Pause")
    assert names[1] == "Focus -0.5 mm"
    assert names[3] == "Focus 0.0 mm"
    assert names[5] == "Focus +0.5 mm"

    gcode = await _gcode(doc_editor)
    lines = [line.strip() for line in gcode.splitlines()]
    assert lines.count("M0") == 3
    # Every pause comes after the laser was switched off.
    for index, line in enumerate(lines):
        if line == "M0":
            previous = [p for p in lines[:index] if p.startswith("M")]
            assert not previous or previous[-1].startswith("M5")
    _assert_golden(gcode, "focus_manual.gcode")


@pytest.mark.asyncio
async def test_focus_z_axis_moves_and_returns(doc_editor, machine, mocker):
    mocker.patch.object(
        type(machine), "has_z_axis", new_callable=mocker.PropertyMock
    ).return_value = True
    cmd = CalibrationTestCmd(doc_editor)
    assert FocusMode.Z_AXIS in cmd.available_focus_modes()
    cmd.create_focus_test(_focus(FocusMode.Z_AXIS))

    gcode = await _gcode(doc_editor)
    z_moves = [
        float(m.group(1))
        for m in re.finditer(r"^G0 Z([-+]?\d*\.?\d+)$", gcode, re.MULTILINE)
    ]
    assert z_moves == pytest.approx([-0.5, 0.5, 0.5, -0.5])
    assert sum(z_moves) == pytest.approx(0.0)
    assert gcode.count("G91") == 4
    _assert_golden(gcode, "focus_z_axis.gcode")


@pytest.mark.asyncio
async def test_focus_ramp_has_no_commands(doc_editor, machine):
    cmd = CalibrationTestCmd(doc_editor)
    layer = cmd.create_focus_test(
        _focus(FocusMode.RAMP, ramp_length=30.0, include_labels=True)
    )
    assert [s.typelabel for s in _steps(layer)] == ["Contour", "Contour"]
    gcode = await _gcode(doc_editor)
    assert "M0" not in gcode
    assert "G91" not in gcode
