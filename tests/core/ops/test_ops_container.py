import pytest
import math
import io
import json
import numpy as np
from contextlib import redirect_stdout
from typing import cast
from rayforge.core.geo.geometry import Geometry
from rayforge.core.ops import (
    Ops,
    OpsSection,
    MoveToCommand,
    LineToCommand,
    ArcToCommand,
    BezierToCommand,
    CurveToCommand,
    SetPowerCommand,
    SetCutSpeedCommand,
    SetTravelSpeedCommand,
    EnableAirAssistCommand,
    DisableAirAssistCommand,
    State,
    MovingCommand,
    SectionType,
    OpsSectionStartCommand,
    OpsSectionEndCommand,
    ScanLinePowerCommand,
    SetLaserCommand,
)


@pytest.fixture
def empty_ops():
    return Ops()


@pytest.fixture
def sample_ops():
    ops = Ops()
    ops.move_to(0, 0)
    ops.line_to(10, 10)
    ops.set_power(0.5)
    ops.enable_air_assist()
    return ops


def test_initialization(empty_ops):
    assert len(empty_ops.commands) == 0
    assert empty_ops.last_move_to == (0.0, 0.0, 0.0)


def test_add_commands(empty_ops):
    empty_ops.move_to(5, 5)
    assert len(empty_ops.commands) == 1
    assert isinstance(empty_ops.commands[0], MoveToCommand)

    empty_ops.line_to(10, 10)
    assert isinstance(empty_ops.commands[1], LineToCommand)


def test_clear_commands(sample_ops):
    sample_ops.clear()
    assert len(sample_ops.commands) == 0


def test_ops_addition(sample_ops):
    ops2 = Ops()
    ops2.move_to(20, 20)
    combined = sample_ops + ops2
    assert len(combined) == len(sample_ops) + len(ops2)


def test_ops_multiplication(sample_ops):
    multiplied = sample_ops * 3
    assert len(multiplied) == 3 * len(sample_ops)


def test_ops_extend(sample_ops):
    # Create another Ops object to extend with
    ops2 = Ops()
    ops2.move_to(20, 20)
    ops2.set_cut_speed(1000)

    original_len = len(sample_ops)
    len_to_add = len(ops2)

    # Perform the extend operation
    sample_ops.extend(ops2)

    # Verify the length has increased correctly
    assert len(sample_ops) == original_len + len_to_add

    # Verify the last two commands are the ones from ops2
    assert sample_ops.commands[-2] is ops2.commands[0]
    assert sample_ops.commands[-1] is ops2.commands[1]


def test_ops_extend_with_empty(sample_ops):
    empty_ops = Ops()
    original_len = len(sample_ops)
    sample_ops.extend(empty_ops)
    assert len(sample_ops) == original_len


def test_ops_extend_with_none(sample_ops):
    original_len = len(sample_ops)
    # This test is just to ensure it doesn't raise an exception.
    # The type hint is `Ops`, so this would be a type error, but
    # robust code should handle it.
    sample_ops.extend(None)  # type: ignore
    assert len(sample_ops) == original_len


def test_copy():
    ops_original = Ops()
    ops_original.move_to(10, 10)
    ops_original.line_to(20, 20)
    ops_original.last_move_to = (10, 10, 0)

    ops_copy = ops_original.copy()

    # Check for deepcopy: objects should not be the same instance
    assert ops_original is not ops_copy
    assert ops_original.commands is not ops_copy.commands
    assert ops_original.commands[0] is not ops_copy.commands[0]
    assert ops_original.last_move_to == ops_copy.last_move_to

    # Modify the copy and check that original is unchanged
    ops_copy.translate(5, 5)
    ops_copy.set_power(1.0)

    assert len(ops_original.commands) == 2
    assert ops_original.commands[0].end == (10, 10, 0)
    assert ops_original.commands[1].end == (20, 20, 0)

    assert len(ops_copy.commands) == 3
    assert ops_copy.commands[0].end == (15, 15, 0)
    assert ops_copy.commands[1].end == (25, 25, 0)


def test_preload_state(sample_ops):
    sample_ops.preload_state()

    # Verify that non-state commands have their state attribute set
    for cmd in sample_ops.commands:
        if not cmd.is_state_command():
            assert cmd.state is not None
            assert isinstance(cmd.state, State)

    # Verify that state commands are still present in the commands list
    state_commands = [c for c in sample_ops.commands if c.is_state_command()]
    assert len(state_commands) > 0


def test_move_to(sample_ops):
    sample_ops.move_to(15, 15)
    last_cmd = sample_ops.commands[-1]
    assert isinstance(last_cmd, MoveToCommand)
    assert last_cmd.end == (15.0, 15.0, 0.0)


def test_line_to(sample_ops):
    sample_ops.line_to(20, 20)
    last_cmd = sample_ops.commands[-1]
    assert isinstance(last_cmd, LineToCommand)
    assert last_cmd.end == (20.0, 20.0, 0.0)


def test_close_path(sample_ops):
    sample_ops.move_to(5, 5, -1.0)
    sample_ops.close_path()
    last_cmd = sample_ops.commands[-1]
    assert isinstance(last_cmd, LineToCommand)
    assert last_cmd.end == sample_ops.last_move_to
    assert last_cmd.end == (5.0, 5.0, -1.0)


def test_arc_to(sample_ops):
    sample_ops.arc_to(5, 5, 2, 3, clockwise=False)
    last_cmd = sample_ops.commands[-1]
    assert isinstance(last_cmd, ArcToCommand)
    assert last_cmd.end == (5.0, 5.0, 0.0)
    assert last_cmd.clockwise is False


def test_bezier_to():
    ops = Ops()
    ops.move_to(0.0, 0.0, 10.0)
    ops.bezier_to(
        c1=(1.0, 1.0, 10.0),
        c2=(2.0, 1.0, 20.0),
        end=(3.0, 0.0, 20.0),
    )

    assert len(ops.commands) == 2  # move_to + bezier_to
    assert isinstance(ops.commands[0], MoveToCommand)
    assert isinstance(ops.commands[1], BezierToCommand)
    assert ops.commands[1].end == (3.0, 0.0, 20.0)
    assert ops.commands[1].control1 == (1.0, 1.0, 10.0)
    assert ops.commands[1].control2 == (2.0, 1.0, 20.0)


def test_bezier_to_no_start_point():
    ops = Ops()
    # Should not raise an error, just do nothing and log a warning.
    ops.bezier_to((1, 1, 1), (2, 2, 2), (3, 3, 3))
    assert ops.is_empty()


def test_set_power(sample_ops):
    sample_ops.set_power(0.8)
    last_cmd = sample_ops.commands[-1]
    assert isinstance(last_cmd, SetPowerCommand)
    assert last_cmd.power == 0.8


def test_set_cut_speed(sample_ops):
    sample_ops.set_cut_speed(300)
    last_cmd = sample_ops.commands[-1]
    assert isinstance(last_cmd, SetCutSpeedCommand)
    assert last_cmd.speed == 300.0


def test_set_travel_speed(sample_ops):
    sample_ops.set_travel_speed(2000)
    last_cmd = sample_ops.commands[-1]
    assert isinstance(last_cmd, SetTravelSpeedCommand)
    assert last_cmd.speed == 2000.0


def test_set_laser():
    ops = Ops()
    ops.set_laser("laser-abc")
    last_cmd = ops.commands[-1]
    assert isinstance(last_cmd, SetLaserCommand)
    assert last_cmd.laser_uid == "laser-abc"


def test_enable_disable_air_assist(empty_ops):
    empty_ops.enable_air_assist()
    assert isinstance(empty_ops.commands[-1], EnableAirAssistCommand)

    empty_ops.disable_air_assist()
    assert isinstance(empty_ops.commands[-1], DisableAirAssistCommand)


def test_scan_to(empty_ops):
    """Test the scan_to method with default and custom power values."""
    # Test with default power values
    empty_ops.scan_to(10, 20, 5)
    assert isinstance(empty_ops.commands[-1], ScanLinePowerCommand)
    scan_cmd = empty_ops.commands[-1]
    assert scan_cmd.end == (10.0, 20.0, 5.0)
    assert scan_cmd.power_values == bytearray([255])

    # Test with custom power values
    custom_power = bytearray([100, 150, 200, 150, 100])
    empty_ops.scan_to(30, 40, 2, custom_power)
    assert isinstance(empty_ops.commands[-1], ScanLinePowerCommand)
    scan_cmd = empty_ops.commands[-1]
    assert scan_cmd.end == (30.0, 40.0, 2.0)
    assert scan_cmd.power_values == custom_power


def test_rect_default_ignores_travel():
    """Tests that Ops.rect() ignores travel moves by default."""
    ops = Ops()
    ops.move_to(0, 0)
    ops.line_to(10, 10)
    ops.move_to(100, 100)  # This move should be ignored
    min_x, min_y, max_x, max_y = ops.rect()
    assert (min_x, min_y, max_x, max_y) == (0.0, 0.0, 10.0, 10.0)


def test_rect_includes_travel():
    """Tests that Ops.rect(include_travel=True) includes travel moves."""
    ops = Ops()
    ops.move_to(-20, -20)
    ops.line_to(10, 10)
    ops.move_to(100, 100)  # This move should be included
    min_x, min_y, max_x, max_y = ops.rect(include_travel=True)
    # Points considered: (-20,-20), (10,10) from first segment,
    # and (10,10), (100,100) from second
    assert (min_x, min_y, max_x, max_y) == (-20.0, -20.0, 100.0, 100.0)


def test_get_frame(sample_ops):
    frame = sample_ops.get_frame(power=1.0, speed=500)
    assert sum(1 for c in frame if c.is_travel_command()) == 1  # move_to
    assert sum(1 for c in frame if c.is_cutting_command()) == 4  # line_to

    min_x, min_y, max_x, max_y = sample_ops.rect()

    expected_points = [
        (min_x, min_y, 0.0),
        (min_x, max_y, 0.0),
        (max_x, max_y, 0.0),
        (max_x, min_y, 0.0),
        (min_x, min_y, 0.0),
    ]

    frame_points = [
        cmd.end for cmd in frame.commands if isinstance(cmd, MovingCommand)
    ]
    assert frame_points == expected_points


def test_get_frame_empty(empty_ops):
    frame = empty_ops.get_frame()
    assert len(frame.commands) == 0


def test_distance(sample_ops):
    sample_ops.move_to(20, 20, -5)  # Travel with Z change
    distance = sample_ops.distance()
    # Distance should be 2D
    expected = math.dist((0, 0), (10, 10)) + math.dist((10, 10), (20, 20))
    assert distance == pytest.approx(expected)


def test_cut_distance(sample_ops):
    # Add a travel move to ensure it's not counted
    sample_ops.move_to(100, 100)
    cut_dist = sample_ops.cut_distance()
    # Only the initial line_to(10, 10) from (0,0) should be counted
    expected = math.hypot(10, 10)
    assert cut_dist == pytest.approx(expected)


def test_segments(sample_ops):
    sample_ops.move_to(5, 5)  # Travel command
    segments = list(sample_ops.segments())
    assert len(segments) > 0
    # First segment should end before the travel command
    assert isinstance(segments[0][-1], LineToCommand)


def test_preload_state_application():
    ops = Ops()
    ops.set_power(0.3)
    ops.line_to(5, 5)
    ops.set_cut_speed(200)
    ops.preload_state()

    line_cmd = ops.commands[1]
    assert line_cmd.state is not None
    assert line_cmd.state.power == 0.3

    state_commands = [cmd for cmd in ops.commands if cmd.is_state_command()]
    assert len(state_commands) == 2

    for cmd in ops.commands:
        if not cmd.is_state_command():
            assert cmd.state is not None
            assert isinstance(cmd.state, State)


def test_translate_3d():
    ops = Ops()
    ops.move_to(10, 20, 30)
    ops.line_to(30, 40, 50)
    ops.translate(5, 10, -20)
    assert ops.commands[0].end is not None
    assert ops.commands[0].end == pytest.approx((15, 30, 10))
    assert ops.commands[1].end is not None
    assert ops.commands[1].end == pytest.approx((35, 50, 30))
    assert ops.last_move_to == pytest.approx((15, 30, 10))


def test_scale_3d():
    ops = Ops()
    ops.move_to(10, 20, 5)
    ops.arc_to(22, 22, 5, 7, z=-10)
    ops.scale(2, 3, 4)  # Non-uniform scale

    assert ops.commands[0].end is not None
    assert ops.commands[0].end == pytest.approx((20, 60, 20))
    assert isinstance(ops.commands[1], LineToCommand)
    final_cmd = ops.commands[-1]
    assert final_cmd.end is not None
    final_point = final_cmd.end
    expected_final_point = (22 * 2, 22 * 3, -10 * 4)
    assert final_point == pytest.approx(expected_final_point)
    assert ops.last_move_to == pytest.approx((20, 60, 20))


def test_rotate_preserves_z():
    ops = Ops()
    ops.move_to(10, 10, -5)
    ops.rotate(90, 0, 0)
    assert ops.commands[0].end is not None
    x, y, z = ops.commands[0].end
    assert z == -5
    assert x == pytest.approx(-10)
    assert y == pytest.approx(10)


def test_transform_uniform():
    """Tests applying a uniform transformation (rotation + translation)."""
    ops = Ops()
    ops.move_to(10, 0)
    ops.arc_to(0, 10, i=-10, j=0, clockwise=False)  # 90 deg arc

    # Rotate 90 degrees around origin and translate by (100, 0)
    angle_rad = math.radians(90)
    cos_a, sin_a = math.cos(angle_rad), math.sin(angle_rad)
    matrix = np.array(
        [
            [cos_a, -sin_a, 0, 100],
            [sin_a, cos_a, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ]
    )

    ops.transform(matrix)

    move_cmd = ops.commands[0]
    arc_cmd = ops.commands[1]

    # Original (10,0) -> Rotated (0,10) -> Translated (100, 10)
    assert move_cmd.end == pytest.approx((100, 10, 0))
    # Original (0,10) -> Rotated (-10,0) -> Translated (90, 0)
    assert arc_cmd.end == pytest.approx((90, 0, 0))

    # Arc should NOT be linearized
    assert isinstance(arc_cmd, ArcToCommand)

    # Original offset (-10, 0) should be rotated to (0, -10)
    assert arc_cmd.center_offset[0] == pytest.approx(0)
    assert arc_cmd.center_offset[1] == pytest.approx(-10)


def test_transform_non_uniform():
    """Tests that a non-uniform scale linearizes arcs."""
    ops = Ops()
    ops.move_to(10, 0)
    ops.arc_to(0, 10, i=-10, j=0, clockwise=False)  # 90 deg arc

    # Scale X by 2, Y by 3
    matrix = np.diag([2.0, 3.0, 1.0, 1.0])
    ops.transform(matrix)

    # Original move_to (10,0) -> (20, 0)
    assert ops.commands[0].end == pytest.approx((20, 0, 0))

    # Arc must be linearized into LineToCommands
    assert all(isinstance(c, LineToCommand) for c in ops.commands[1:])
    assert len(ops.commands) > 2

    # Final point should be original arc end (0,10) scaled -> (0, 30)
    assert ops.commands[-1].end == pytest.approx((0, 30, 0))


def test_transform_uniform_bezier():
    """Uniform transform preserves BezierToCommand and transforms controls."""
    ops = Ops()
    ops.move_to(0, 0)
    ops.bezier_to(c1=(10, 0, 0), c2=(10, 10, 0), end=(0, 10, 0))

    angle_rad = math.radians(90)
    cos_a, sin_a = math.cos(angle_rad), math.sin(angle_rad)
    matrix = np.array(
        [
            [cos_a, -sin_a, 0, 5],
            [sin_a, cos_a, 0, 10],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ]
    )
    ops.transform(matrix)

    assert isinstance(ops.commands[1], BezierToCommand)
    cmd = ops.commands[1]
    assert cmd.end == pytest.approx((-5, 10, 0))
    assert cmd.control1 == pytest.approx((5, 20, 0))
    assert cmd.control2 == pytest.approx((-5, 20, 0))


def test_transform_non_uniform_bezier():
    """Non-uniform transform preserves BezierToCommand (affine invariance)."""
    ops = Ops()
    ops.move_to(0, 0)
    ops.bezier_to(c1=(10, 0, 0), c2=(10, 10, 0), end=(0, 10, 0))

    matrix = np.diag([2.0, 3.0, 1.0, 1.0])
    ops.transform(matrix)

    assert isinstance(ops.commands[1], BezierToCommand)
    cmd = ops.commands[1]
    assert cmd.end == pytest.approx((0, 30, 0))
    assert cmd.control1 == pytest.approx((20, 0, 0))
    assert cmd.control2 == pytest.approx((20, 30, 0))


def test_transform_bezier_linearize_matches():
    """Transform then linearize should match linearize then transform.

    Bezier curves are affine-invariant: transforming control points
    then evaluating produces the same curve as evaluating then
    transforming.  We verify by comparing bounding boxes rather than
    point-by-point (different subdivision counts are expected when
    the linearization tolerance is applied in different scales).
    """
    ops = Ops()
    ops.move_to(0, 0)
    ops.bezier_to(c1=(10, 0, 0), c2=(10, 10, 0), end=(0, 10, 0))

    ops_linearized_first = Ops()
    ops_linearized_first.move_to(0, 0)
    ops_linearized_first.bezier_to(
        c1=(10, 0, 0), c2=(10, 10, 0), end=(0, 10, 0)
    )
    ops_linearized_first.linearize_curves()

    matrix = np.array(
        [
            [2, 0, 0, 5],
            [0, 2, 0, 3],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ]
    )

    ops.transform(matrix)
    ops.linearize_curves()

    ops_linearized_first.transform(matrix)

    def _endpoints(o):
        return np.array(
            [c.end for c in o.commands if isinstance(c, MovingCommand)]
        )

    pts_a = _endpoints(ops)
    pts_b = _endpoints(ops_linearized_first)

    assert pts_a[0] == pytest.approx(pts_b[0], abs=1e-9)
    assert pts_a[-1] == pytest.approx(pts_b[-1], abs=1e-9)

    for i in range(3):
        assert pts_a[:, i].min() == pytest.approx(pts_b[:, i].min(), abs=0.1)
        assert pts_a[:, i].max() == pytest.approx(pts_b[:, i].max(), abs=0.1)


def test_linearize_all():
    ops = Ops()
    ops.move_to(10, 0)
    ops.line_to(20, 0)
    ops.arc_to(10, 10, i=-10, j=0, clockwise=False)  # Semicircle
    ops.set_power(1.0)

    ops.linearize_all()

    assert len(ops.commands) > 4  # Move, Line, SetPower, plus linearized arc
    assert isinstance(ops.commands[0], MoveToCommand)
    assert isinstance(ops.commands[1], LineToCommand)
    # All geometric commands after the first two should be LineTo
    moving_cmds_after = [
        c for c in ops.commands[2:] if isinstance(c, MovingCommand)
    ]
    assert all(isinstance(c, LineToCommand) for c in moving_cmds_after)
    # Check that state command is still there
    assert any(isinstance(c, SetPowerCommand) for c in ops.commands)


@pytest.fixture
def clip_rect():
    return (0.0, 0.0, 100.0, 100.0)


def test_clip_fully_inside(clip_rect):
    ops = Ops()
    ops.move_to(10, 10, -1)
    ops.line_to(90, 90, -1)
    clipped_ops = ops.clip_rect(clip_rect)
    assert len(clipped_ops.commands) == 2
    assert isinstance(clipped_ops.commands[0], MoveToCommand)
    assert isinstance(clipped_ops.commands[1], LineToCommand)
    assert clipped_ops.commands[1].end is not None
    assert clipped_ops.commands[1].end == pytest.approx((90.0, 90.0, -1.0))


def test_clip_fully_outside(clip_rect):
    ops = Ops()
    ops.move_to(110, 110, 0)
    ops.line_to(120, 120, 0)
    clipped_ops = ops.clip_rect(clip_rect)
    assert len(clipped_ops.commands) == 0


def test_clip_with_arc():
    """Verify clip works on arcs via the new generic linearize interface."""
    ops = Ops()
    ops.move_to(0, 50)
    ops.arc_to(100, 50, i=50, j=0, clockwise=False)  # Semicircle
    clip_rect = (40.0, 0.0, 60.0, 100.0)  # A vertical slice through the middle
    clipped_ops = ops.clip_rect(clip_rect)

    # Check that there are drawing commands left
    drawing_cmds = [c for c in clipped_ops if c.is_cutting_command()]
    assert len(drawing_cmds) > 0

    # Check that all remaining points are within the rect bounds
    for cmd in clipped_ops:
        if isinstance(cmd, MovingCommand) and cmd.end is not None:
            x, y, z = cmd.end
            assert clip_rect[0] <= x <= clip_rect[2]
            assert clip_rect[1] <= y <= clip_rect[3]


def test_clip_scanlinepowercommand_start_outside():
    """Tests clipping a scanline that starts outside and ends inside."""
    ops = Ops()
    ops.move_to(0, 50, 10)
    ops.scan_to(100, 50, 10, bytearray(range(100)))
    clip_rect = (50, 0, 150, 100)
    clipped_ops = ops.clip_rect(clip_rect)

    assert len(clipped_ops.commands) == 2  # MoveTo, ScanLinePowerCommand
    assert isinstance(clipped_ops.commands[0], MoveToCommand)
    clipped_cmd = cast(ScanLinePowerCommand, clipped_ops.commands[1])

    # 1. Verify it's still a ScanLinePowerCommand (not linearized)
    assert isinstance(clipped_cmd, ScanLinePowerCommand)

    # 2. Verify new geometry (starts at the clip boundary)
    assert clipped_ops.commands[0].end == pytest.approx((50, 50, 10))
    assert clipped_cmd.end == pytest.approx((100, 50, 10))

    # 3. Verify power values are sliced correctly (original was 100 values)
    # The clip starts 50% of the way through the line.
    assert len(clipped_cmd.power_values) == 50
    assert clipped_cmd.power_values[0] == 50
    assert clipped_cmd.power_values[-1] == 99


def test_clip_scanlinepowercommand_crossing_with_z_interp():
    """Tests a scanline that crosses the clip rect with Z interpolation."""
    ops = Ops()
    # Line from (-50, 50, 0) to (150, 50, 200) -> total length 200
    ops.move_to(-50, 50, 0)
    ops.scan_to(150, 50, 200, bytearray(range(200)))
    clip_rect = (0, 0, 100, 100)
    clipped_ops = ops.clip_rect(clip_rect)

    assert len(clipped_ops.commands) == 2
    clipped_cmd = cast(ScanLinePowerCommand, clipped_ops.commands[1])
    assert isinstance(clipped_cmd, ScanLinePowerCommand)

    # The line starts 50 units before x=0 and ends 50 units after x=100.
    # The clipped portion is from x=0 to x=100.
    # t_start = 50 / 200 = 0.25. t_end = 150 / 200 = 0.75
    expected_z_start = 0 + (0.25 * 200)  # 50
    expected_z_end = 0 + (0.75 * 200)  # 150

    assert clipped_ops.commands[0].end == pytest.approx(
        (0, 50, expected_z_start)
    )
    assert clipped_cmd.end == pytest.approx((100, 50, expected_z_end))

    # Power values should be sliced from index 50 to 150.
    expected_len = int(200 * 0.75) - int(200 * 0.25)
    assert len(clipped_cmd.power_values) == expected_len
    assert clipped_cmd.power_values[0] == 50
    assert clipped_cmd.power_values[-1] == 149


def test_clip_scanlinepowercommand_fully_outside():
    """Tests that a fully outside scanline is removed."""
    ops = Ops()
    ops.move_to(200, 50, 10)
    ops.scan_to(300, 50, 10, bytearray(range(100)))
    clip_rect = (0, 0, 100, 100)
    clipped_ops = ops.clip_rect(clip_rect)
    assert len(clipped_ops.commands) == 0


def test_subtract_regions():
    ops = Ops()
    ops.move_to(0, 50, -5)
    ops.line_to(100, 50, 5)
    region = [(40.0, 45.0), (60.0, 45.0), (60.0, 55.0), (40.0, 55.0)]
    ops.subtract_regions([region])
    assert len(ops.commands) == 4
    assert ops.commands[1].end == pytest.approx((40.0, 50.0, -1.0))
    assert ops.commands[3].end == pytest.approx((100.0, 50.0, 5.0))


def test_subtract_regions_with_scanline():
    ops = Ops()
    ops.move_to(0, 50, 0)
    ops.scan_to(100, 50, 0, bytearray([100] * 100))
    # Region to cut out of the middle
    region = [(40.0, 40.0), (60.0, 40.0), (60.0, 60.0), (40.0, 60.0)]
    ops.subtract_regions([region])

    # Expected: M(0,50), S(->40,50), M(60,50), S(->100,50)
    assert len(ops.commands) == 4
    assert isinstance(ops.commands[0], MoveToCommand)
    assert ops.commands[0].end == pytest.approx((0, 50, 0))

    scan1 = cast(ScanLinePowerCommand, ops.commands[1])
    assert isinstance(scan1, ScanLinePowerCommand)
    assert scan1.end == pytest.approx((40, 50, 0))
    assert len(scan1.power_values) == 40

    assert isinstance(ops.commands[2], MoveToCommand)
    assert ops.commands[2].end == pytest.approx((60, 50, 0))

    scan2 = cast(ScanLinePowerCommand, ops.commands[3])
    assert isinstance(scan2, ScanLinePowerCommand)
    assert scan2.end == pytest.approx((100, 50, 0))
    assert len(scan2.power_values) == 40


def test_clip_to_regions_basic():
    ops = Ops()
    ops.move_to(0, 50, -5)
    ops.line_to(100, 50, 5)
    region = [(40.0, 45.0), (60.0, 45.0), (60.0, 55.0), (40.0, 55.0)]

    ops.clip_to_regions([region])

    assert len(ops.commands) == 2
    assert ops.commands[0].end == pytest.approx((40.0, 50.0, -1.0))
    assert ops.commands[1].end == pytest.approx((60.0, 50.0, 1.0))


def test_clip_to_regions_fully_outside():
    ops = Ops()
    ops.move_to(0, 50, 0)
    ops.line_to(30, 50, 0)
    region = [(40.0, 45.0), (60.0, 45.0), (60.0, 55.0), (40.0, 55.0)]

    ops.clip_to_regions([region])

    assert len(ops.commands) == 0


def test_clip_to_regions_fully_inside():
    ops = Ops()
    ops.move_to(45, 50, 0)
    ops.line_to(55, 50, 0)
    region = [(40.0, 45.0), (60.0, 45.0), (60.0, 55.0), (40.0, 55.0)]

    ops.clip_to_regions([region])

    assert len(ops.commands) == 2
    assert ops.commands[1].end == pytest.approx((55, 50, 0))


def test_clip_to_regions_multiple_regions():
    ops = Ops()
    ops.move_to(0, 50, 0)
    ops.line_to(100, 50, 0)
    region1 = [(20.0, 45.0), (30.0, 45.0), (30.0, 55.0), (20.0, 55.0)]
    region2 = [(70.0, 45.0), (80.0, 45.0), (80.0, 55.0), (70.0, 55.0)]

    ops.clip_to_regions([region1, region2])

    assert len(ops.commands) == 4
    assert ops.commands[0].end == pytest.approx((20.0, 50.0, 0.0))
    assert ops.commands[1].end == pytest.approx((30.0, 50.0, 0.0))
    assert ops.commands[2].end == pytest.approx((70.0, 50.0, 0.0))
    assert ops.commands[3].end == pytest.approx((80.0, 50.0, 0.0))


def test_clip_to_regions_with_scanline():
    ops = Ops()
    ops.move_to(0, 50, 0)
    ops.scan_to(100, 50, 0, bytearray([100] * 100))
    region = [(40.0, 40.0), (60.0, 40.0), (60.0, 60.0), (40.0, 60.0)]

    ops.clip_to_regions([region])

    assert len(ops.commands) == 2
    assert isinstance(ops.commands[0], MoveToCommand)
    assert ops.commands[0].end == pytest.approx((40, 50, 0))

    scan = cast(ScanLinePowerCommand, ops.commands[1])
    assert isinstance(scan, ScanLinePowerCommand)
    assert scan.end == pytest.approx((60, 50, 0))
    assert len(scan.power_values) == 20


def test_clip_to_regions_empty_regions():
    ops = Ops()
    ops.move_to(0, 50, 0)
    ops.line_to(100, 50, 0)

    original_len = len(ops.commands)
    ops.clip_to_regions([])

    assert len(ops.commands) == original_len


def test_clip_to_regions_preserves_state_commands():
    ops = Ops()
    ops.set_power(0.8)
    ops.move_to(0, 50, 0)
    ops.line_to(100, 50, 0)
    region = [(40.0, 45.0), (60.0, 45.0), (60.0, 55.0), (40.0, 55.0)]

    ops.clip_to_regions([region])

    assert len(ops.commands) == 3
    assert isinstance(ops.commands[0], SetPowerCommand)
    assert isinstance(ops.commands[1], MoveToCommand)
    assert isinstance(ops.commands[2], LineToCommand)


def create_circle_polygon(cx, cy, radius, num_segments=32):
    points = []
    for i in range(num_segments):
        angle = 2 * math.pi * i / num_segments
        x = cx + radius * math.cos(angle)
        y = cy + radius * math.sin(angle)
        points.append((x, y))
    return points


class TestClipToCircleRegions:
    """Tests for clipping Ops to circular regions."""

    def test_clip_to_circle_basic(self):
        ops = Ops()
        ops.move_to(0, 50, -5)
        ops.line_to(100, 50, 5)
        circle = create_circle_polygon(50, 50, 20)

        ops.clip_to_regions([circle])

        assert len(ops.commands) == 2
        assert ops.commands[0].end is not None
        assert ops.commands[0].end[0] == pytest.approx(30.0, abs=0.5)
        assert ops.commands[0].end[1] == pytest.approx(50.0)
        assert ops.commands[1].end is not None
        assert ops.commands[1].end[0] == pytest.approx(70.0, abs=0.5)
        assert ops.commands[1].end[1] == pytest.approx(50.0)

    def test_clip_to_circle_fully_outside(self):
        ops = Ops()
        ops.move_to(0, 50, 0)
        ops.line_to(25, 50, 0)
        circle = create_circle_polygon(50, 50, 20)

        ops.clip_to_regions([circle])

        assert len(ops.commands) == 0

    def test_clip_to_circle_fully_inside(self):
        ops = Ops()
        ops.move_to(45, 50, 0)
        ops.line_to(55, 50, 0)
        circle = create_circle_polygon(50, 50, 20)

        ops.clip_to_regions([circle])

        assert len(ops.commands) == 2
        assert ops.commands[1].end == pytest.approx((55, 50, 0))

    def test_clip_to_multiple_circles(self):
        ops = Ops()
        ops.move_to(0, 50, 0)
        ops.line_to(100, 50, 0)
        circle1 = create_circle_polygon(25, 50, 10)
        circle2 = create_circle_polygon(75, 50, 10)

        ops.clip_to_regions([circle1, circle2])

        assert len(ops.commands) == 4
        assert ops.commands[0].end is not None
        assert ops.commands[0].end[0] == pytest.approx(15.0, abs=0.5)
        assert ops.commands[1].end is not None
        assert ops.commands[1].end[0] == pytest.approx(35.0, abs=0.5)
        assert ops.commands[2].end is not None
        assert ops.commands[2].end[0] == pytest.approx(65.0, abs=0.5)
        assert ops.commands[3].end is not None
        assert ops.commands[3].end[0] == pytest.approx(85.0, abs=0.5)

    def test_clip_to_circle_vertical_line(self):
        ops = Ops()
        ops.move_to(50, 0, 0)
        ops.line_to(50, 100, 0)
        circle = create_circle_polygon(50, 50, 20)

        ops.clip_to_regions([circle])

        assert len(ops.commands) == 2
        assert ops.commands[0].end is not None
        assert ops.commands[0].end[1] == pytest.approx(30.0, abs=0.5)
        assert ops.commands[1].end is not None
        assert ops.commands[1].end[1] == pytest.approx(70.0, abs=0.5)

    def test_clip_to_circle_diagonal_line(self):
        ops = Ops()
        ops.move_to(0, 0, 0)
        ops.line_to(100, 100, 0)
        circle = create_circle_polygon(50, 50, 20)

        ops.clip_to_regions([circle])

        assert len(ops.commands) == 2
        assert ops.commands[0].end is not None
        start_x = ops.commands[0].end[0]
        assert ops.commands[1].end is not None
        end_x = ops.commands[1].end[0]
        assert start_x == pytest.approx(50 - 20 * math.sqrt(2) / 2, abs=0.5)
        assert end_x == pytest.approx(50 + 20 * math.sqrt(2) / 2, abs=0.5)

    def test_clip_to_circle_with_scanline(self):
        ops = Ops()
        ops.move_to(0, 50, 0)
        ops.scan_to(100, 50, 0, bytearray([100] * 100))
        circle = create_circle_polygon(50, 50, 20)

        ops.clip_to_regions([circle])

        assert len(ops.commands) == 2
        assert isinstance(ops.commands[0], MoveToCommand)
        assert ops.commands[0].end is not None
        assert ops.commands[0].end[0] == pytest.approx(30.0, abs=0.5)

        scan = cast(ScanLinePowerCommand, ops.commands[1])
        assert isinstance(scan, ScanLinePowerCommand)
        assert scan.end is not None
        assert scan.end[0] == pytest.approx(70.0, abs=0.5)
        assert len(scan.power_values) == pytest.approx(40, abs=2)

    def test_clip_to_circle_with_arc(self):
        ops = Ops()
        ops.move_to(30, 30)
        ops.arc_to(70, 30, i=20, j=0, clockwise=True)
        circle = create_circle_polygon(50, 50, 20)

        ops.clip_to_regions([circle])

        assert len(ops.commands) > 0
        drawing_cmds = [c for c in ops if c.is_cutting_command()]
        assert len(drawing_cmds) > 0
        for cmd in ops.commands:
            if isinstance(cmd, MovingCommand) and cmd.end is not None:
                x, y, z = cmd.end
                dist = math.hypot(x - 50, y - 50)
                assert dist <= 20.5

    def test_clip_to_circle_preserves_state_commands(self):
        ops = Ops()
        ops.set_power(0.8)
        ops.move_to(0, 50, 0)
        ops.line_to(100, 50, 0)
        circle = create_circle_polygon(50, 50, 20)

        ops.clip_to_regions([circle])

        assert len(ops.commands) == 3
        assert isinstance(ops.commands[0], SetPowerCommand)
        assert isinstance(ops.commands[1], MoveToCommand)
        assert isinstance(ops.commands[2], LineToCommand)


def test_from_geometry():
    # Use the actual Geometry class instead of mocks to ensure correct types
    geo_obj = Geometry()
    geo_obj.move_to(10, 10, 0)
    geo_obj.line_to(20, 20, 0)
    geo_obj.arc_to(30, 10, -10, 0, clockwise=False, z=0)

    ops = Ops.from_geometry(geo_obj)

    assert len(ops.commands) == 3
    assert isinstance(ops.commands[0], MoveToCommand)
    assert ops.commands[0].end == (10, 10, 0)
    assert isinstance(ops.commands[1], LineToCommand)
    assert ops.commands[1].end == (20, 20, 0)
    assert isinstance(ops.commands[2], ArcToCommand)
    assert ops.commands[2].end == (30, 10, 0)
    assert ops.commands[2].center_offset == (-10, 0)
    assert ops.commands[2].clockwise is False
    assert ops.last_move_to == geo_obj.last_move_to


def test_from_geometry_with_bezier():
    geo_obj = Geometry()
    geo_obj.move_to(10, 10, 0)
    geo_obj.line_to(20, 20, 0)
    geo_obj.arc_to_as_bezier(30, 10, -10, 0, clockwise=False, z=0)

    ops = Ops.from_geometry(geo_obj)

    assert isinstance(ops.commands[0], MoveToCommand)
    assert ops.commands[0].end == pytest.approx((10, 10, 0))
    assert isinstance(ops.commands[1], LineToCommand)
    assert ops.commands[1].end == pytest.approx((20, 20, 0))
    assert all(isinstance(c, BezierToCommand) for c in ops.commands[2:])
    assert ops.commands[-1].end == pytest.approx((30, 10, 0))
    assert ops.last_move_to == geo_obj.last_move_to


def test_serialization_deserialization_all_types():
    """Tests that all command types can be serialized and deserialized."""
    ops = Ops()
    ops.job_start()
    ops.layer_start("layer-1")
    ops.workpiece_start("wp-1")
    ops.ops_section_start(SectionType.RASTER_FILL, "wp-1")
    ops.set_travel_speed(5000)
    ops.set_cut_speed(1000)
    ops.set_power(0.8)
    ops.enable_air_assist()
    ops.set_laser("laser-2")
    ops.move_to(1, 1, 1)
    ops.line_to(2, 2, 2)
    ops.arc_to(3, 1, 1, 1, clockwise=False)
    ops.scan_to(12, 2, 2, bytearray([50, 150]))
    ops.ops_section_end(SectionType.RASTER_FILL)
    ops.workpiece_end("wp-1")
    ops.layer_end("layer-1")
    ops.job_end()
    ops.last_move_to = (1, 1, 1)

    data = ops.to_dict()
    new_ops = Ops.from_dict(data)

    assert len(ops.commands) == len(new_ops.commands)
    assert new_ops.last_move_to == (1, 1, 1)

    for old_cmd, new_cmd in zip(ops.commands, new_ops.commands):
        # Check type and dict representation to ensure all fields are preserved
        assert type(old_cmd) is type(new_cmd)
        assert old_cmd.to_dict() == new_cmd.to_dict()


def test_translate_with_scanline():
    """Tests that translate() correctly transforms ScanLinePowerCommand."""
    ops = Ops()
    ops.move_to(10, 20, 30)
    ops.scan_to(40, 50, 60, bytearray([1, 2, 3]))
    ops.translate(5, -10, 15)

    move_cmd = cast(MoveToCommand, ops.commands[0])
    translated_cmd = cast(ScanLinePowerCommand, ops.commands[1])
    assert isinstance(translated_cmd, ScanLinePowerCommand)

    # Check if both start_point (from move_to) and end are translated
    assert move_cmd.end == pytest.approx((15, 10, 45))
    assert translated_cmd.end == pytest.approx((45, 40, 75))


def test_clip_at_no_hit():
    """Tests that clip_at does nothing if no point is found."""
    ops = Ops()
    ops.move_to(0, 0)
    ops.line_to(10, 10)
    original_commands = ops.commands[:]
    # Point is far away from the path
    assert ops.clip_at(100, 100, 1.0) is False
    assert ops.commands == original_commands


def test_clip_at_on_line_segment():
    """Tests creating a gap in a simple line segment."""
    ops = Ops()
    ops.move_to(0, 50, 10)
    ops.line_to(100, 50, 20)  # Z should be interpolated

    # Clip near the midpoint
    assert ops.clip_at(50, 50, 10.0) is True

    # Expected:
    # Move(0,50,10), Line(45,50,14.5), Move(55,50,15.5), Line(100,50,20)
    assert len(ops.commands) == 4
    assert isinstance(ops.commands[0], MoveToCommand)
    assert isinstance(ops.commands[1], LineToCommand)
    assert isinstance(ops.commands[2], MoveToCommand)
    assert isinstance(ops.commands[3], LineToCommand)

    # Check the points
    assert ops.commands[1].end == pytest.approx((45.0, 50.0, 14.5))
    assert ops.commands[2].end == pytest.approx((55.0, 50.0, 15.5))
    assert ops.commands[3].end == pytest.approx((100.0, 50.0, 20.0))


def test_clip_at_on_arc_segment():
    """Tests creating a gap in an arc segment."""
    ops = Ops()
    ops.move_to(10, 0)
    # 90 deg CCW arc, radius 10, center (0,0)
    ops.arc_to(0, 10, i=-10, j=0, clockwise=False)

    # Clip near the 45-degree point on the arc
    point_on_arc_x = 10 * math.cos(math.radians(45))
    point_on_arc_y = 10 * math.sin(math.radians(45))
    assert ops.clip_at(point_on_arc_x, point_on_arc_y, 2.0) is True

    # The arc gets linearized by subtract_regions, so we expect a series
    # of LineTo commands with a gap in the middle.
    assert len(ops.commands) > 3
    # Verify there is a MoveTo command somewhere in the middle,
    # indicating a gap
    assert any(isinstance(cmd, MoveToCommand) for cmd in ops.commands[1:]), (
        "No MoveToCommand found, indicating no gap was created."
    )


def test_clip_at_start_of_subpath():
    """Tests clipping at the very beginning of a subpath."""
    ops = Ops()
    ops.move_to(0, 50)
    ops.line_to(100, 50)

    # Clip at x=1, width=2. Should clip from 0 to 2.
    assert ops.clip_at(1, 50, 2.0) is True

    # Expected: Move(0,50), Move(2,50), Line(100,50)
    assert len(ops.commands) == 3
    assert isinstance(ops.commands[0], MoveToCommand)
    assert ops.commands[0].end == pytest.approx((0, 50, 0))
    assert isinstance(ops.commands[1], MoveToCommand)
    assert ops.commands[1].end == pytest.approx((2.0, 50.0, 0.0))
    assert ops.commands[2].end == pytest.approx((100.0, 50.0, 0.0))


def test_clip_at_end_of_subpath():
    """Tests clipping at the very end of a subpath."""
    ops = Ops()
    ops.move_to(0, 50)
    ops.line_to(100, 50)

    # Clip at x=99, width=2. Should clip from 98 to 100.
    assert ops.clip_at(99, 50, 2.0) is True

    # Expected: Move(0,50), Line(98,50), Move(100,50)
    assert len(ops.commands) == 3
    assert isinstance(ops.commands[0], MoveToCommand)
    assert ops.commands[0].end == pytest.approx((0, 50, 0))
    assert isinstance(ops.commands[1], LineToCommand)
    assert ops.commands[1].end == pytest.approx((98.0, 50.0, 0.0))
    assert isinstance(ops.commands[2], MoveToCommand)
    assert ops.commands[2].end == pytest.approx((100.0, 50.0, 0.0))


def test_clip_at_spans_multiple_segments():
    """
    Tests that a clip correctly creates a gap across the boundary of two
    connected LineTo commands.
    """
    ops = Ops()
    ops.move_to(0, 0)
    ops.line_to(50, 0)  # Segment 1 (index 1)
    ops.line_to(100, 50)  # Segment 2 (index 2)
    ops.line_to(100, 100)  # Segment 3 (index 3)

    # Clip at (50, 0) with a width of 20.
    # This should remove from x=40 on the first line to some point on the
    # second line.
    assert ops.clip_at(50, 0, 20.0) is True

    # Original: M, L, L, L -> 4 commands
    # Expected: M, L(shortened), M(to skip gap), L(shortened), L -> 5+ commands
    assert len(ops.commands) > 4
    assert isinstance(ops.commands[0], MoveToCommand)
    assert isinstance(ops.commands[1], LineToCommand)
    assert isinstance(ops.commands[2], MoveToCommand)

    # The first line segment should end before 50
    assert ops.commands[1].end[0] < 50
    # The new path should start after 50
    assert ops.commands[2].end[0] > 50

    # Ensure the entire original path after the clip is still present
    assert ops.commands[-1].end == pytest.approx((100, 100, 0))


def test_clip_at_ignores_state_commands():
    """
    Tests that clip_at correctly handles state commands, ensuring they are
    not part of the geometric subpath.
    """
    ops = Ops()
    ops.move_to(0, 0)
    ops.line_to(50, 0)  # Subpath 1
    ops.set_power(1.0)
    ops.move_to(60, 0)
    ops.line_to(100, 0)  # Subpath 2

    # Clip the second line segment
    assert ops.clip_at(80, 0, 10.0) is True

    # Path 1 should be unchanged. Path 2 should be clipped.
    assert len(ops.commands) == 7
    # Path 1
    assert ops.commands[0].end == (0, 0, 0)
    assert ops.commands[1].end == (50, 0, 0)
    assert isinstance(ops.commands[2], SetPowerCommand)
    # Path 2 (clipped)
    assert ops.commands[3].end == (60, 0, 0)
    assert ops.commands[4].end == pytest.approx((75, 0, 0))
    assert ops.commands[5].end == pytest.approx((85, 0, 0))
    assert ops.commands[6].end == pytest.approx((100, 0, 0))


def test_clip_at_with_state_commands_in_subpath():
    """
    Tests that clip_at correctly handles state commands within a continuous
    subpath. The geometry index and commands index must be properly aligned.
    """
    ops = Ops()
    ops.move_to(0, 0)
    ops.line_to(50, 0)  # Segment 1
    ops.set_power(0.5)  # State command in the middle of subpath
    ops.line_to(100, 0)  # Segment 2

    # Clip on the second segment (after the state command)
    # The clip point at x=75 is on segment 2, which is at geometry index 2
    # but would be at commands index 3 if not handled correctly.
    assert ops.clip_at(75, 0, 10.0) is True

    # Expected: Move(0,0), Line(50,0), SetPower, Line(70,0), Move(80,0),
    #           Line(100,0)
    # The gap is centered at x=75 with width 10, so from 70 to 80.
    # Segment 1 (0-50) is kept whole.
    # Segment 2 (50-100) is split: 50-70 kept, 70-80 gap, 80-100 kept.
    assert len(ops.commands) == 6
    assert isinstance(ops.commands[0], MoveToCommand)
    assert ops.commands[0].end == (0, 0, 0)
    assert isinstance(ops.commands[1], LineToCommand)
    assert ops.commands[1].end == pytest.approx((50.0, 0.0, 0.0))
    assert isinstance(ops.commands[2], SetPowerCommand)
    assert isinstance(ops.commands[3], LineToCommand)
    assert ops.commands[3].end == pytest.approx((70.0, 0.0, 0.0))
    assert isinstance(ops.commands[4], MoveToCommand)
    assert ops.commands[4].end == pytest.approx((80.0, 0.0, 0.0))
    assert isinstance(ops.commands[5], LineToCommand)
    assert ops.commands[5].end == pytest.approx((100.0, 0.0, 0.0))


def test_clip_at_end_of_segment_with_state_command():
    """
    Tests clipping at the exact endpoint of a segment when there's a state
    command before the next segment.
    """
    ops = Ops()
    ops.move_to(0, 0)
    ops.line_to(100, 0)  # Segment 1
    ops.set_power(0.8)  # State command
    ops.line_to(200, 0)  # Segment 2

    # Clip at the exact endpoint of segment 1 (x=100)
    assert ops.clip_at(100, 0, 10.0) is True

    # The clip should be centered at x=100, so gap from 95 to 105
    # Segment 1 should end at 95, segment 2 should start at 105
    # Note: State command at position 100 is inside the gap, so it's removed
    assert isinstance(ops.commands[0], MoveToCommand)
    assert ops.commands[0].end == (0, 0, 0)
    assert isinstance(ops.commands[1], LineToCommand)
    assert ops.commands[1].end == pytest.approx((95.0, 0.0, 0.0))
    assert isinstance(ops.commands[2], MoveToCommand)
    assert ops.commands[2].end == pytest.approx((105.0, 0.0, 0.0))
    assert isinstance(ops.commands[3], LineToCommand)
    assert ops.commands[3].end == pytest.approx((200.0, 0.0, 0.0))


def test_dump(sample_ops):
    """Ensures dump() runs without error and produces output."""
    f = io.StringIO()
    with redirect_stdout(f):
        sample_ops.dump()
    output = f.getvalue()
    assert "MoveToCommand" in output
    assert "LineToCommand" in output


def test_estimate_time_empty(empty_ops):
    """Test time estimation for empty Ops object."""
    time_est = empty_ops.estimate_time()
    assert time_est == 0.0


def test_estimate_time_basic():
    """Test basic time estimation with simple movements."""
    ops = Ops()
    ops.move_to(0, 0)
    ops.line_to(100, 0)  # 100mm cut
    ops.move_to(0, 100)  # 141.42mm travel
    ops.line_to(100, 100)  # 100mm cut

    # Default speeds: 1000 mm/min cut, 3000 mm/min travel
    # Disable acceleration for simpler calculation
    time_est = ops.estimate_time(acceleration=0)

    # Expected: 100mm @ 1000mm/min + 141.42mm @ 3000mm/min + 100mm @ 1000mm/min
    # = 6s + 2.828s + 6s = 14.828s
    expected_time = 6.0 + 2.828 + 6.0
    assert time_est == pytest.approx(expected_time, rel=1e-3)


def test_estimate_time_with_custom_speeds():
    """Test time estimation with custom speeds."""
    ops = Ops()
    ops.move_to(0, 0)
    ops.line_to(60, 0)  # 60mm cut
    ops.move_to(0, 80)  # 100mm travel

    # Custom speeds: 1200 mm/min cut, 2400 mm/min travel
    # Disable acceleration for simpler calculation
    time_est = ops.estimate_time(
        default_cut_speed=1200.0, default_travel_speed=2400.0, acceleration=0
    )

    # Expected: 60mm @ 1200mm/min + 100mm @ 2400mm/min
    # = 3s + 2.5s = 5.5s
    expected_time = 3.0 + 2.5
    assert time_est == pytest.approx(expected_time, rel=1e-3)


def test_estimate_time_with_state_commands():
    """Test time estimation respects state commands."""
    ops = Ops()
    ops.move_to(0, 0)
    ops.set_cut_speed(2000)  # Faster cutting speed
    ops.line_to(100, 0)  # 100mm cut at 2000mm/min
    ops.set_travel_speed(6000)  # Faster travel speed
    ops.move_to(0, 100)  # 141.42mm travel at 6000mm/min

    # Disable acceleration for simpler calculation
    time_est = ops.estimate_time(acceleration=0)

    # Expected: 100mm @ 2000mm/min + 141.42mm @ 6000mm/min
    # = 3s + 1.414s = 4.414s
    expected_time = 3.0 + 1.414
    assert time_est == pytest.approx(expected_time, rel=1e-3)


def test_estimate_time_with_acceleration():
    """Test time estimation with acceleration consideration."""
    ops = Ops()
    ops.move_to(0, 0)
    ops.line_to(10, 0)  # Short movement

    # With high acceleration, short moves take longer due to accel/decel
    time_est_no_accel = ops.estimate_time(acceleration=0)
    time_est_with_accel = ops.estimate_time(acceleration=1000)

    # With acceleration, time should be longer for short moves
    assert time_est_with_accel > time_est_no_accel


def test_estimate_time_with_arc():
    """Test time estimation with arc commands."""
    ops = Ops()
    ops.move_to(10, 0)
    # 90 degree arc with radius 10 (quarter circle)
    ops.arc_to(0, 10, i=-10, j=0, clockwise=False)

    # Disable acceleration for simpler calculation
    time_est = ops.estimate_time(acceleration=0)

    # The actual distance is calculated as the straight line distance
    # from (10,0) to (0,10) which is sqrt(10^2 + 10^2) = 14.142mm
    # Plus the initial move_to(10, 0) which is 10mm travel
    # Expected: 14.142mm cut @ 1000mm/min = 0.8485s
    #         + 10mm travel  @ 3000mm/min = 0.2s
    #         = 1.0485s total
    expected_time = math.sqrt(10**2 + 10**2) / 1000 * 60 + 10.0 / 3000 * 60
    assert time_est == pytest.approx(expected_time, rel=1e-3)


def test_estimate_time_ignores_state_commands():
    """Test that state commands don't directly add to time."""
    ops = Ops()
    ops.move_to(0, 0)
    ops.set_power(0.5)  # State command
    ops.set_cut_speed(1000)  # State command
    ops.enable_air_assist()  # State command
    ops.line_to(60, 0)  # 60mm cut

    # Disable acceleration for simpler calculation
    time_est = ops.estimate_time(acceleration=0)

    # Only the line movement should count: 60mm @ 1000mm/min = 3.6s
    expected_time = 60.0 / 1000 * 60
    assert time_est == pytest.approx(expected_time, rel=1e-3)


def test_estimate_time_with_scanline():
    """Test time estimation with ScanLinePowerCommand."""
    ops = Ops()
    ops.move_to(0, 50)
    ops.scan_to(100, 50, 0, bytearray([100] * 100))

    # Disable acceleration for simpler calculation
    time_est = ops.estimate_time(acceleration=0)

    # The scanline command has 100 power values, which might be interpreted
    # as 100 segments of 1mm each, so total distance is 100mm
    # Plus the initial move_to(0, 50) which is 50mm travel
    # Expected: 100mm cut   @ 1000mm/min = 6s
    #         + 50mm travel @ 3000mm/min = 1s = 7s total
    expected_time = 100.0 / 1000 * 60 + 50.0 / 3000 * 60
    assert time_est == pytest.approx(expected_time, rel=1e-3)


def test_estimate_time_complex_path():
    """Test time estimation with a complex path."""
    ops = Ops()
    ops.move_to(0, 0)
    ops.set_cut_speed(1500)

    # Square
    ops.line_to(50, 0)  # 50mm
    ops.line_to(50, 50)  # 50mm
    ops.line_to(0, 50)  # 50mm
    ops.line_to(0, 0)  # 50mm

    # Move to new position
    ops.set_travel_speed(3000)
    ops.move_to(100, 0)  # 100mm travel

    # Another square
    ops.set_cut_speed(2000)
    ops.line_to(150, 0)  # 50mm
    ops.line_to(150, 50)  # 50mm

    # Disable acceleration for simpler calculation
    time_est = ops.estimate_time(acceleration=0)

    # First square: 200mm @ 1500mm/min = 8s
    # Travel: 100mm @ 3000mm/min = 2s
    # Second square: 100mm @ 2000mm/min = 3s
    # Total: 13s
    expected_time = 8.0 + 2.0 + 3.0
    assert time_est == pytest.approx(expected_time, rel=1e-3)


def test_numpy_serialization_round_trip_all_commands():
    """
    Tests the NumPy serialization round trip with all command types to ensure
    the hybrid serialization strategy works correctly.
    """
    ops = Ops()
    # Add one of each command type
    ops.job_start()  # Marker
    ops.layer_start("layer-1")  # Marker with data
    ops.set_travel_speed(6000)  # State with data
    ops.set_cut_speed(1500)  # State with data
    ops.set_power(0.75)  # State with data
    ops.enable_air_assist()  # State
    ops.set_laser("laser-xyz")  # State with data
    ops.move_to(1, 2, 3)  # Geometric
    ops.line_to(4, 5, 6)  # Geometric
    ops.arc_to(x=7, y=8, z=9, i=1, j=-1, clockwise=False)  # Geometric
    ops.bezier_to(
        c1=(8.0, 9.0, 10.0),
        c2=(9.0, 10.0, 11.0),
        end=(10.0, 11.0, 12.0),
    )
    ops.scan_to(10, 11, 12, bytearray([10, 20, 30]))  # Geometric
    ops.disable_air_assist()  # State
    ops.layer_end("layer-1")  # Marker with data
    ops.job_end()  # Marker

    # Perform the round trip
    arrays = ops.to_numpy_arrays()
    reconstructed_ops = Ops.from_numpy_arrays(arrays)

    # Assertions
    assert len(reconstructed_ops.commands) == len(ops.commands)
    for original_cmd, recon_cmd in zip(
        ops.commands, reconstructed_ops.commands
    ):
        assert type(original_cmd) is type(recon_cmd)
        # Use to_dict for a comprehensive comparison of all fields
        assert original_cmd.to_dict() == recon_cmd.to_dict()


def test_numpy_serialization_structure_hybrid():
    """
    Verifies the internal structure of the serialized arrays for a hybrid
    set of commands.
    """
    ops = Ops()
    ops.move_to(1, 1, 1)  # Geometric, index 0
    ops.set_power(0.5)  # State, index 1
    ops.line_to(2, 2, 2)  # Geometric, index 2

    arrays = ops.to_numpy_arrays()

    # Check that the JSON byte array exists and has content
    assert "state_marker_json_bytes" in arrays
    json_bytes = arrays["state_marker_json_bytes"]
    assert json_bytes.size > 0

    # Decode and verify the content
    json_str = json_bytes.tobytes().decode("utf-8")
    data = json.loads(json_str)

    # The dictionary should contain the data for the command at index 1
    assert "1" in data
    assert "0" not in data
    assert "2" not in data
    assert data["1"]["type"] == "SetPowerCommand"
    assert data["1"]["power"] == 0.5

    # Verify that geometric data is still correctly placed
    assert np.allclose(arrays["endpoints"][0], [1, 1, 1])
    assert np.allclose(arrays["endpoints"][2], [2, 2, 2])
    # The endpoint for the state command should be zero, as it's not used
    assert np.allclose(arrays["endpoints"][1], [0, 0, 0])


def test_numpy_serialization_round_trip_only_state():
    """Tests round-trip with only state/marker commands."""
    ops = Ops()
    ops.set_power(0.9)
    ops.set_laser("laser-abc")
    ops.layer_start("my-layer")

    arrays = ops.to_numpy_arrays()
    reconstructed_ops = Ops.from_numpy_arrays(arrays)

    assert len(reconstructed_ops.commands) == 3
    for original, recon in zip(ops.commands, reconstructed_ops.commands):
        assert original.to_dict() == recon.to_dict()


def test_numpy_serialization_round_trip_empty():
    """Tests round-trip with an empty Ops object."""
    ops = Ops()
    arrays = ops.to_numpy_arrays()
    reconstructed_ops = Ops.from_numpy_arrays(arrays)
    assert reconstructed_ops.is_empty()


def test_numpy_serialization_bezier_arrays():
    """Verifies bezier_data and bezier_map are populated correctly."""
    ops = Ops()
    ops.move_to(0, 0, 0)  # index 0
    ops.bezier_to(c1=(1, 2, 3), c2=(4, 5, 6), end=(7, 8, 9))  # index 1
    ops.line_to(10, 10, 10)  # index 2
    ops.bezier_to(
        c1=(11, 12, 13), c2=(14, 15, 16), end=(17, 18, 19)
    )  # index 3

    arrays = ops.to_numpy_arrays()

    assert "bezier_data" in arrays
    assert "bezier_map" in arrays
    assert arrays["bezier_data"].shape == (2, 6)
    assert arrays["bezier_map"][0] == -1
    assert arrays["bezier_map"][1] == 0
    assert arrays["bezier_map"][2] == -1
    assert arrays["bezier_map"][3] == 1

    np.testing.assert_allclose(arrays["bezier_data"][0], [1, 2, 3, 4, 5, 6])
    np.testing.assert_allclose(
        arrays["bezier_data"][1], [11, 12, 13, 14, 15, 16]
    )


def test_numpy_serialization_bezier_round_trip():
    """Tests that BezierToCommand survives numpy serialization."""
    ops = Ops()
    ops.move_to(0, 0, 0)
    ops.bezier_to(c1=(1.5, 2.5, 3.5), c2=(4.5, 5.5, 6.5), end=(7.5, 8.5, 9.5))
    ops.set_power(0.5)
    ops.add(
        BezierToCommand(
            end=(70, 80, 90),
            control1=(10, 20, 30),
            control2=(40, 50, 60),
        )
    )

    arrays = ops.to_numpy_arrays()
    reconstructed = Ops.from_numpy_arrays(arrays)

    assert len(reconstructed.commands) == 4
    assert isinstance(reconstructed.commands[0], MoveToCommand)
    assert isinstance(reconstructed.commands[1], BezierToCommand)
    assert isinstance(reconstructed.commands[2], SetPowerCommand)
    assert isinstance(reconstructed.commands[3], BezierToCommand)

    cmd1 = reconstructed.commands[1]
    assert cmd1.end == pytest.approx((7.5, 8.5, 9.5))
    assert cmd1.control1 == pytest.approx((1.5, 2.5, 3.5))
    assert cmd1.control2 == pytest.approx((4.5, 5.5, 6.5))

    cmd3 = reconstructed.commands[3]
    assert cmd3.end == pytest.approx((70, 80, 90))
    assert cmd3.control1 == pytest.approx((10, 20, 30))
    assert cmd3.control2 == pytest.approx((40, 50, 60))


def test_linearize_curves():
    """Tests that linearize_curves replaces beziers with lines."""
    ops = Ops()
    ops.move_to(0, 0, 0)
    ops.bezier_to(c1=(10, 0, 0), c2=(10, 10, 0), end=(0, 10, 0))
    ops.set_power(0.8)
    ops.line_to(5, 5, 0)

    assert isinstance(ops.commands[1], BezierToCommand)
    ops.linearize_curves()

    assert not any(isinstance(c, CurveToCommand) for c in ops.commands)
    assert isinstance(ops.commands[0], MoveToCommand)
    assert all(
        isinstance(c, LineToCommand)
        for c in ops.commands[1:-1]
        if isinstance(c, MovingCommand)
    )
    assert isinstance(ops.commands[-2], SetPowerCommand)
    assert isinstance(ops.commands[-1], LineToCommand)
    assert ops.commands[-1].end == (5, 5, 0)


def test_linearize_curves_preserves_arcs():
    """Tests that linearize_curves does not touch arcs."""
    ops = Ops()
    ops.move_to(10, 0)
    ops.arc_to(0, 10, i=-10, j=0, clockwise=False)
    ops.bezier_to(c1=(0, 10, 0), c2=(-10, 10, 0), end=(-10, 0, 0))

    ops.linearize_curves()

    assert isinstance(ops.commands[1], ArcToCommand)
    assert isinstance(ops.commands[-1], LineToCommand)
    assert not any(isinstance(c, CurveToCommand) for c in ops.commands)


def test_linearize_arcs():
    """Tests that linearize_arcs replaces arcs with lines."""
    ops = Ops()
    ops.move_to(10, 0)
    ops.line_to(20, 0)
    ops.arc_to(10, 10, i=-10, j=0, clockwise=False)
    ops.set_power(1.0)
    ops.move_to(10, 10)
    ops.bezier_to(c1=(10, 10, 0), c2=(20, 10, 0), end=(20, 0, 0))


def test_linearize_arcs_preserves_beziers():
    """Tests that linearize_arcs does not touch bezier curves."""
    ops = Ops()
    ops.move_to(0, 0)
    ops.bezier_to(c1=(10, 0, 0), c2=(10, 10, 0), end=(0, 10, 0))
    ops.set_power(0.8)

    ops.linearize_arcs()

    bezier_cmds = [c for c in ops.commands if isinstance(c, BezierToCommand)]
    assert len(bezier_cmds) == 1
    assert bezier_cmds[0].control1 == (10, 0, 0)
    assert bezier_cmds[0].control2 == (10, 10, 0)
    assert bezier_cmds[0].end == (0, 10, 0)


class TestIterSections:
    """Tests for Ops.iter_sections()."""

    def test_empty_ops(self):
        ops = Ops()
        assert list(ops.iter_sections()) == []

    def test_no_sections(self):
        ops = Ops()
        ops.move_to(0, 0)
        ops.line_to(10, 10)
        ops.set_power(0.5)

        sections = list(ops.iter_sections())
        assert len(sections) == 1
        assert sections[0].section_type is None
        assert sections[0].markers == []
        assert len(sections[0].commands) == 3

    def test_single_section(self):
        ops = Ops()
        ops.ops_section_start(SectionType.VECTOR_OUTLINE, "wp-1")
        ops.move_to(0, 0)
        ops.line_to(10, 10)
        ops.ops_section_end(SectionType.VECTOR_OUTLINE)

        sections = list(ops.iter_sections())
        assert len(sections) == 1
        assert sections[0].section_type == SectionType.VECTOR_OUTLINE
        assert len(sections[0].markers) == 2
        assert isinstance(sections[0].markers[0], OpsSectionStartCommand)
        assert isinstance(sections[0].markers[1], OpsSectionEndCommand)
        assert len(sections[0].commands) == 2

    def test_multiple_sections(self):
        ops = Ops()
        ops.ops_section_start(SectionType.VECTOR_OUTLINE, "wp-1")
        ops.move_to(0, 0)
        ops.line_to(10, 0)
        ops.ops_section_end(SectionType.VECTOR_OUTLINE)

        ops.ops_section_start(SectionType.RASTER_FILL, "wp-1")
        ops.move_to(0, 5)
        ops.line_to(10, 5)
        ops.ops_section_end(SectionType.RASTER_FILL)

        sections = list(ops.iter_sections())
        assert len(sections) == 2
        assert sections[0].section_type == SectionType.VECTOR_OUTLINE
        assert sections[1].section_type == SectionType.RASTER_FILL
        assert len(sections[0].commands) == 2
        assert len(sections[1].commands) == 2

    def test_commands_before_section(self):
        ops = Ops()
        ops.set_power(0.8)
        ops.ops_section_start(SectionType.VECTOR_OUTLINE, "wp-1")
        ops.move_to(0, 0)
        ops.line_to(10, 10)
        ops.ops_section_end(SectionType.VECTOR_OUTLINE)

        sections = list(ops.iter_sections())
        assert len(sections) == 2
        assert sections[0].section_type is None
        assert sections[0].markers == []
        assert len(sections[0].commands) == 1
        assert isinstance(sections[0].commands[0], SetPowerCommand)
        assert sections[1].section_type == SectionType.VECTOR_OUTLINE

    def test_commands_after_section(self):
        ops = Ops()
        ops.ops_section_start(SectionType.VECTOR_OUTLINE, "wp-1")
        ops.move_to(0, 0)
        ops.line_to(10, 10)
        ops.ops_section_end(SectionType.VECTOR_OUTLINE)
        ops.set_power(0.5)

        sections = list(ops.iter_sections())
        assert len(sections) == 2
        assert sections[0].section_type == SectionType.VECTOR_OUTLINE
        assert sections[1].section_type is None
        assert sections[1].markers == []
        assert len(sections[1].commands) == 1
        assert isinstance(sections[1].commands[0], SetPowerCommand)

    def test_commands_between_sections(self):
        ops = Ops()
        ops.ops_section_start(SectionType.VECTOR_OUTLINE, "wp-1")
        ops.move_to(0, 0)
        ops.line_to(5, 0)
        ops.ops_section_end(SectionType.VECTOR_OUTLINE)

        ops.set_power(0.5)

        ops.ops_section_start(SectionType.RASTER_FILL, "wp-1")
        ops.move_to(0, 5)
        ops.line_to(5, 5)
        ops.ops_section_end(SectionType.RASTER_FILL)

        sections = list(ops.iter_sections())
        assert len(sections) == 3
        assert sections[0].section_type == SectionType.VECTOR_OUTLINE
        assert sections[1].section_type is None
        assert sections[1].markers == []
        assert isinstance(sections[1].commands[0], SetPowerCommand)
        assert sections[2].section_type == SectionType.RASTER_FILL

    def test_section_markers_preserved(self):
        ops = Ops()
        start_cmd = OpsSectionStartCommand(SectionType.VECTOR_OUTLINE, "wp-1")
        end_cmd = OpsSectionEndCommand(SectionType.VECTOR_OUTLINE)
        ops.commands.append(start_cmd)
        ops.move_to(0, 0)
        ops.line_to(10, 10)
        ops.commands.append(end_cmd)

        sections = list(ops.iter_sections())
        assert len(sections) == 1
        assert sections[0].markers[0] is start_cmd
        assert sections[0].markers[1] is end_cmd

    def test_empty_section(self):
        ops = Ops()
        ops.ops_section_start(SectionType.VECTOR_OUTLINE, "wp-1")
        ops.ops_section_end(SectionType.VECTOR_OUTLINE)

        sections = list(ops.iter_sections())
        assert len(sections) == 1
        assert sections[0].section_type == SectionType.VECTOR_OUTLINE
        assert sections[0].commands == []
        assert len(sections[0].markers) == 2

    def test_consecutive_empty_sections(self):
        ops = Ops()
        ops.ops_section_start(SectionType.VECTOR_OUTLINE, "wp-1")
        ops.ops_section_end(SectionType.VECTOR_OUTLINE)
        ops.ops_section_start(SectionType.RASTER_FILL, "wp-1")
        ops.ops_section_end(SectionType.RASTER_FILL)

        sections = list(ops.iter_sections())
        assert len(sections) == 2
        assert sections[0].section_type == SectionType.VECTOR_OUTLINE
        assert sections[0].commands == []
        assert sections[1].section_type == SectionType.RASTER_FILL
        assert sections[1].commands == []

    def test_commands_reconstructed_correctly(self):
        ops = Ops()
        ops.set_power(1.0)
        ops.ops_section_start(SectionType.VECTOR_OUTLINE, "wp-1")
        ops.move_to(0, 0)
        ops.line_to(10, 10)
        ops.line_to(20, 0)
        ops.ops_section_end(SectionType.VECTOR_OUTLINE)
        ops.set_power(0.5)

        sections = list(ops.iter_sections())
        reconstructed = []
        for section in sections:
            if section.section_type is not None:
                reconstructed.append(section.markers[0])
                reconstructed.extend(section.commands)
                if len(section.markers) > 1:
                    reconstructed.append(section.markers[1])
            else:
                reconstructed.extend(section.commands)

        assert reconstructed == ops.commands

    def test_returns_ops_section_namedtuple(self):
        ops = Ops()
        ops.move_to(0, 0)
        section = list(ops.iter_sections())[0]
        assert isinstance(section, OpsSection)
        assert hasattr(section, "section_type")
        assert hasattr(section, "markers")
        assert hasattr(section, "commands")
