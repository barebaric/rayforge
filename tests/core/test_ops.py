import pytest
import math
import numpy as np
from typing import cast
from rayforge.core.ops import (
    Ops,
    MoveToCommand,
    LineToCommand,
    ArcToCommand,
    SetPowerCommand,
    SetCutSpeedCommand,
    SetTravelSpeedCommand,
    EnableAirAssistCommand,
    DisableAirAssistCommand,
    State,
    MovingCommand,
)


def _create_translate_matrix(x, y, z):
    """Creates a NumPy translation matrix."""
    return np.array(
        [
            [1, 0, 0, x],
            [0, 1, 0, y],
            [0, 0, 1, z],
            [0, 0, 0, 1],
        ],
        dtype=float,
    )


def _create_scale_matrix(sx, sy, sz):
    """Creates a NumPy scaling matrix."""
    return np.array(
        [
            [sx, 0, 0, 0],
            [0, sy, 0, 0],
            [0, 0, sz, 0],
            [0, 0, 0, 1],
        ],
        dtype=float,
    )


def _create_z_rotate_matrix(angle_rad):
    """Creates a NumPy Z-axis rotation matrix."""
    c = math.cos(angle_rad)
    s = math.sin(angle_rad)
    return np.array(
        [
            [c, -s, 0, 0],
            [s, c, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ],
        dtype=float,
    )


@pytest.fixture
def empty_ops():
    return Ops()


@pytest.fixture
def sample_ops():
    ops = Ops()
    ops.move_to(0, 0)
    ops.line_to(10, 10)
    ops.set_power(500)
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


def test_move_to_3d(sample_ops):
    sample_ops.move_to(15, 15, -5.0)
    last_cmd = sample_ops.commands[-1]
    assert isinstance(last_cmd, MoveToCommand)
    assert last_cmd.end == (15.0, 15.0, -5.0)


def test_line_to(sample_ops):
    sample_ops.line_to(20, 20)
    last_cmd = sample_ops.commands[-1]
    assert isinstance(last_cmd, LineToCommand)
    assert last_cmd.end == (20.0, 20.0, 0.0)


def test_line_to_3d(sample_ops):
    sample_ops.line_to(20, 20, -2.5)
    last_cmd = sample_ops.commands[-1]
    assert isinstance(last_cmd, LineToCommand)
    assert last_cmd.end == (20.0, 20.0, -2.5)


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


def test_arc_to_3d(sample_ops):
    sample_ops.arc_to(5, 5, 2, 3, clockwise=False, z=-10.0)
    last_cmd = sample_ops.commands[-1]
    assert isinstance(last_cmd, ArcToCommand)
    assert last_cmd.end == (5.0, 5.0, -10.0)


def test_set_power(sample_ops):
    sample_ops.set_power(800)
    last_cmd = sample_ops.commands[-1]
    assert isinstance(last_cmd, SetPowerCommand)
    assert last_cmd.power == 800.0


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


def test_enable_disable_air_assist(empty_ops):
    empty_ops.enable_air_assist()
    assert isinstance(empty_ops.commands[-1], EnableAirAssistCommand)

    empty_ops.disable_air_assist()
    assert isinstance(empty_ops.commands[-1], DisableAirAssistCommand)


def test_get_frame(sample_ops):
    frame = sample_ops.get_frame(power=1000, speed=500)
    assert sum(1 for c in frame if c.is_travel_command()) == 1  # move_to
    assert sum(1 for c in frame if c.is_cutting_command()) == 4  # line_to

    occupied_points = [(0, 0, 0), (10, 10, 0)]
    xs = [p[0] for p in occupied_points]
    ys = [p[1] for p in occupied_points]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)

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
    sample_ops.line_to(10, 10, -10)  # Cut with Z change
    cut_distance = sample_ops.cut_distance()
    # Distance should be 2D
    expected = math.dist((0, 0), (10, 10))
    assert cut_distance == pytest.approx(expected)


def test_segments(sample_ops):
    sample_ops.move_to(5, 5)  # Travel command
    segments = list(sample_ops.segments())
    assert len(segments) > 0
    # First segment should end before the travel command
    assert isinstance(segments[0][-1], LineToCommand)


def test_state_allow_rapid_change():
    state1 = State(air_assist=True)
    state2 = State(air_assist=True)
    assert state1.allow_rapid_change(state2)

    state3 = State(air_assist=False)
    assert not state1.allow_rapid_change(state3)


def test_arc_to_cutting_command():
    arc_cmd = ArcToCommand((5, 5, 0), (2, 3), True)
    assert arc_cmd.is_cutting_command()


def test_command_inheritance():
    move_cmd = MoveToCommand((0, 0, 0))
    assert move_cmd.is_travel_command()
    assert not move_cmd.is_cutting_command()

    line_cmd = LineToCommand((5, 5, 0))
    assert line_cmd.is_cutting_command()
    assert not line_cmd.is_travel_command()


def test_preload_state_application():
    ops = Ops()
    ops.set_power(300)
    ops.line_to(5, 5)
    ops.set_cut_speed(200)
    ops.preload_state()

    line_cmd = ops.commands[1]
    assert line_cmd.state is not None
    assert line_cmd.state.power == 300

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
    # Use a valid arc where start and end points have the same distance to
    # center start=(10,20), center=(15,27), end=(22,22), r^2=74
    ops.arc_to(22, 22, 5, 7, z=-10)
    ops.scale(2, 3, 4)  # Non-uniform scale

    # First command is the scaled move_to
    assert ops.commands[0].end is not None
    assert ops.commands[0].end == pytest.approx((20, 60, 20))

    # Non-uniform scale linearizes the arc into LineToCommands
    assert isinstance(ops.commands[1], LineToCommand)

    # The final point of the arc should be the scaled original end point
    final_cmd = ops.commands[-1]
    assert final_cmd.end is not None
    final_point = final_cmd.end
    expected_final_point = (22 * 2, 22 * 3, -10 * 4)
    assert final_point == pytest.approx(expected_final_point)

    # last_move_to should also be scaled
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


# --- Tests for Ops.clip() ---


@pytest.fixture
def clip_rect():
    return (0.0, 0.0, 100.0, 100.0)


def test_clip_fully_inside(clip_rect):
    ops = Ops()
    ops.move_to(10, 10, -1)
    ops.line_to(90, 90, -1)
    clipped_ops = ops.clip(clip_rect)
    assert len(clipped_ops.commands) == 2
    assert isinstance(clipped_ops.commands[0], MoveToCommand)
    assert isinstance(clipped_ops.commands[1], LineToCommand)
    assert clipped_ops.commands[1].end is not None
    assert clipped_ops.commands[1].end == pytest.approx((90.0, 90.0, -1.0))


def test_clip_fully_outside(clip_rect):
    ops = Ops()
    ops.move_to(110, 110, 0)
    ops.line_to(120, 120, 0)
    clipped_ops = ops.clip(clip_rect)
    assert len(clipped_ops.commands) == 0


def test_clip_line_crossing_one_boundary(clip_rect):
    ops = Ops()
    ops.move_to(50, 50, 0)
    ops.line_to(150, 50, 0)  # Exits right
    clipped_ops = ops.clip(clip_rect)
    assert len(clipped_ops.commands) == 2
    assert clipped_ops.commands[1].end is not None
    assert clipped_ops.commands[1].end == pytest.approx((100.0, 50.0, 0.0))


def test_clip_line_crossing_two_boundaries(clip_rect):
    ops = Ops()
    ops.move_to(-50, 50, 0)  # Starts left
    ops.line_to(150, 50, 0)  # Exits right
    clipped_ops = ops.clip(clip_rect)
    assert len(clipped_ops.commands) == 2
    assert clipped_ops.commands[0].end is not None
    assert clipped_ops.commands[1].end is not None
    assert clipped_ops.commands[0].end == pytest.approx((0.0, 50.0, 0.0))
    assert clipped_ops.commands[1].end == pytest.approx((100.0, 50.0, 0.0))


def test_clip_interpolates_z(clip_rect):
    ops = Ops()
    ops.move_to(50, -50, -10)  # Starts below
    ops.line_to(50, 150, 10)  # Exits above
    clipped_ops = ops.clip(clip_rect)
    assert len(clipped_ops.commands) == 2
    assert clipped_ops.commands[0].end is not None
    assert clipped_ops.commands[1].end is not None
    assert clipped_ops.commands[0].end == pytest.approx(
        (50.0, 0.0, -5.0)
    )  # Z should be halfway
    assert clipped_ops.commands[1].end == pytest.approx(
        (50.0, 100.0, 5.0)
    )  # Z should be 3/4 of the way


def test_clip_arc_partially_inside(clip_rect):
    ops = Ops()
    ops.move_to(100, 50, 0)  # Start on right edge
    ops.arc_to(50, 100, i=-50, j=0, z=-10)  # Arc to top edge with z change
    clipped_ops = ops.clip(clip_rect)
    # Result will be a series of line segments
    assert len(clipped_ops.commands) > 2
    all_points_in = all(
        c.end is not None
        and 0 <= c.end[0] <= 100.001
        and 0 <= c.end[1] <= 100.001
        for c in clipped_ops.commands
        if isinstance(c, MovingCommand)
    )
    assert all_points_in


def test_clip_preserves_state_commands(clip_rect):
    ops = Ops()
    ops.set_power(500)
    ops.move_to(-10, 50, 0)
    ops.line_to(110, 50, 0)
    ops.set_cut_speed(1000)
    clipped_ops = ops.clip(clip_rect)
    state_cmds = [c for c in clipped_ops.commands if c.is_state_command()]
    assert len(state_cmds) == 2
    assert isinstance(state_cmds[0], SetPowerCommand)
    assert isinstance(state_cmds[1], SetCutSpeedCommand)


def test_clip_arc_reentering_visible_area(clip_rect):
    ops = Ops()
    ops.move_to(10, 90)
    ops.arc_to(90, 90, i=40, j=-20, clockwise=True)  # Max y > 100
    clipped_ops = ops.clip(clip_rect)
    move_tos = [
        c for c in clipped_ops.commands if isinstance(c, MoveToCommand)
    ]
    assert len(move_tos) == 2
    for cmd in clipped_ops.commands:
        if isinstance(cmd, MovingCommand):
            assert cmd.end is not None
            assert 0 <= cmd.end[0] <= 100.000001
            assert 0 <= cmd.end[1] <= 100.000001


def test_transform_identity():
    ops = Ops()
    ops.move_to(10, 20, 30)
    ops.arc_to(50, 60, i=5, j=7, z=40)
    original_ops = ops.copy()

    # The constructor with no arguments creates an identity matrix.
    identity_matrix = np.identity(4, dtype=float)
    ops.transform(identity_matrix)

    arc_cmd = cast(ArcToCommand, ops.commands[1])
    orig_arc_cmd = cast(ArcToCommand, original_ops.commands[1])

    assert ops.commands[0].end == pytest.approx(original_ops.commands[0].end)
    assert arc_cmd.end == pytest.approx(orig_arc_cmd.end)
    assert arc_cmd.center_offset == pytest.approx(orig_arc_cmd.center_offset)
    assert ops.last_move_to == pytest.approx(original_ops.last_move_to)


def test_transform_translate():
    ops = Ops()
    ops.move_to(10, 20, 30)
    ops.arc_to(50, 60, i=5, j=7, z=40)

    # Use the factory function
    translate_matrix = _create_translate_matrix(10, -5, 15)
    ops.transform(translate_matrix)
    arc_cmd = cast(ArcToCommand, ops.commands[1])

    # End points should be translated
    assert ops.commands[0].end == pytest.approx((20, 15, 45))
    assert arc_cmd.end == pytest.approx((60, 55, 55))

    # Center offset is a vector and should NOT be translated
    assert arc_cmd.center_offset == pytest.approx((5, 7))
    assert ops.last_move_to == pytest.approx((20, 15, 45))


def test_transform_scale():
    ops = Ops()
    ops.move_to(10, 20, 5)
    # Use a valid arc where start and end points have the same distance to
    # center start=(10,20), center=(15,27), end=(22,22), r^2=74
    ops.arc_to(22, 22, i=5, j=7, z=-10)
    scale_matrix = _create_scale_matrix(2, 3, 4)
    ops.transform(scale_matrix)

    assert ops.commands[0].end == pytest.approx((20, 60, 20))
    # Arcs are linearized on non-uniform scale, so there are many line commands
    assert isinstance(ops.commands[1], LineToCommand)
    final_cmd = ops.commands[-1]
    assert final_cmd.end is not None
    final_point = final_cmd.end
    expected_final_point = (22 * 2, 22 * 3, -10 * 4)
    assert final_point == pytest.approx(expected_final_point)


def test_transform_rotate():
    ops = Ops()
    ops.move_to(10, 0, -5)
    ops.arc_to(20, 0, i=5, j=0, z=-5)
    rotate_matrix = _create_z_rotate_matrix(math.radians(90))
    ops.transform(rotate_matrix)
    arc_cmd = cast(ArcToCommand, ops.commands[1])

    # Check move_to end point
    assert ops.commands[0].end is not None
    x0, y0, z0 = ops.commands[0].end
    assert x0 == pytest.approx(0)
    assert y0 == pytest.approx(10)
    assert z0 == -5

    # Check arc_to end point
    assert arc_cmd.end is not None
    x1, y1, z1 = arc_cmd.end
    assert x1 == pytest.approx(0)
    assert y1 == pytest.approx(20)
    assert z1 == -5

    # Check arc_to center offset vector
    i, j = arc_cmd.center_offset
    assert i == pytest.approx(0)
    assert j == pytest.approx(5)


def test_transform_shear():
    """Tests a combined scale and rotate matrix which results in shear."""
    ops = Ops()
    ops.move_to(10, 0, 0)
    ops.line_to(10, 10, 0)
    ops.arc_to(20, 10, i=5, j=0, z=0)

    # Create a shear-inducing matrix by scaling then rotating
    scale_m = _create_scale_matrix(2, 1, 1)
    rotate_m = _create_z_rotate_matrix(math.radians(45))
    shear_m = rotate_m @ scale_m  # Apply scale first, then rotate

    # Manually calculate expected points for clarity
    p0_vec = shear_m @ np.array([10, 0, 0, 1])
    p1_vec = shear_m @ np.array([10, 10, 0, 1])
    p2_vec = shear_m @ np.array([20, 10, 0, 1])

    ops.transform(shear_m)

    assert ops.commands[0].end is not None
    assert ops.commands[1].end is not None
    assert ops.commands[0].end[0] == pytest.approx(p0_vec[0])
    assert ops.commands[0].end[1] == pytest.approx(p0_vec[1])
    assert ops.commands[1].end[0] == pytest.approx(p1_vec[0])
    assert ops.commands[1].end[1] == pytest.approx(p1_vec[1])
    # Arc is linearized into many LineToCommands
    assert isinstance(ops.commands[2], LineToCommand)
    final_cmd = ops.commands[-1]
    assert final_cmd.end is not None
    final_point = final_cmd.end
    assert final_point[0] == pytest.approx(p2_vec[0])
    assert final_point[1] == pytest.approx(p2_vec[1])
