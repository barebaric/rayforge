import pytest
import math
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
    OpsSectionStartCommand,
    OpsSectionEndCommand,
    SectionType,
    ScanLinePowerCommand,
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
        num_steps=2,
    )

    assert len(ops.commands) == 3  # move_to + 2 line_to
    assert isinstance(ops.commands[0], MoveToCommand)
    assert isinstance(ops.commands[1], LineToCommand)
    assert isinstance(ops.commands[2], LineToCommand)

    # Check interpolated points
    # t=0.5
    # B(0.5) = 0.125*p0 + 0.375*c1 + 0.375*c2 + 0.125*p1
    p0 = (0.0, 0.0, 10.0)
    c1 = (1.0, 1.0, 10.0)
    c2 = (2.0, 1.0, 20.0)
    p1 = (3.0, 0.0, 20.0)
    expected_x = 0.125 * p0[0] + 0.375 * c1[0] + 0.375 * c2[0] + 0.125 * p1[0]
    expected_y = 0.125 * p0[1] + 0.375 * c1[1] + 0.375 * c2[1] + 0.125 * p1[1]
    expected_z = 0.125 * p0[2] + 0.375 * c1[2] + 0.375 * c2[2] + 0.125 * p1[2]
    # This is the endpoint of the first segment (t=0.5).
    assert ops.commands[1].end == pytest.approx(
        (expected_x, expected_y, expected_z)
    )

    # Check final point (t=1.0)
    assert ops.commands[2].end == pytest.approx(p1)


def test_bezier_to_no_start_point():
    ops = Ops()
    # Should not raise an error, just do nothing and log a warning.
    ops.bezier_to((1, 1, 1), (2, 2, 2), (3, 3, 3))
    assert ops.is_empty()


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


def test_clip_with_arc():
    """Verify clip works on arcs via the new generic linearize interface."""
    ops = Ops()
    ops.move_to(0, 50)
    ops.arc_to(100, 50, i=50, j=0, clockwise=False)  # Semicircle
    clip_rect = (40.0, 0.0, 60.0, 100.0)  # A vertical slice through the middle
    clipped_ops = ops.clip(clip_rect)

    # Check that there are drawing commands left
    drawing_cmds = [c for c in clipped_ops if c.is_cutting_command()]
    assert len(drawing_cmds) > 0

    # Check that all remaining points are within the rect bounds
    for cmd in clipped_ops:
        if isinstance(cmd, MovingCommand):
            x, y, z = cmd.end
            assert clip_rect[0] <= x <= clip_rect[2]
            assert clip_rect[1] <= y <= clip_rect[3]


def test_clip_scanlinepowercommand_start_outside():
    """Tests clipping a scanline that starts outside and ends inside."""
    ops = Ops()
    ops.add(
        ScanLinePowerCommand(
            start_point=(0, 50, 10),
            end=(100, 50, 10),
            power_values=bytearray(range(100)),
        )
    )
    clip_rect = (50, 0, 150, 100)
    clipped_ops = ops.clip(clip_rect)

    assert len(clipped_ops.commands) == 2  # MoveTo, ScanLinePowerCommand
    assert isinstance(clipped_ops.commands[0], MoveToCommand)
    clipped_cmd = cast(ScanLinePowerCommand, clipped_ops.commands[1])

    # 1. Verify it's still a ScanLinePowerCommand (not linearized)
    assert isinstance(clipped_cmd, ScanLinePowerCommand)

    # 2. Verify new geometry
    assert clipped_cmd.start_point == pytest.approx((50, 50, 10))
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
    ops.add(
        ScanLinePowerCommand(
            start_point=(-50, 50, 0),
            end=(150, 50, 200),
            power_values=bytearray(range(200)),
        )
    )
    clip_rect = (0, 0, 100, 100)
    clipped_ops = ops.clip(clip_rect)

    assert len(clipped_ops.commands) == 2
    clipped_cmd = cast(ScanLinePowerCommand, clipped_ops.commands[1])
    assert isinstance(clipped_cmd, ScanLinePowerCommand)

    # The line starts 50 units before x=0 and ends 50 units after x=100.
    # The clipped portion is from x=0 to x=100.
    # t_start = 50 / 200 = 0.25. t_end = 150 / 200 = 0.75
    expected_z_start = 0 + (0.25 * 200)  # 50
    expected_z_end = 0 + (0.75 * 200)  # 150

    assert clipped_cmd.start_point == pytest.approx((0, 50, expected_z_start))
    assert clipped_cmd.end == pytest.approx((100, 50, expected_z_end))

    # Power values should be sliced from index 50 to 150.
    expected_len = int(200 * 0.75) - int(200 * 0.25)
    assert len(clipped_cmd.power_values) == expected_len
    assert clipped_cmd.power_values[0] == 50
    assert clipped_cmd.power_values[-1] == 149


def test_clip_scanlinepowercommand_fully_outside():
    """Tests that a fully outside scanline is removed."""
    ops = Ops()
    ops.add(
        ScanLinePowerCommand(
            start_point=(200, 50, 10),
            end=(300, 50, 10),
            power_values=bytearray(range(100)),
        )
    )
    clip_rect = (0, 0, 100, 100)
    clipped_ops = ops.clip(clip_rect)
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


def test_serialization_with_section_markers():
    ops = Ops()
    ops.move_to(0, 0)
    ops.add(OpsSectionStartCommand(SectionType.RASTER_FILL, "wp-abc"))
    ops.line_to(10, 0)
    ops.add(OpsSectionEndCommand(SectionType.RASTER_FILL))

    data = ops.to_dict()
    new_ops = Ops.from_dict(data)

    assert len(new_ops.commands) == 4
    start_cmd = cast(OpsSectionStartCommand, new_ops.commands[1])
    end_cmd = cast(OpsSectionEndCommand, new_ops.commands[3])
    assert isinstance(start_cmd, OpsSectionStartCommand)
    assert isinstance(end_cmd, OpsSectionEndCommand)
    assert start_cmd.section_type == SectionType.RASTER_FILL
    assert start_cmd.workpiece_uid == "wp-abc"
    assert end_cmd.section_type == SectionType.RASTER_FILL


def test_translate_with_scanline():
    """Tests that translate() correctly transforms ScanLinePowerCommand."""
    ops = Ops()
    ops.add(
        ScanLinePowerCommand(
            start_point=(10, 20, 30),
            end=(40, 50, 60),
            power_values=bytearray([1, 2, 3]),
        )
    )
    ops.translate(5, -10, 15)

    translated_cmd = cast(ScanLinePowerCommand, ops.commands[0])
    assert isinstance(translated_cmd, ScanLinePowerCommand)

    # Check if both start_point and end are translated
    assert translated_cmd.start_point == pytest.approx((15, 10, 45))
    assert translated_cmd.end == pytest.approx((45, 40, 75))
