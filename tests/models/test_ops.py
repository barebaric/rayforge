import pytest
import math
from rayforge.models.ops import (
    Ops, MoveToCommand, LineToCommand,
    ArcToCommand, SetPowerCommand, SetCutSpeedCommand,
    SetTravelSpeedCommand, EnableAirAssistCommand,
    DisableAirAssistCommand, State
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
    assert last_cmd.end == (15.0, 15.0)


def test_line_to(sample_ops):
    sample_ops.line_to(20, 20)
    last_cmd = sample_ops.commands[-1]
    assert isinstance(last_cmd, LineToCommand)
    assert last_cmd.end == (20.0, 20.0)


def test_close_path(sample_ops):
    sample_ops.close_path()
    last_cmd = sample_ops.commands[-1]
    assert isinstance(last_cmd, LineToCommand)
    assert last_cmd.end == sample_ops.last_move_to


def test_arc_to(sample_ops):
    sample_ops.arc_to(5, 5, 2, 3, clockwise=False)
    last_cmd = sample_ops.commands[-1]
    assert isinstance(last_cmd, ArcToCommand)
    assert last_cmd.end == (5.0, 5.0)
    assert last_cmd.clockwise is False


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

    # Check frame coordinates
    occupied_points = [(0, 0), (10, 10)]
    xs = [p[0] for p in occupied_points]
    ys = [p[1] for p in occupied_points]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)

    # Check if the frame commands are correctly ordered
    expected_points = [
        (min_x, min_y),
        (min_x, max_y),
        (max_x, max_y),
        (max_x, min_y),
        (min_x, min_y)
    ]

    # Skip state commands (first two)
    frame_points = [cmd.end for cmd in frame.commands[2:]]
    assert frame_points == expected_points


def test_get_frame_empty(empty_ops):
    frame = empty_ops.get_frame()
    assert len(frame.commands) == 0


def test_distance(sample_ops):
    # Add a travel command
    sample_ops.move_to(20, 20)
    distance = sample_ops.distance()
    expected = math.dist((0, 0), (10, 10)) + math.dist((10, 10), (20, 20))
    assert distance == pytest.approx(expected)


def test_cut_distance(sample_ops):
    cut_distance = sample_ops.cut_distance()
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
    arc_cmd = ArcToCommand((5, 5), (2, 3), True)
    assert arc_cmd.is_cutting_command()


def test_command_inheritance():
    move_cmd = MoveToCommand((0, 0))
    assert move_cmd.is_travel_command()
    assert not move_cmd.is_cutting_command()

    line_cmd = LineToCommand((5, 5))
    assert line_cmd.is_cutting_command()
    assert not line_cmd.is_travel_command()


def test_preload_state_application():
    ops = Ops()
    ops.set_power(300)  # State command
    ops.line_to(5, 5)   # Non-state command
    ops.set_cut_speed(200)  # State command
    ops.preload_state()

    # After preload, the line_to command should have the state with power 300
    line_cmd = ops.commands[1]
    assert line_cmd.state.power == 300  # Verify state is applied correctly

    # Verify that state commands are still present
    state_commands = [cmd for cmd in ops.commands if cmd.is_state_command()]
    assert len(state_commands) == 2  # set_power and set_cut_speed are still present

    # Verify that non-state commands have their state attribute set
    for cmd in ops.commands:
        if not cmd.is_state_command():
            assert cmd.state is not None
            assert isinstance(cmd.state, State)

def test_translate():
    ops = Ops()
    ops.move_to(10, 20)
    ops.line_to(30, 40)
    ops.arc_to(50, 60, 5, 7)
    ops.translate(5, 10)

    assert ops.commands[0].end == (15, 30)      # MoveTo
    assert ops.commands[1].end == (35, 50)      # LineTo
    assert ops.commands[2].end == (55, 70)      # ArcTo endpoint
    assert ops.commands[2].center_offset == (5, 7)  # ArcTo center offset
    assert ops.last_move_to == (15, 30)

def test_arc_translation():
    ops = Ops()
    ops.move_to(10, 10)
    ops.arc_to(20, 20, 5, 5)  # Center at (15, 15)
    ops.translate(5, 5)

    # Endpoint translated
    assert ops.commands[1].end == (25, 25)
    # Offset remains same
    assert ops.commands[1].center_offset == (5, 5)
    # Implicit center: (10+5+5, 10+5+5) = (20, 20)

def test_scale():
    ops = Ops()
    ops.move_to(10, 20)
    ops.line_to(30, 40)
    ops.arc_to(50, 60, 5, 7)
    ops.scale(2, 3)

    assert ops.commands[0].end == (20, 60)      # MoveTo
    assert ops.commands[1].end == (60, 120)     # LineTo
    assert ops.commands[2].end == (100, 180)    # ArcTo endpoint
    assert ops.commands[2].center_offset == (10, 21)  # ArcTo center offset
    assert ops.last_move_to == (20, 60)

def test_arc_scaling():
    ops = Ops()
    ops.move_to(10, 10)
    ops.arc_to(20, 20, 5, 5)
    ops.scale(2, 3)

    # Endpoint scaled
    assert ops.commands[1].end == (40, 60)
    # Offset scaled
    assert ops.commands[1].center_offset == (10, 15)
    # Implicit center: (20 + 10, 30 + 15) = (30, 45)

# --- Tests for Ops.clip() ---

@pytest.fixture
def clip_rect():
    return (0.0, 0.0, 100.0, 100.0)

def test_clip_fully_inside(clip_rect):
    ops = Ops()
    ops.move_to(10, 10)
    ops.line_to(90, 90)
    clipped_ops = ops.clip(clip_rect)
    assert len(clipped_ops.commands) == 2
    assert isinstance(clipped_ops.commands[0], MoveToCommand)
    assert isinstance(clipped_ops.commands[1], LineToCommand)

def test_clip_fully_outside(clip_rect):
    ops = Ops()
    ops.move_to(110, 110)
    ops.line_to(120, 120)
    clipped_ops = ops.clip(clip_rect)
    assert len(clipped_ops.commands) == 0

def test_clip_line_crossing_one_boundary(clip_rect):
    ops = Ops()
    ops.move_to(50, 50)
    ops.line_to(150, 50) # Exits right
    clipped_ops = ops.clip(clip_rect)
    assert len(clipped_ops.commands) == 2
    assert clipped_ops.commands[1].end == (100.0, 50.0)

def test_clip_line_crossing_two_boundaries(clip_rect):
    ops = Ops()
    ops.move_to(-50, 50) # Starts left
    ops.line_to(150, 50) # Exits right
    clipped_ops = ops.clip(clip_rect)
    assert len(clipped_ops.commands) == 2
    assert clipped_ops.commands[0].end == (0.0, 50.0)
    assert clipped_ops.commands[1].end == (100.0, 50.0)

def test_clip_arc_partially_inside(clip_rect):
    ops = Ops()
    ops.move_to(100, 50) # Start on right edge
    ops.arc_to(50, 100, i=-50, j=0) # Arc to top edge
    clipped_ops = ops.clip(clip_rect)
    # Result will be a series of line segments
    assert len(clipped_ops.commands) > 2
    assert all(c.end[0] <= 100.1 and c.end[1] <= 100.1 for c in clipped_ops.commands)

def test_clip_preserves_state_commands(clip_rect):
    ops = Ops()
    ops.set_power(500)
    ops.move_to(-10, 50)
    ops.line_to(110, 50)
    ops.set_cut_speed(1000)
    clipped_ops = ops.clip(clip_rect)
    state_cmds = [c for c in clipped_ops.commands if c.is_state_command()]
    assert len(state_cmds) == 2
    assert isinstance(state_cmds[0], SetPowerCommand)
    assert isinstance(state_cmds[1], SetCutSpeedCommand)

def test_clip_arc_reentering_visible_area(clip_rect):
    ops = Ops()
    # This arc starts at (10, 90), bulges up past y=100,
    # and re-enters to end at (90, 90).
    # Center is at (50, 70), radius is sqrt(2000) ~= 44.7
    # Max y is at 70 + 44.7 = 114.7, so it goes outside the clip rect (y_max=100)
    ops.move_to(10, 90)
    ops.arc_to(90, 90, i=40, j=-20, clockwise=True)
    clipped_ops = ops.clip(clip_rect)

    # Expected: two separate move_to/line_to sequences
    move_tos = [c for c in clipped_ops.commands if isinstance(c, MoveToCommand)]
    assert len(move_tos) == 2

    # All points must be within the bounds
    for cmd in clipped_ops.commands:
        if not cmd.is_state_command():
            assert 0 <= cmd.end[0] <= 100
            # Use a small tolerance for floating point inaccuracies at the boundary
            assert 0 <= cmd.end[1] <= 100.000001
