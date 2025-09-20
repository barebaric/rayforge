import pytest
from rayforge.core.ops.commands import (
    ArcToCommand,
    MoveToCommand,
    LineToCommand,
    State,
    OpsSectionStartCommand,
    OpsSectionEndCommand,
    SectionType,
    ScanLinePowerCommand,
    SetPowerCommand,
)


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


def test_section_markers_are_marker_commands():
    start_cmd = OpsSectionStartCommand(SectionType.VECTOR_OUTLINE, "uid123")
    end_cmd = OpsSectionEndCommand(SectionType.VECTOR_OUTLINE)
    assert start_cmd.is_marker_command()
    assert end_cmd.is_marker_command()
    # Also check they aren't other types
    assert not start_cmd.is_state_command()
    assert not start_cmd.is_cutting_command()
    assert not start_cmd.is_travel_command()


def test_command_serialization():
    start_cmd = OpsSectionStartCommand(SectionType.RASTER_FILL, "wp-abc")
    data = start_cmd.to_dict()
    assert data["type"] == "OpsSectionStartCommand"
    assert data["section_type"] == "RASTER_FILL"
    assert data["workpiece_uid"] == "wp-abc"

    end_cmd = OpsSectionEndCommand(SectionType.RASTER_FILL)
    data = end_cmd.to_dict()
    assert data["type"] == "OpsSectionEndCommand"
    assert data["section_type"] == "RASTER_FILL"


def test_line_to_command_linearize():
    """Tests that a LineToCommand linearizes to itself."""
    cmd = LineToCommand((10, 20, 30))
    linearized = cmd.linearize((0, 0, 0))
    assert len(linearized) == 1
    assert linearized[0] is cmd


def test_arc_to_command_linearize():
    """Tests that an ArcToCommand linearizes to a series of LineToCommands."""
    start_point = (10, 0, 5)
    arc_cmd = ArcToCommand(
        end=(0, 10, 5), center_offset=(-10, 0), clockwise=False
    )
    linearized = arc_cmd.linearize(start_point)

    assert len(linearized) > 1
    assert all(isinstance(c, LineToCommand) for c in linearized)

    # Check that the chain of linearized segments matches the original arc
    final_point = linearized[-1].end
    assert final_point == pytest.approx(arc_cmd.end)


def test_arc_to_command_reverse_geometry():
    """
    Tests the public geometry reversal logic for an ArcToCommand.
    """
    original_start = (10, 0, 0)
    original_end = (0, 10, 0)
    # Center is at (0,0), so offset from start (10,0) is (-10, 0)
    arc = ArcToCommand(original_end, center_offset=(-10, 0), clockwise=False)

    # When flipping, the command's endpoint is updated first. The new start
    # is the original end, and the new end is the original start.
    arc.end = original_start
    arc.reverse_geometry(
        original_start=original_start, original_end=original_end
    )

    # The direction should be inverted
    assert arc.clockwise is True

    # The center should still be (0,0). The new start point is (0,10).
    # The new offset should be from (0,10) to (0,0), which is (0, -10).
    assert arc.center_offset[0] == pytest.approx(0)
    assert arc.center_offset[1] == pytest.approx(-10)


def test_scan_line_power_command_linearize():
    """Tests that a ScanLinePowerCommand linearizes correctly."""
    cmd = ScanLinePowerCommand(
        start_point=(0, 0, 5),
        end=(3, 0, 5),
        power_values=bytearray([100, 200, 200]),
    )
    linearized = cmd.linearize((0, 0, 0))

    # Expect 5 commands: Set(100), Line(1,0), Set(200), Line(2,0), Line(3,0)
    assert len(linearized) == 5
    # First power set
    assert isinstance(linearized[0], SetPowerCommand)
    assert linearized[0].power == 100
    # First line segment
    assert isinstance(linearized[1], LineToCommand)
    assert linearized[1].end == pytest.approx((1.0, 0.0, 5.0))
    # Second power set
    assert isinstance(linearized[2], SetPowerCommand)
    assert linearized[2].power == 200
    # Second line segment
    assert isinstance(linearized[3], LineToCommand)
    assert linearized[3].end == pytest.approx((2.0, 0.0, 5.0))
    # Third line segment (power is unchanged)
    assert isinstance(linearized[4], LineToCommand)
    assert linearized[4].end == pytest.approx((3.0, 0.0, 5.0))


def test_scan_line_power_command_reverse_geometry():
    """Tests the reversal of a ScanLinePowerCommand."""
    start = (0, 0, 0)
    end = (10, 10, 10)
    powers = bytearray([10, 20, 30])
    cmd = ScanLinePowerCommand(start, end, powers)

    cmd.reverse_geometry()

    assert cmd.start_point == end
    assert cmd.end == start
    assert cmd.power_values == bytearray([30, 20, 10])


def test_command_distance_calculation():
    """Tests the distance() method on various command types."""
    last_point = (0.0, 0.0, 0.0)

    # LineTo should calculate distance from the last point
    line_cmd = LineToCommand((3.0, 4.0, 0.0))
    assert line_cmd.distance(last_point) == pytest.approx(5.0)

    # ScanLinePowerCommand should use its own start_point, ignoring last_point
    scan_cmd = ScanLinePowerCommand(
        start_point=(10.0, 0.0, 0.0),
        end=(13.0, 4.0, 0.0),
        power_values=bytearray(),
    )
    assert scan_cmd.distance(last_point) == pytest.approx(5.0)

    # State commands should have zero distance
    state_cmd = SetPowerCommand(100)
    assert state_cmd.distance(last_point) == 0.0
