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
    SetCutSpeedCommand,
    SetTravelSpeedCommand,
    EnableAirAssistCommand,
    DisableAirAssistCommand,
    JobStartCommand,
    JobEndCommand,
    LayerStartCommand,
    LayerEndCommand,
    WorkpieceStartCommand,
    WorkpieceEndCommand,
    SetLaserCommand,
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


def test_command_repr():
    """Ensures the __repr__ method works without error."""
    cmd = LineToCommand((1, 2, 3))
    rep = repr(cmd)
    assert "LineToCommand" in rep
    assert "'end': (1, 2, 3)" in rep


def test_set_power_command():
    cmd = SetPowerCommand(power=0.6)
    state = State(power=0)
    cmd.apply_to_state(state)
    assert state.power == 0.6
    assert cmd.is_state_command()
    data = cmd.to_dict()
    assert data["type"] == "SetPowerCommand"
    assert data["power"] == 0.6


def test_set_cut_speed_command():
    cmd = SetCutSpeedCommand(speed=1000)
    state = State(cut_speed=0)
    cmd.apply_to_state(state)
    assert state.cut_speed == 1000
    assert cmd.is_state_command()
    data = cmd.to_dict()
    assert data["type"] == "SetCutSpeedCommand"
    assert data["speed"] == 1000


def test_set_travel_speed_command():
    cmd = SetTravelSpeedCommand(speed=5000)
    state = State(travel_speed=0)
    cmd.apply_to_state(state)
    assert state.travel_speed == 5000
    assert cmd.is_state_command()
    data = cmd.to_dict()
    assert data["type"] == "SetTravelSpeedCommand"
    assert data["speed"] == 5000


def test_enable_air_assist_command():
    cmd = EnableAirAssistCommand()
    state = State(air_assist=False)
    cmd.apply_to_state(state)
    assert state.air_assist is True
    assert cmd.is_state_command()
    data = cmd.to_dict()
    assert data["type"] == "EnableAirAssistCommand"


def test_disable_air_assist_command():
    cmd = DisableAirAssistCommand()
    state = State(air_assist=True)
    cmd.apply_to_state(state)
    assert state.air_assist is False
    assert cmd.is_state_command()
    data = cmd.to_dict()
    assert data["type"] == "DisableAirAssistCommand"


def test_set_laser_command():
    cmd = SetLaserCommand(laser_uid="laser-123")
    state = State(active_laser_uid=None)
    cmd.apply_to_state(state)
    assert state.active_laser_uid == "laser-123"
    assert cmd.is_state_command()
    data = cmd.to_dict()
    assert data["type"] == "SetLaserCommand"
    assert data["laser_uid"] == "laser-123"


# --- Marker Command Tests ---


@pytest.mark.parametrize(
    "cmd_class, args",
    [
        (JobStartCommand, ()),
        (JobEndCommand, ()),
        (LayerStartCommand, ("layer-1",)),
        (LayerEndCommand, ("layer-1",)),
        (WorkpieceStartCommand, ("wp-1",)),
        (WorkpieceEndCommand, ("wp-1",)),
        (OpsSectionStartCommand, (SectionType.VECTOR_OUTLINE, "wp-1")),
        (OpsSectionEndCommand, (SectionType.VECTOR_OUTLINE,)),
    ],
)
def test_all_marker_commands(cmd_class, args):
    """Tests that all marker commands identify as such and not other types."""
    cmd = cmd_class(*args)
    assert cmd.is_marker()
    assert not cmd.is_cutting_command()
    assert not cmd.is_travel_command()
    assert not cmd.is_state_command()


def test_job_marker_serialization():
    start = JobStartCommand().to_dict()
    assert start["type"] == "JobStartCommand"
    end = JobEndCommand().to_dict()
    assert end["type"] == "JobEndCommand"


def test_layer_marker_serialization():
    start = LayerStartCommand("layer-abc").to_dict()
    assert start["type"] == "LayerStartCommand"
    assert start["layer_uid"] == "layer-abc"
    end = LayerEndCommand("layer-abc").to_dict()
    assert end["type"] == "LayerEndCommand"
    assert end["layer_uid"] == "layer-abc"


def test_workpiece_marker_serialization():
    start = WorkpieceStartCommand("wp-123").to_dict()
    assert start["type"] == "WorkpieceStartCommand"
    assert start["workpiece_uid"] == "wp-123"
    end = WorkpieceEndCommand("wp-123").to_dict()
    assert end["type"] == "WorkpieceEndCommand"
    assert end["workpiece_uid"] == "wp-123"


def test_ops_section_serialization():
    start_cmd = OpsSectionStartCommand(SectionType.RASTER_FILL, "wp-abc")
    data = start_cmd.to_dict()
    assert data["type"] == "OpsSectionStartCommand"
    assert data["section_type"] == "RASTER_FILL"
    assert data["workpiece_uid"] == "wp-abc"

    end_cmd = OpsSectionEndCommand(SectionType.RASTER_FILL)
    data = end_cmd.to_dict()
    assert data["type"] == "OpsSectionEndCommand"
    assert data["section_type"] == "RASTER_FILL"


def test_move_to_command():
    """Tests MoveToCommand-specific methods."""
    cmd = MoveToCommand((10, 20, 30))
    # Test linearize
    linearized = cmd.linearize((0, 0, 0))
    assert len(linearized) == 1
    assert linearized[0] is cmd
    # Test serialization
    data = cmd.to_dict()
    assert data["type"] == "MoveToCommand"
    assert data["end"] == (10, 20, 30)


def test_line_to_command_linearize_and_dict():
    """Tests that a LineToCommand linearizes to itself and serializes."""
    cmd = LineToCommand((10, 20, 30))
    # Test linearize
    linearized = cmd.linearize((0, 0, 0))
    assert len(linearized) == 1
    assert linearized[0] is cmd
    # Test serialization
    data = cmd.to_dict()
    assert data["type"] == "LineToCommand"
    assert data["end"] == (10, 20, 30)


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


def test_arc_to_command_serialization():
    cmd = ArcToCommand(end=(0, 10, 5), center_offset=(-10, 0), clockwise=False)
    data = cmd.to_dict()
    assert data["type"] == "ArcToCommand"
    assert data["end"] == (0, 10, 5)
    assert data["center_offset"] == (-10, 0)
    assert data["clockwise"] is False


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
    start_point = (0, 0, 5)
    cmd = ScanLinePowerCommand(
        end=(3, 0, 5),
        power_values=bytearray([100, 200, 200]),
    )
    linearized = cmd.linearize(start_point)

    # Power changes from 100 to 200 after the first pixel.
    # The next two pixels are 200, so they are one segment.
    # Expected: Set(100), LineTo(pixel 0 end), Set(200), LineTo(final end)
    assert len(linearized) == 4

    # Segment 1 (power 100/255 normalized)
    assert isinstance(linearized[0], SetPowerCommand)
    assert linearized[0].power == pytest.approx(100.0 / 255.0)
    assert isinstance(linearized[1], LineToCommand)
    assert linearized[1].end == pytest.approx((1.0, 0.0, 5.0))

    # Segment 2 (power 200/255 normalized)
    assert isinstance(linearized[2], SetPowerCommand)
    assert linearized[2].power == pytest.approx(200.0 / 255.0)
    # This line covers the last two pixels and goes to the final end point
    assert isinstance(linearized[3], LineToCommand)
    assert linearized[3].end == pytest.approx((3.0, 0.0, 5.0))


def test_scan_line_power_command_linearize_constant_power():
    """Tests linearizing a scanline with constant power."""
    start_point = (0, 10, 0)
    cmd = ScanLinePowerCommand(
        end=(5, 10, 0), power_values=bytearray([150, 150, 150, 150, 150])
    )
    linearized = cmd.linearize(start_point)

    # Should be one SetPower and one LineTo the final destination
    assert len(linearized) == 2
    assert isinstance(linearized[0], SetPowerCommand)
    assert linearized[0].power == pytest.approx(150.0 / 255.0)
    assert isinstance(linearized[1], LineToCommand)
    assert linearized[1].end == (5, 10, 0)


def test_scan_line_power_command_linearize_empty():
    """
    Tests that linearizing a ScanLinePowerCommand with no power values
    yields no commands.
    """
    cmd = ScanLinePowerCommand(
        end=(3, 0, 5),
        power_values=bytearray([]),
    )
    linearized = cmd.linearize((0, 0, 5))
    assert linearized == []


def test_scan_line_power_command_properties_and_dict():
    """Tests the properties and serialization of ScanLinePowerCommand."""
    cmd = ScanLinePowerCommand(
        end=(3, 0, 5),
        power_values=bytearray([100, 200]),
    )
    assert cmd.is_cutting_command()
    data = cmd.to_dict()
    assert data["type"] == "ScanLinePowerCommand"
    assert "start_point" not in data
    assert data["end"] == (3, 0, 5)
    assert data["power_values"] == [100, 200]


def test_command_distance_calculation():
    """Tests the distance() method on various command types."""
    last_point = (0.0, 0.0, 0.0)

    # LineTo should calculate distance from the last point
    line_cmd = LineToCommand((3.0, 4.0, 0.0))
    assert line_cmd.distance(last_point) == pytest.approx(5.0)

    # ArcTo should calculate chord distance from the last point
    arc_cmd = ArcToCommand(
        end=(13.0, 4.0, 0.0), center_offset=(0, 0), clockwise=True
    )
    assert arc_cmd.distance((10.0, 0.0, 0.0)) == pytest.approx(5.0)

    # ScanLinePowerCommand should now behave like LineTo
    scan_cmd = ScanLinePowerCommand(
        end=(13.0, 4.0, 0.0),
        power_values=bytearray(),
    )
    assert scan_cmd.distance((10.0, 0.0, 0.0)) == pytest.approx(5.0)

    # State commands should have zero distance
    state_cmd = SetPowerCommand(1.0)
    assert state_cmd.distance(last_point) == 0.0


def test_moving_command_distance_with_no_last_point():
    """Tests that distance() returns 0 if no start point is available."""
    cmd = LineToCommand((3, 4, 0))
    assert cmd.distance(None) == 0.0


def test_scan_line_power_command_split_by_power_simple():
    """Tests splitting a scanline with one central active segment."""
    start_point = (0.0, 10.0, 0.0)
    end_point = (10.0, 10.0, 0.0)
    powers = bytearray([0, 0, 100, 100, 100, 100, 0, 0, 0, 0])
    cmd = ScanLinePowerCommand(end=end_point, power_values=powers)

    min_power = 50

    result = cmd.split_by_power(start_point, min_power)

    assert len(result) == 2
    move_cmd, scan_cmd = result
    assert isinstance(move_cmd, MoveToCommand)
    assert isinstance(scan_cmd, ScanLinePowerCommand)

    # Active segment: t=0.2 to t=0.6 -> x=2.0 to x=6.0
    assert move_cmd.end == pytest.approx((2.0, 10.0, 0.0))
    assert scan_cmd.end == pytest.approx((6.0, 10.0, 0.0))
    assert scan_cmd.power_values == bytearray([100, 100, 100, 100])


def test_scan_line_power_command_split_by_power_multiple():
    """Tests splitting a scanline with multiple active segments."""
    start_point = (0.0, 20.0, 0.0)
    end_point = (10.0, 20.0, 0.0)
    powers = bytearray([100, 100, 0, 0, 200, 200, 200, 0, 150, 150])
    cmd = ScanLinePowerCommand(end=end_point, power_values=powers)

    result = cmd.split_by_power(start_point, min_power=50)

    assert len(result) == 6
    # --- Seg 1 (idx 0-1) t=0.0 to t=0.2 -> x=0 to x=2
    move1, scan1 = result[0], result[1]
    assert isinstance(scan1, ScanLinePowerCommand)
    assert move1.end == pytest.approx((0.0, 20.0, 0.0))
    assert scan1.end == pytest.approx((2.0, 20.0, 0.0))
    assert scan1.power_values == bytearray([100, 100])

    # --- Seg 2 (idx 4-6) t=0.4 to t=0.7 -> x=4 to x=7
    move2, scan2 = result[2], result[3]
    assert isinstance(scan2, ScanLinePowerCommand)
    assert move2.end == pytest.approx((4.0, 20.0, 0.0))
    assert scan2.end == pytest.approx((7.0, 20.0, 0.0))
    assert scan2.power_values == bytearray([200, 200, 200])

    # --- Seg 3 (idx 8-9) t=0.8 to t=1.0 -> x=8 to x=10
    move3, scan3 = result[4], result[5]
    assert isinstance(scan3, ScanLinePowerCommand)
    assert move3.end == pytest.approx((8.0, 20.0, 0.0))
    assert scan3.end == pytest.approx((10.0, 20.0, 0.0))
    assert scan3.power_values == bytearray([150, 150])


def test_scan_line_power_command_split_by_power_fully_on():
    """
    Tests splitting a scanline that is entirely above the power threshold.
    """
    start_point = (10, 10, 0)
    end_point = (20, 10, 0)
    powers = bytearray([100, 150, 200])
    cmd = ScanLinePowerCommand(end=end_point, power_values=powers)

    result = cmd.split_by_power(start_point, min_power=50)

    assert len(result) == 2
    move_cmd, scan_cmd = result
    assert isinstance(scan_cmd, ScanLinePowerCommand)
    assert move_cmd.end == pytest.approx((10.0, 10.0, 0.0))
    assert scan_cmd.end == pytest.approx((20.0, 10.0, 0.0))
    assert scan_cmd.power_values == powers


def test_scan_line_power_command_split_by_power_fully_off():
    """Tests splitting a scanline that is entirely below the threshold."""
    start_point = (10, 10, 0)
    end_point = (20, 10, 0)
    powers = bytearray([10, 20, 30])
    cmd = ScanLinePowerCommand(end=end_point, power_values=powers)

    result = cmd.split_by_power(start_point, min_power=50)
    assert result == []


def test_scan_line_power_command_split_by_power_empty():
    """Tests splitting a scanline with no power data."""
    start_point = (10, 10, 0)
    end_point = (20, 10, 0)
    powers = bytearray([])
    cmd = ScanLinePowerCommand(end=end_point, power_values=powers)

    result = cmd.split_by_power(start_point, min_power=50)
    assert result == []
