from rayforge.core.ops.commands import (
    ArcToCommand,
    MoveToCommand,
    LineToCommand,
    State,
    OpsSectionStartCommand,
    OpsSectionEndCommand,
    SectionType,
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
