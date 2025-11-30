from typing import List
from rayforge.core.ops import (
    Command,
    MoveToCommand,
    LineToCommand,
    ArcToCommand,
    SetPowerCommand,
    DisableAirAssistCommand,
    State,
    JobStartCommand,
)
from rayforge.core.ops.group import (
    group_by_command_type,
    group_by_state_continuity,
    group_by_path_continuity,
)


def _to_dict(item):
    if hasattr(item, "__iter__") and not isinstance(item, tuple):
        return [_to_dict(c) for c in item]
    if hasattr(item, "__dict__"):
        d = item.__dict__.copy()
        # Remove state as it's not relevant for this comparison and can be None
        d.pop("state", None)
        d.pop("_state_ref_for_pyreverse", None)
        return d
    return item


def _pathcompare(one, two):
    return _to_dict(one) == _to_dict(two)


def test_group_by_command_type_empty():
    assert group_by_command_type([]) == []


def test_group_by_command_type_single_move():
    commands: List[Command] = [MoveToCommand((0, 0, 0))]
    assert _pathcompare(
        group_by_command_type(commands), [[MoveToCommand((0, 0, 0))]]
    )


def test_group_by_command_type_move_and_line():
    commands: List[Command] = [
        MoveToCommand((0, 0, 0)),
        LineToCommand((1, 0, 0)),
    ]
    assert _pathcompare(
        group_by_command_type(commands),
        [[MoveToCommand((0, 0, 0)), LineToCommand((1, 0, 0))]],
    )


def test_group_by_command_type_move_and_arc():
    commands: List[Command] = [
        MoveToCommand((0, 0, 0)),
        ArcToCommand((1, 0, 0), (1, 1), False),
        LineToCommand((2, 0, 0)),
        LineToCommand((3, 0, 0)),
    ]
    expected = [
        [MoveToCommand((0, 0, 0)), ArcToCommand((1, 0, 0), (1, 1), False)],
        [
            MoveToCommand((1, 0, 0)),
            LineToCommand((2, 0, 0)),
            LineToCommand((3, 0, 0)),
        ],
    ]
    assert _pathcompare(group_by_command_type(commands), expected)


def test_group_by_command_type_state_commands():
    commands: List[Command] = [
        SetPowerCommand(1.0),
        MoveToCommand((0, 0, 0)),
        LineToCommand((1, 0, 0)),
        DisableAirAssistCommand(),
    ]
    assert _pathcompare(
        group_by_command_type(commands),
        [
            [SetPowerCommand(1.0)],
            [MoveToCommand((0, 0, 0)), LineToCommand((1, 0, 0))],
            [DisableAirAssistCommand()],
        ],
    )


def _create_commands_with_states(states_config: List[bool]) -> List[Command]:
    """Helper to create commands with specified air_assist states."""
    commands = []
    for i, air_on in enumerate(states_config):
        state = State(power=1.0, air_assist=air_on)
        cmd = LineToCommand((float(i), float(i), 0.0))
        cmd.state = state
        commands.append(cmd)
    return commands


def test_group_by_state_continuity():
    """Test splitting commands by non-reorderable state changes."""
    # All same state -> 1 segment
    cmds1 = _create_commands_with_states([True, True, True])
    assert len(group_by_state_continuity(cmds1)) == 1
    assert len(group_by_state_continuity(cmds1)[0]) == 3

    # State change -> 2 segments
    cmds2 = _create_commands_with_states([True, True, False])
    assert len(group_by_state_continuity(cmds2)) == 2
    assert len(group_by_state_continuity(cmds2)[0]) == 2
    assert len(group_by_state_continuity(cmds2)[1]) == 1

    # Multiple state changes
    cmds3 = _create_commands_with_states(
        [False, True, True, False, False, True]
    )
    segments = group_by_state_continuity(cmds3)
    assert len(segments) == 4
    assert [len(s) for s in segments] == [1, 2, 2, 1]

    # Empty and single command lists
    assert group_by_state_continuity([]) == []
    cmds4 = _create_commands_with_states([True])
    assert len(group_by_state_continuity(cmds4)) == 1

    # Test with marker commands
    cmds_with_marker = _create_commands_with_states([True, True])
    cmds_with_marker.insert(1, JobStartCommand())
    segments_with_marker = group_by_state_continuity(cmds_with_marker)
    assert len(segments_with_marker) == 3
    assert [len(s) for s in segments_with_marker] == [1, 1, 1]
    assert isinstance(segments_with_marker[1][0], JobStartCommand)


def test_group_by_path_continuity():
    """Test splitting a list of commands into re-orderable paths."""
    cmds: List[Command] = [
        MoveToCommand((0, 0, 0)),
        LineToCommand((10, 0, 0)),
        LineToCommand((10, 10, 0)),
        MoveToCommand((100, 100, 0)),
        LineToCommand((110, 100, 0)),
    ]
    segments = group_by_path_continuity(cmds)
    assert len(segments) == 2
    assert len(segments[0]) == 3
    assert isinstance(segments[0][0], MoveToCommand)
    assert len(segments[1]) == 2
    assert isinstance(segments[1][0], MoveToCommand)

    # Test with a travel command at the end
    cmds.append(MoveToCommand((0, 0, 0)))
    segments = group_by_path_continuity(cmds)
    assert len(segments) == 3
    assert len(segments[2]) == 1
