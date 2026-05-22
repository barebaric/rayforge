from typing import List

from raygeo.ops import Ops
from raygeo.ops.state import State
from raygeo.ops.types import CommandType


def test_group_by_command_type_empty():
    ops = Ops()
    assert list(ops.segment_indices()) == []


def test_group_by_command_type_single_move():
    ops = Ops()
    ops.move_to(0, 0)
    indices = list(ops.segment_indices())
    assert len(indices) == 1
    assert len(indices[0]) == 1
    assert ops.command_type(indices[0][0]) == CommandType.MOVE_TO


def test_group_by_command_type_move_and_line():
    ops = Ops()
    ops.move_to(0, 0)
    ops.line_to(1, 0)
    indices = list(ops.segment_indices())
    assert len(indices) == 1
    assert len(indices[0]) == 2


def test_group_by_command_type_move_and_arc():
    ops = Ops()
    ops.move_to(0, 0)
    ops.arc_to(1, 0, 1, 1, False)
    ops.line_to(2, 0)
    ops.line_to(3, 0)
    indices = list(ops.segment_indices())
    assert len(indices) == 1
    assert indices[0][0] == 0
    assert ops.command_type(indices[0][1]) == CommandType.ARC_TO


def test_group_by_command_type_state_commands():
    ops = Ops()
    ops.set_power(1.0)
    ops.move_to(0, 0)
    ops.line_to(1, 0)
    ops.disable_air_assist()
    indices = list(ops.segment_indices())
    assert len(indices) == 3
    assert ops.is_state(indices[0][0])
    assert ops.is_travel(indices[1][0])
    assert ops.is_state(indices[2][0])


def _create_ops_with_states(states_config: List[bool]) -> Ops:
    """Helper to create ops with specified air_assist states."""
    ops = Ops()
    for i, air_on in enumerate(states_config):
        ops.line_to(float(i), float(i))
    for i, air_on in enumerate(states_config):
        ops.set_state_at(
            i,
            State(
                power=1.0,
                air_assist=air_on,
                cut_speed=None,
                travel_speed=None,
                active_laser_uid=None,
                frequency=None,
                pulse_width=None,
            ),
        )
    return ops


def test_group_by_state_continuity():
    """Test splitting commands by non-reorderable state changes."""
    # All same state -> 1 segment
    ops1 = _create_ops_with_states([True, True, True])
    groups = ops1.group_by_state_continuity()
    assert len(groups) == 1
    assert groups[0].len() == 3

    # State change -> 2 segments
    ops2 = _create_ops_with_states([True, True, False])
    groups = ops2.group_by_state_continuity()
    assert len(groups) == 2
    assert groups[0].len() == 2
    assert groups[1].len() == 1

    # Multiple state changes
    ops3 = _create_ops_with_states([False, True, True, False, False, True])
    groups = ops3.group_by_state_continuity()
    assert len(groups) == 4
    assert [g.len() for g in groups] == [1, 2, 2, 1]

    # Empty
    ops_empty = Ops()
    assert ops_empty.group_by_state_continuity() == []

    # Single command
    ops4 = _create_ops_with_states([True])
    assert len(ops4.group_by_state_continuity()) == 1

    # Test with marker commands
    ops_marker = Ops()
    ops_marker.line_to(0, 0)
    ops_marker.job_start()
    ops_marker.line_to(1, 1)
    ops_marker.set_state_at(
        0,
        State(
            power=1.0,
            air_assist=True,
            cut_speed=None,
            travel_speed=None,
            active_laser_uid=None,
            frequency=None,
            pulse_width=None,
        ),
    )
    ops_marker.set_state_at(
        2,
        State(
            power=1.0,
            air_assist=True,
            cut_speed=None,
            travel_speed=None,
            active_laser_uid=None,
            frequency=None,
            pulse_width=None,
        ),
    )
    groups_m = ops_marker.group_by_state_continuity()
    assert len(groups_m) == 3
    assert [g.len() for g in groups_m] == [1, 1, 1]
    assert groups_m[1].is_marker(0)


def test_group_by_path_continuity():
    """Test splitting a list of commands into re-orderable paths."""
    ops = Ops()
    ops.move_to(0, 0)
    ops.line_to(10, 0)
    ops.line_to(10, 10)
    ops.move_to(100, 100)
    ops.line_to(110, 100)
    indices = list(ops.segment_indices())
    assert len(indices) == 2
    assert len(indices[0]) == 3
    assert ops.is_travel(indices[0][0])
    assert len(indices[1]) == 2
    assert ops.is_travel(indices[1][0])

    # Test with a travel command at the end
    ops.move_to(0, 0)
    indices = list(ops.segment_indices())
    assert len(indices) == 3
    assert len(indices[2]) == 1
