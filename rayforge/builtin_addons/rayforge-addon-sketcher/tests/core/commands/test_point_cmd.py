import pytest
from sketcher.core import Sketch
from sketcher.core.commands import (
    MovePointCommand,
    UnstickJunctionCommand,
)


@pytest.fixture
def sketch():
    """Create a basic sketch for testing."""
    return Sketch()


def test_move_point_command_initialization(sketch):
    """Test that MovePointCommand initializes correctly."""
    p_id = sketch.add_point(10.0, 20.0)
    cmd = MovePointCommand(sketch, p_id, (10.0, 20.0), (30.0, 40.0))

    assert cmd.sketch is sketch
    assert cmd.point_id == p_id
    assert cmd.start_pos == (10.0, 20.0)
    assert cmd.end_pos == (30.0, 40.0)


def test_move_point_command_initialization_with_snapshot(sketch):
    """Test that MovePointCommand can be initialized with a snapshot."""
    p_id = sketch.add_point(10.0, 20.0)
    snapshot = {"points": {p_id: (10.0, 20.0)}, "entities": {}, "arrays": []}
    cmd = MovePointCommand(sketch, p_id, (10.0, 20.0), (30.0, 40.0), snapshot)

    assert cmd._snapshot == snapshot


def test_move_point_command_execute(sketch):
    """Test that execute moves the point to end position."""
    p_id = sketch.add_point(10.0, 20.0)
    cmd = MovePointCommand(sketch, p_id, (10.0, 20.0), (30.0, 40.0))

    assert sketch.registry.get_point(p_id)[0] == 10.0
    assert sketch.registry.get_point(p_id)[1] == 20.0

    cmd.execute()

    assert sketch.registry.get_point(p_id)[0] == 30.0
    assert sketch.registry.get_point(p_id)[1] == 40.0


def test_move_point_command_undo(sketch):
    """Test that undo moves the point back to start position."""
    p_id = sketch.add_point(10.0, 20.0)
    cmd = MovePointCommand(sketch, p_id, (10.0, 20.0), (30.0, 40.0))

    cmd.execute()
    assert sketch.registry.get_point(p_id)[0] == 30.0
    assert sketch.registry.get_point(p_id)[1] == 40.0

    cmd.undo()

    assert sketch.registry.get_point(p_id)[0] == 10.0
    assert sketch.registry.get_point(p_id)[1] == 20.0


def test_move_point_command_execute_undo_cycle(sketch):
    """Test that execute and undo can be called multiple times."""
    p_id = sketch.add_point(10.0, 20.0)
    cmd = MovePointCommand(sketch, p_id, (10.0, 20.0), (30.0, 40.0))

    for _ in range(3):
        cmd.execute()
        assert sketch.registry.get_point(p_id)[0] == 30.0
        assert sketch.registry.get_point(p_id)[1] == 40.0

        cmd.undo()
        assert sketch.registry.get_point(p_id)[0] == 10.0
        assert sketch.registry.get_point(p_id)[1] == 20.0


def test_move_point_command_can_coalesce_with(sketch):
    """Test that can_coalesce_with works correctly."""
    p_id = sketch.add_point(10.0, 20.0)
    cmd1 = MovePointCommand(sketch, p_id, (10.0, 20.0), (30.0, 40.0))
    cmd2 = MovePointCommand(sketch, p_id, (30.0, 40.0), (50.0, 60.0))
    p_id2 = sketch.add_point(100.0, 200.0)
    cmd3 = MovePointCommand(sketch, p_id2, (100.0, 200.0), (150.0, 250.0))

    assert cmd1.can_coalesce_with(cmd2) is True
    assert cmd1.can_coalesce_with(cmd3) is False


def test_move_point_command_coalesce_with(sketch):
    """Test that coalesce_with merges commands correctly."""
    p_id = sketch.add_point(10.0, 20.0)
    cmd1 = MovePointCommand(sketch, p_id, (10.0, 20.0), (30.0, 40.0))
    cmd2 = MovePointCommand(sketch, p_id, (30.0, 40.0), (50.0, 60.0))

    result = cmd1.coalesce_with(cmd2)

    assert result is True
    assert cmd1.end_pos == (50.0, 60.0)
    assert cmd1.start_pos == (10.0, 20.0)


def test_move_point_command_coalesce_with_invalid(sketch):
    """Test that coalesce_with returns False for incompatible commands."""
    p_id = sketch.add_point(10.0, 20.0)
    cmd1 = MovePointCommand(sketch, p_id, (10.0, 20.0), (30.0, 40.0))
    p_id2 = sketch.add_point(100.0, 200.0)
    cmd2 = MovePointCommand(sketch, p_id2, (100.0, 200.0), (150.0, 250.0))

    result = cmd1.coalesce_with(cmd2)

    assert result is False


def test_move_point_command_position_lookup(sketch):
    """The command reads the point position from the registry."""
    p_id = sketch.add_point(10.0, 20.0)
    MovePointCommand(sketch, p_id, (10.0, 20.0), (30.0, 40.0))

    position = sketch.registry.get_point(p_id)

    assert position is not None
    assert position == (10.0, 20.0)


def test_move_point_command_get_point_missing(sketch):
    """A missing point yields no position."""
    MovePointCommand(sketch, 9999, (0.0, 0.0), (10.0, 10.0))

    position = sketch.registry.get_point(9999)

    assert position is None


def test_unstick_junction_command_initialization(sketch):
    """Test that UnstickJunctionCommand initializes correctly."""
    p_id = sketch.add_point(10.0, 20.0)
    cmd = UnstickJunctionCommand(sketch, p_id)

    assert cmd.sketch is sketch
    assert cmd.junction_pid == p_id
    assert cmd.new_pid is None
    assert cmd.modified_entity_ids == set()


def test_unstick_junction_command_execute_with_two_lines(sketch):
    """Test that execute creates a new point for two lines."""
    p1 = sketch.add_point(0.0, 0.0)
    p2 = sketch.add_point(10.0, 0.0)
    p3 = sketch.add_point(10.0, 10.0)

    line1_id = sketch.add_line(p1, p2)
    line2_id = sketch.add_line(p2, p3)

    cmd = UnstickJunctionCommand(sketch, p2)

    initial_points_count = len(sketch.registry.points())

    cmd.execute()

    assert len(sketch.registry.points()) == initial_points_count + 1
    assert cmd.new_pid is not None

    line1 = sketch.registry.get_entity_object(line1_id)
    line2 = sketch.registry.get_entity_object(line2_id)

    assert line1.p1_idx == p1
    assert line1.p2_idx == p2

    assert line2.p1_idx == cmd.new_pid
    assert line2.p2_idx == p3


def test_unstick_junction_command_execute_with_single_line(sketch):
    """Test that execute does nothing with a single line."""
    p1 = sketch.add_point(0.0, 0.0)
    p2 = sketch.add_point(10.0, 0.0)

    sketch.add_line(p1, p2)

    cmd = UnstickJunctionCommand(sketch, p2)

    initial_points_count = len(sketch.registry.points())

    cmd.execute()

    assert len(sketch.registry.points()) == initial_points_count
    assert cmd.new_pid is None


def test_unstick_junction_command_execute_with_no_entities(sketch):
    """Test that execute does nothing when no entities at junction."""
    p1 = sketch.add_point(10.0, 20.0)

    cmd = UnstickJunctionCommand(sketch, p1)

    initial_points_count = len(sketch.registry.points())

    cmd.execute()

    assert len(sketch.registry.points()) == initial_points_count
    assert cmd.new_pid is None


def test_unstick_junction_command_undo(sketch):
    """Test that undo restores the original junction."""
    p1 = sketch.add_point(0.0, 0.0)
    p2 = sketch.add_point(10.0, 0.0)
    p3 = sketch.add_point(10.0, 10.0)

    sketch.add_line(p1, p2)
    line2_id = sketch.add_line(p2, p3)

    cmd = UnstickJunctionCommand(sketch, p2)

    cmd.execute()

    assert cmd.new_pid is not None
    line2 = sketch.registry.get_entity_object(line2_id)
    assert line2.p1_idx == cmd.new_pid

    cmd.undo()

    line2 = sketch.registry.get_entity_object(line2_id)
    assert line2.p1_idx == p2


def test_unstick_junction_command_execute_undo_cycle(sketch):
    """Test that execute and undo can be called multiple times."""
    p1 = sketch.add_point(0.0, 0.0)
    p2 = sketch.add_point(10.0, 0.0)
    p3 = sketch.add_point(10.0, 10.0)

    sketch.add_line(p1, p2)
    line2_id = sketch.add_line(p2, p3)

    cmd = UnstickJunctionCommand(sketch, p2)

    for _ in range(3):
        cmd.execute()
        line2 = sketch.registry.get_entity_object(line2_id)
        assert line2.p1_idx != p2

        cmd.undo()
        line2 = sketch.registry.get_entity_object(line2_id)
        assert line2.p1_idx == p2


def test_unstick_junction_command_with_arc(sketch):
    """Test that execute works with arcs at junction."""
    p1 = sketch.add_point(0.0, 0.0)
    p2 = sketch.add_point(10.0, 0.0)
    p3 = sketch.add_point(5.0, 5.0)

    sketch.add_line(p1, p2)
    arc_id = sketch.add_arc(p2, p3, p1)

    cmd = UnstickJunctionCommand(sketch, p2)

    cmd.execute()

    assert cmd.new_pid is not None
    arc = sketch.registry.get_entity_object(arc_id)
    assert arc.start_idx == cmd.new_pid


def test_unstick_junction_command_with_circle(sketch):
    """Test that execute works with circles at junction."""
    p1 = sketch.add_point(0.0, 0.0)
    p2 = sketch.add_point(10.0, 0.0)
    p3 = sketch.add_point(10.0, 10.0)

    sketch.add_line(p1, p2)
    circle_id = sketch.add_circle(p2, p3)

    cmd = UnstickJunctionCommand(sketch, p2)

    cmd.execute()

    assert cmd.new_pid is not None
    circle = sketch.registry.get_entity_object(circle_id)
    assert circle.center_idx == cmd.new_pid
