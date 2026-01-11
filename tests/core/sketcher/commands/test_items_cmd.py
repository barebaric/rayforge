import pytest

from rayforge.core.sketcher import Sketch
from rayforge.core.sketcher.commands import AddItemsCommand, RemoveItemsCommand
from rayforge.core.sketcher.entities import Point, Line
from rayforge.core.sketcher.constraints import DistanceConstraint


@pytest.fixture
def sketch():
    """Create a basic sketch for testing."""
    return Sketch()


def test_add_items_command_initialization(sketch):
    """Test that AddItemsCommand initializes correctly."""
    p = Point(-1, 10.0, 20.0)
    cmd = AddItemsCommand(sketch, "Add Items", points=[p])

    assert cmd.sketch is sketch
    assert cmd.name == "Add Items"
    assert len(cmd.points) == 1
    assert len(cmd.entities) == 0
    assert len(cmd.constraints) == 0


def test_add_items_command_add_points(sketch):
    """Test that execute adds points to the sketch."""
    p1 = Point(-1, 10.0, 20.0)
    p2 = Point(-2, 30.0, 40.0)
    cmd = AddItemsCommand(sketch, "Add Items", points=[p1, p2])

    initial_count = len(sketch.registry.points)

    cmd.execute()

    assert len(sketch.registry.points) == initial_count + 2
    assert p1.id >= 0
    assert p2.id >= 0


def test_add_items_command_add_entities(sketch):
    """Test that execute adds entities to the sketch."""
    p1 = Point(-1, 10.0, 20.0)
    p2 = Point(-2, 30.0, 40.0)
    line = Line(-1, p1.id, p2.id)
    cmd = AddItemsCommand(
        sketch, "Add Items", points=[p1, p2], entities=[line]
    )

    cmd.execute()

    assert len(sketch.registry.entities) == 1
    assert line.id >= 0
    assert line.p1_idx >= 0
    assert line.p2_idx >= 0
    assert p1.id >= 0
    assert p2.id >= 0


def test_add_items_command_add_constraints(sketch):
    """Test that execute adds constraints to the sketch."""
    p1 = sketch.add_point(0.0, 0.0)
    p2 = sketch.add_point(10.0, 0.0)
    constraint = DistanceConstraint(p1, p2, 10.0)
    cmd = AddItemsCommand(sketch, "Add Items", constraints=[constraint])

    initial_count = len(sketch.constraints)

    cmd.execute()

    assert len(sketch.constraints) == initial_count + 1
    assert constraint in sketch.constraints


def test_add_items_command_undo(sketch):
    """Test that undo removes added items from the sketch."""
    p1 = Point(-1, 10.0, 20.0)
    p2 = Point(-2, 30.0, 40.0)
    line = Line(-1, p1.id, p2.id)
    cmd = AddItemsCommand(
        sketch, "Add Items", points=[p1, p2], entities=[line]
    )

    cmd.execute()
    assert len(sketch.registry.points) > 0
    assert len(sketch.registry.entities) > 0

    cmd.undo()

    assert len(sketch.registry.points) == 1
    assert len(sketch.registry.entities) == 0


def test_add_items_command_execute_undo_cycle(sketch):
    """Test that execute and undo can be called multiple times."""
    p1 = Point(-1, 10.0, 20.0)
    cmd = AddItemsCommand(sketch, "Add Items", points=[p1])

    for _ in range(3):
        cmd.execute()
        assert len(sketch.registry.points) == 2

        cmd.undo()
        assert len(sketch.registry.points) == 1


def test_remove_items_command_initialization(sketch):
    """Test that RemoveItemsCommand initializes correctly."""
    p1 = sketch.add_point(10.0, 20.0)
    cmd = RemoveItemsCommand(sketch, "Remove Items", points=[p1])

    assert cmd.sketch is sketch
    assert cmd.name == "Remove Items"
    assert len(cmd.points) == 1
    assert len(cmd.entities) == 0
    assert len(cmd.constraints) == 0


def test_remove_items_command_remove_points(sketch):
    """Test that execute removes points from the sketch."""
    p1 = sketch.add_point(10.0, 20.0)
    p2 = sketch.add_point(30.0, 40.0)
    point1 = sketch.registry.get_point(p1)
    point2 = sketch.registry.get_point(p2)
    cmd = RemoveItemsCommand(sketch, "Remove Items", points=[point1, point2])

    assert len(sketch.registry.points) == 3

    cmd.execute()

    assert len(sketch.registry.points) == 1


def test_remove_items_command_remove_entities(sketch):
    """Test that execute removes entities from the sketch."""
    p1 = sketch.add_point(10.0, 20.0)
    p2 = sketch.add_point(30.0, 40.0)
    line = sketch.add_line(p1, p2)
    line_ent = sketch.registry.get_entity(line)

    cmd = RemoveItemsCommand(sketch, "Remove Items", entities=[line_ent])

    assert len(sketch.registry.entities) == 1

    cmd.execute()

    assert len(sketch.registry.entities) == 0


def test_remove_items_command_remove_constraints(sketch):
    """Test that execute removes constraints from the sketch."""
    p1 = sketch.add_point(0.0, 0.0)
    p2 = sketch.add_point(10.0, 0.0)
    constraint = DistanceConstraint(p1, p2, 10.0)
    sketch.constraints.append(constraint)

    assert len(sketch.constraints) == 1

    cmd = RemoveItemsCommand(sketch, "Remove Items", constraints=[constraint])
    cmd.execute()

    assert len(sketch.constraints) == 0


def test_remove_items_command_undo(sketch):
    """Test that undo restores removed items to the sketch."""
    p1 = sketch.add_point(10.0, 20.0)
    p2 = sketch.add_point(30.0, 40.0)
    line = sketch.add_line(p1, p2)
    line_ent = sketch.registry.get_entity(line)
    point1 = sketch.registry.get_point(p1)
    point2 = sketch.registry.get_point(p2)

    cmd = RemoveItemsCommand(
        sketch, "Remove Items", points=[point1, point2], entities=[line_ent]
    )

    cmd.execute()
    assert len(sketch.registry.points) == 1
    assert len(sketch.registry.entities) == 0

    cmd.undo()

    assert len(sketch.registry.points) == 3
    assert len(sketch.registry.entities) == 1


def test_remove_items_command_execute_undo_cycle(sketch):
    """Test that execute and undo can be called multiple times."""
    p1 = sketch.add_point(10.0, 20.0)
    point1 = sketch.registry.get_point(p1)
    cmd = RemoveItemsCommand(sketch, "Remove Items", points=[point1])

    for _ in range(3):
        cmd.execute()
        assert len(sketch.registry.points) == 1

        cmd.undo()
        assert len(sketch.registry.points) == 2


def test_add_and_remove_items_command_interaction(sketch):
    """Test interaction between add and remove items commands."""
    p1 = Point(-1, 10.0, 20.0)
    p2 = Point(-2, 30.0, 40.0)

    add_cmd = AddItemsCommand(sketch, "Add Items", points=[p1, p2])
    add_cmd.execute()

    point1 = sketch.registry.get_point(p1.id)
    point2 = sketch.registry.get_point(p2.id)

    remove_cmd = RemoveItemsCommand(
        sketch, "Remove Items", points=[point1, point2]
    )

    remove_cmd.execute()
    assert len(sketch.registry.points) == 1

    remove_cmd.undo()
    assert len(sketch.registry.points) == 3

    add_cmd.undo()
    assert len(sketch.registry.points) == 1


def test_add_items_command_with_temp_ids(sketch):
    """Test that temporary IDs are properly reassigned."""
    p1 = Point(-1, 10.0, 20.0)
    p2 = Point(-2, 30.0, 40.0)
    line = Line(-3, p1.id, p2.id)

    cmd = AddItemsCommand(
        sketch, "Add Items", points=[p1, p2], entities=[line]
    )

    cmd.execute()

    assert p1.id >= 0
    assert p2.id >= 0
    assert line.id >= 0
    assert line.p1_idx == p1.id
    assert line.p2_idx == p2.id
