import pytest
from raygeo.sketch.constraints.distance import DistanceConstraint
from raygeo.sketch.entities.line import Line

from sketcher.core import Sketch
from sketcher.core.commands import AddItemsCommand
from sketcher.core.commands.items import RemoveItemsCommand


@pytest.fixture
def sketch():
    """Create a basic sketch for testing."""
    return Sketch()


def test_add_items_command_initialization(sketch):
    """Test that AddItemsCommand initializes correctly."""
    cmd = AddItemsCommand(sketch, "Add Items")

    assert cmd.sketch is sketch
    assert cmd.name == "Add Items"
    assert cmd.point_ids == []
    assert cmd.entity_ids == []
    assert cmd.constraints == []


def test_add_items_command_add_points(sketch):
    """Test that the builder adds points to the sketch."""
    cmd = AddItemsCommand(sketch, "Add Items")
    initial_count = len(sketch.registry.points())

    pid1 = cmd.add_point(10.0, 20.0)
    pid2 = cmd.add_point(30.0, 40.0)
    cmd.execute()

    assert len(sketch.registry.points()) == initial_count + 2
    assert pid1 in cmd.point_ids
    assert pid2 in cmd.point_ids


def test_add_items_command_add_entities(sketch):
    """Test that the builder adds entities to the sketch."""
    cmd = AddItemsCommand(sketch, "Add Items")
    p1 = cmd.add_point(10.0, 20.0)
    p2 = cmd.add_point(30.0, 40.0)
    eid = cmd.add_entity(Line(p1, p2))
    cmd.execute()

    assert len(sketch.registry.entities()) == 1
    assert eid in cmd.entity_ids
    line = sketch.registry.get_entity_object(eid)
    assert line.p1_idx == p1
    assert line.p2_idx == p2


def test_add_items_command_add_constraints(sketch):
    """Test that the builder adds constraints to the sketch."""
    p1 = sketch.add_point(0.0, 0.0)
    p2 = sketch.add_point(10.0, 0.0)
    cmd = AddItemsCommand(sketch, "Add Items")
    constraint = cmd.add_constraint(DistanceConstraint(p1, p2, 10.0))
    cmd.execute()

    initial_count = len(sketch.constraints)
    assert initial_count == 1
    assert sketch.constraints[0] is constraint


def test_add_items_command_undo(sketch):
    """Test that undo removes added items from the sketch."""
    cmd = AddItemsCommand(sketch, "Add Items")
    p1 = cmd.add_point(10.0, 20.0)
    p2 = cmd.add_point(30.0, 40.0)
    cmd.add_entity(Line(p1, p2))
    cmd.execute()
    assert len(sketch.registry.points()) == 3
    assert len(sketch.registry.entities()) == 1

    cmd.undo()

    assert len(sketch.registry.points()) == 1
    assert len(sketch.registry.entities()) == 0


def test_add_items_command_execute_undo_cycle(sketch):
    """Test that execute and undo can be called multiple times."""
    cmd = AddItemsCommand(sketch, "Add Items")
    cmd.add_point(10.0, 20.0)
    cmd.execute()

    for _ in range(3):
        cmd._redo()
        assert len(sketch.registry.points()) == 2

        cmd.undo()
        assert len(sketch.registry.points()) == 1


def test_remove_items_command_initialization(sketch):
    """Test that RemoveItemsCommand initializes correctly."""
    cmd = RemoveItemsCommand(sketch, "Remove Items")

    assert cmd.sketch is sketch
    assert cmd.name == "Remove Items"
    assert cmd.point_dicts == []
    assert cmd.entity_ids == []
    assert cmd.constraints == []


def test_remove_items_command_remove_points(sketch):
    """Test that execute removes points from the sketch."""
    p1 = sketch.add_point(10.0, 20.0)
    p2 = sketch.add_point(30.0, 40.0)
    position1 = sketch.registry.get_point(p1)
    position2 = sketch.registry.get_point(p2)
    cmd = RemoveItemsCommand(
        sketch,
        "Remove Items",
        points=[
            {"id": p1, "x": position1[0], "y": position1[1], "fixed": False},
            {"id": p2, "x": position2[0], "y": position2[1], "fixed": False},
        ],
    )

    assert len(sketch.registry.points()) == 3

    cmd.execute()

    assert len(sketch.registry.points()) == 1


def test_remove_items_command_remove_entities(sketch):
    """Test that execute removes entities from the sketch."""
    p1 = sketch.add_point(10.0, 20.0)
    p2 = sketch.add_point(30.0, 40.0)
    line = sketch.add_line(p1, p2)

    cmd = RemoveItemsCommand(sketch, "Remove Items", entity_ids=[line])

    assert len(sketch.registry.entities()) == 1

    cmd.execute()

    assert len(sketch.registry.entities()) == 0


def test_remove_items_command_remove_constraints(sketch):
    """Test that execute removes constraints from the sketch."""
    p1 = sketch.add_point(0.0, 0.0)
    p2 = sketch.add_point(10.0, 0.0)
    constraint = sketch.add_constraint(DistanceConstraint(p1, p2, 10.0))

    assert len(sketch.constraints) == 1

    cmd = RemoveItemsCommand(sketch, "Remove Items", constraints=[constraint])
    cmd.execute()

    assert len(sketch.constraints) == 0


def test_remove_items_command_undo(sketch):
    """Test that undo restores removed items to the sketch."""
    p1 = sketch.add_point(10.0, 20.0)
    p2 = sketch.add_point(30.0, 40.0)
    line = sketch.add_line(p1, p2)
    position1 = sketch.registry.get_point(p1)
    position2 = sketch.registry.get_point(p2)

    cmd = RemoveItemsCommand(
        sketch,
        "Remove Items",
        points=[
            {"id": p1, "x": position1[0], "y": position1[1], "fixed": False},
            {"id": p2, "x": position2[0], "y": position2[1], "fixed": False},
        ],
        entity_ids=[line],
    )

    cmd.execute()
    assert len(sketch.registry.points()) == 1
    assert len(sketch.registry.entities()) == 0

    cmd.undo()

    assert len(sketch.registry.points()) == 3
    assert len(sketch.registry.entities()) == 1


def test_remove_items_command_execute_undo_cycle(sketch):
    """Test that execute and undo can be called multiple times."""
    p1 = sketch.add_point(10.0, 20.0)
    position1 = sketch.registry.get_point(p1)
    cmd = RemoveItemsCommand(
        sketch,
        "Remove Items",
        points=[
            {"id": p1, "x": position1[0], "y": position1[1], "fixed": False}
        ],
    )
    cmd.execute()

    for _ in range(3):
        cmd.undo()
        assert len(sketch.registry.points()) == 2

        cmd._redo()
        assert len(sketch.registry.points()) == 1


def test_add_and_remove_items_command_interaction(sketch):
    """Test interaction between add and remove items commands."""
    add_cmd = AddItemsCommand(sketch, "Add Items")
    p1 = add_cmd.add_point(10.0, 20.0)
    p2 = add_cmd.add_point(30.0, 40.0)
    add_cmd.execute()

    remove_cmd = RemoveItemsCommand(
        sketch,
        "Remove Items",
        points=[
            {"id": p1, "x": 10.0, "y": 20.0, "fixed": False},
            {"id": p2, "x": 30.0, "y": 40.0, "fixed": False},
        ],
    )

    remove_cmd.execute()
    assert len(sketch.registry.points()) == 1

    remove_cmd.undo()
    assert len(sketch.registry.points()) == 3

    add_cmd.undo()
    assert len(sketch.registry.points()) == 1


def test_add_items_entity_ids_are_stable_across_redo(sketch):
    """Test that redo restores the added entity with its original id."""
    cmd = AddItemsCommand(sketch, "Add Items")
    p1 = cmd.add_point(10.0, 20.0)
    p2 = cmd.add_point(30.0, 40.0)
    eid = cmd.add_entity(Line(p1, p2))
    cmd.execute()

    cmd.undo()
    cmd._redo()

    entity = sketch.registry.get_entity_object(eid)
    assert entity is not None
    assert entity.p1_idx == p1
    assert entity.p2_idx == p2
