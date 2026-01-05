import pytest

from rayforge.core.sketcher import Sketch
from rayforge.core.sketcher.commands import ToggleConstructionCommand


@pytest.fixture
def sketch():
    """Create a basic sketch for testing."""
    return Sketch()


@pytest.fixture
def entities(sketch):
    """Create entities for testing."""
    p1 = sketch.add_point(0.0, 0.0)
    p2 = sketch.add_point(10.0, 0.0)
    line1_id = sketch.add_line(p1, p2)
    p3 = sketch.add_point(10.0, 10.0)
    line2_id = sketch.add_line(p2, p3)
    return [line1_id, line2_id]


def test_toggle_construction_command_initialization(sketch, entities):
    """Test that ToggleConstructionCommand initializes correctly."""
    cmd = ToggleConstructionCommand(sketch, "Toggle", entities)

    assert cmd.sketch is sketch
    assert cmd.name == "Toggle"
    assert cmd.entity_ids == entities
    assert cmd.original_states == {}
    assert cmd.new_state is None


def test_toggle_construction_command_normal_to_construction(sketch, entities):
    """Test toggling from normal to construction state."""
    cmd = ToggleConstructionCommand(sketch, "Toggle", entities)

    for eid in entities:
        ent = sketch.registry.get_entity(eid)
        assert not ent.construction

    cmd.execute()

    for eid in entities:
        ent = sketch.registry.get_entity(eid)
        assert ent.construction

    assert cmd.new_state is True


def test_toggle_construction_command_construction_to_normal(sketch, entities):
    """Test toggling from construction to normal state."""
    for eid in entities:
        ent = sketch.registry.get_entity(eid)
        ent.construction = True

    cmd = ToggleConstructionCommand(sketch, "Toggle", entities)

    for eid in entities:
        ent = sketch.registry.get_entity(eid)
        assert ent.construction

    cmd.execute()

    for eid in entities:
        ent = sketch.registry.get_entity(eid)
        assert not ent.construction

    assert cmd.new_state is False


def test_toggle_construction_command_mixed_states(sketch, entities):
    """Test toggling with mixed initial states."""
    ent1 = sketch.registry.get_entity(entities[0])
    ent1.construction = True

    cmd = ToggleConstructionCommand(sketch, "Toggle", entities)

    cmd.execute()

    for eid in entities:
        ent = sketch.registry.get_entity(eid)
        assert ent.construction

    assert cmd.new_state is True


def test_toggle_construction_command_undo(sketch, entities):
    """Test that undo restores original construction states."""
    ent1 = sketch.registry.get_entity(entities[0])
    ent1.construction = True

    cmd = ToggleConstructionCommand(sketch, "Toggle", entities)
    cmd.execute()

    for eid in entities:
        ent = sketch.registry.get_entity(eid)
        assert ent.construction

    cmd.undo()

    assert sketch.registry.get_entity(entities[0]).construction is True
    assert sketch.registry.get_entity(entities[1]).construction is False


def test_toggle_construction_command_with_invalid_entity_id(sketch):
    """Test that invalid entity IDs are handled gracefully."""
    cmd = ToggleConstructionCommand(sketch, "Toggle", [9999])

    cmd.execute()

    assert cmd.original_states == {}
    assert cmd.new_state is None


def test_toggle_construction_cmd_with_mixed_valid_invalid(sketch, entities):
    """Test with a mix of valid and invalid entity IDs."""
    cmd = ToggleConstructionCommand(sketch, "Toggle", entities + [9999])

    cmd.execute()

    for eid in entities:
        ent = sketch.registry.get_entity(eid)
        assert ent.construction

    assert cmd.new_state is True


def test_toggle_construction_command_execute_undo_cycle(sketch, entities):
    """Test that execute and undo can be called multiple times."""
    cmd = ToggleConstructionCommand(sketch, "Toggle", entities)

    for _ in range(3):
        cmd.execute()
        for eid in entities:
            assert sketch.registry.get_entity(eid).construction

        cmd.undo()
        for eid in entities:
            assert not sketch.registry.get_entity(eid).construction


def test_toggle_construction_command_empty_entity_list(sketch):
    """Test with an empty list of entity IDs."""
    cmd = ToggleConstructionCommand(sketch, "Toggle", [])

    cmd.execute()

    assert cmd.original_states == {}
    assert cmd.new_state is None
