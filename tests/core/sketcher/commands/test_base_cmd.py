import pytest

from rayforge.core.sketcher import Sketch
from rayforge.core.sketcher.commands import SketchChangeCommand


class ConcreteSketchChangeCommand(SketchChangeCommand):
    """Concrete implementation for testing SketchChangeCommand."""

    def __init__(self, sketch: "Sketch", name: str = "Test Command"):
        super().__init__(sketch, name)
        self.executed = False
        self.undone = False

    def _do_execute(self) -> None:
        self.executed = True

    def _do_undo(self) -> None:
        self.undone = True


@pytest.fixture
def sketch():
    """Create a basic sketch for testing."""
    return Sketch()


@pytest.fixture
def command(sketch):
    """Create a concrete command for testing."""
    return ConcreteSketchChangeCommand(sketch, "Test Command")


def test_sketch_change_command_initialization(sketch):
    """Test that SketchChangeCommand initializes correctly."""
    cmd = ConcreteSketchChangeCommand(sketch, "Test Command")
    assert cmd.sketch is sketch
    assert cmd.name == "Test Command"
    assert cmd._state_snapshot == {}
    assert not cmd.executed
    assert not cmd.undone


def test_capture_snapshot(sketch, command):
    """Test that capture_snapshot stores point coordinates."""
    p1_id = sketch.add_point(10.0, 20.0)
    p2_id = sketch.add_point(30.0, 40.0)

    command.capture_snapshot()

    assert len(command._state_snapshot) == 3
    assert command._state_snapshot[p1_id] == (10.0, 20.0)
    assert command._state_snapshot[p2_id] == (30.0, 40.0)


def test_restore_snapshot(sketch, command):
    """Test that restore_snapshot restores point coordinates."""
    p1_id = sketch.add_point(10.0, 20.0)
    p2_id = sketch.add_point(30.0, 40.0)

    command.capture_snapshot()

    sketch.registry.get_point(p1_id).x = 100.0
    sketch.registry.get_point(p1_id).y = 200.0
    sketch.registry.get_point(p2_id).x = 300.0
    sketch.registry.get_point(p2_id).y = 400.0

    command.restore_snapshot()

    assert sketch.registry.get_point(p1_id).x == 10.0
    assert sketch.registry.get_point(p1_id).y == 20.0
    assert sketch.registry.get_point(p2_id).x == 30.0
    assert sketch.registry.get_point(p2_id).y == 40.0


def test_restore_snapshot_with_missing_point(sketch, command):
    """Test that restore_snapshot handles missing points gracefully."""
    sketch.add_point(10.0, 20.0)
    command.capture_snapshot()

    sketch.registry.points = []

    command.restore_snapshot()

    assert not command.executed


def test_execute_captures_snapshot_if_empty(sketch, command):
    """Test that execute captures snapshot if not already done."""
    p1_id = sketch.add_point(10.0, 20.0)

    assert command._state_snapshot == {}

    command.execute()

    assert command._state_snapshot[p1_id] == (10.0, 20.0)
    assert command.executed


def test_execute_uses_existing_snapshot(sketch):
    """Test that execute uses existing snapshot."""
    p1_id = sketch.add_point(10.0, 20.0)
    cmd = ConcreteSketchChangeCommand(sketch, "Test Command")

    cmd.capture_snapshot()
    sketch.registry.get_point(p1_id).x = 100.0

    cmd.execute()

    assert cmd._state_snapshot[p1_id] == (10.0, 20.0)
    assert cmd.executed


def test_undo_restores_snapshot(sketch, command):
    """Test that undo restores the snapshot."""
    p1_id = sketch.add_point(10.0, 20.0)

    command.execute()
    sketch.registry.get_point(p1_id).x = 100.0
    sketch.registry.get_point(p1_id).y = 200.0

    command.undo()

    assert command.undone
    assert sketch.registry.get_point(p1_id).x == 10.0
    assert sketch.registry.get_point(p1_id).y == 20.0


def test_abstract_methods_raise_not_implemented(sketch):
    """Test that abstract methods raise NotImplementedError."""

    class IncompleteCommand(SketchChangeCommand):
        pass

    cmd = IncompleteCommand(sketch, "Incomplete")

    with pytest.raises(NotImplementedError):
        cmd._do_execute()

    with pytest.raises(NotImplementedError):
        cmd._do_undo()
