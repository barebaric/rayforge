import pytest

from rayforge.core.sketcher import Sketch
from rayforge.core.sketcher.sketch import Fill
from rayforge.core.sketcher.commands import AddFillCommand, RemoveFillCommand


@pytest.fixture
def sketch():
    """Create a basic sketch for testing."""
    return Sketch()


@pytest.fixture
def boundary():
    """Create a boundary for testing fills."""
    return [(1, True), (2, False), (3, True)]


def test_add_fill_command_initialization(sketch, boundary):
    """Test that AddFillCommand initializes correctly."""
    cmd = AddFillCommand(sketch, boundary, "Add Fill")

    assert cmd.sketch is sketch
    assert cmd.name == "Add Fill"
    assert cmd._boundary == boundary
    assert cmd.fill is None


def test_add_fill_command_execute(sketch, boundary):
    """Test that execute adds a fill to the sketch."""
    cmd = AddFillCommand(sketch, boundary, "Add Fill")

    assert len(sketch.fills) == 0

    cmd.execute()

    assert len(sketch.fills) == 1
    assert cmd.fill is not None
    assert cmd.fill in sketch.fills
    assert cmd.fill.boundary == boundary


def test_add_fill_command_creates_fill_once(sketch, boundary):
    """Test that execute creates fill on first call."""
    cmd = AddFillCommand(sketch, boundary, "Add Fill")

    assert cmd.fill is None

    cmd.execute()

    assert cmd.fill is not None
    assert cmd.fill in sketch.fills


def test_add_fill_command_undo(sketch, boundary):
    """Test that undo removes the fill from the sketch."""
    cmd = AddFillCommand(sketch, boundary, "Add Fill")

    cmd.execute()
    assert len(sketch.fills) == 1

    cmd.undo()

    assert len(sketch.fills) == 0


def test_add_fill_command_execute_undo_cycle(sketch, boundary):
    """Test that execute and undo can be called multiple times."""
    cmd = AddFillCommand(sketch, boundary, "Add Fill")

    for _ in range(3):
        cmd.execute()
        assert len(sketch.fills) == 1

        cmd.undo()
        assert len(sketch.fills) == 0


def test_remove_fill_command_initialization(sketch):
    """Test that RemoveFillCommand initializes correctly."""
    fill = Fill(uid="test-uid", boundary=[(1, True)])
    cmd = RemoveFillCommand(sketch, fill, "Remove Fill")

    assert cmd.sketch is sketch
    assert cmd.name == "Remove Fill"
    assert cmd.fill is fill


def test_remove_fill_command_execute(sketch):
    """Test that execute removes a fill from the sketch."""
    fill = Fill(uid="test-uid", boundary=[(1, True)])
    sketch.fills.append(fill)

    assert len(sketch.fills) == 1

    cmd = RemoveFillCommand(sketch, fill, "Remove Fill")
    cmd.execute()

    assert len(sketch.fills) == 0
    assert fill not in sketch.fills


def test_remove_fill_command_execute_nonexistent_fill(sketch):
    """Test that execute handles fill not in sketch gracefully."""
    fill = Fill(uid="test-uid", boundary=[(1, True)])

    assert len(sketch.fills) == 0

    cmd = RemoveFillCommand(sketch, fill, "Remove Fill")
    cmd.execute()

    assert len(sketch.fills) == 0


def test_remove_fill_command_undo(sketch):
    """Test that undo restores the fill to the sketch."""
    fill = Fill(uid="test-uid", boundary=[(1, True)])
    sketch.fills.append(fill)

    cmd = RemoveFillCommand(sketch, fill, "Remove Fill")
    cmd.execute()

    assert len(sketch.fills) == 0

    cmd.undo()

    assert len(sketch.fills) == 1
    assert fill in sketch.fills


def test_remove_fill_command_execute_undo_cycle(sketch):
    """Test that execute and undo can be called multiple times."""
    fill = Fill(uid="test-uid", boundary=[(1, True)])
    sketch.fills.append(fill)

    cmd = RemoveFillCommand(sketch, fill, "Remove Fill")

    for _ in range(3):
        cmd.execute()
        assert len(sketch.fills) == 0

        cmd.undo()
        assert len(sketch.fills) == 1


def test_add_and_remove_fill_command_interaction(sketch, boundary):
    """Test interaction between add and remove fill commands."""
    add_cmd = AddFillCommand(sketch, boundary, "Add Fill")
    add_cmd.execute()

    fill = add_cmd.fill
    assert fill is not None
    remove_cmd = RemoveFillCommand(sketch, fill, "Remove Fill")

    remove_cmd.execute()
    assert len(sketch.fills) == 0

    remove_cmd.undo()
    assert len(sketch.fills) == 1

    add_cmd.undo()
    assert len(sketch.fills) == 0
