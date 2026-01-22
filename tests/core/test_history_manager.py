from rayforge.core.undo import HistoryManager
from rayforge.core.undo.property_cmd import ChangePropertyCommand


class MockObj:
    def __init__(self):
        self.value = 0


def test_checkpoint_initial_state():
    """Test that a new HistoryManager is at checkpoint by default."""
    hm = HistoryManager()
    assert hm.is_at_checkpoint() is True


def test_set_checkpoint_with_empty_history():
    """Test setting checkpoint when history is empty."""
    hm = HistoryManager()
    hm.set_checkpoint()
    assert hm.is_at_checkpoint() is True


def test_set_checkpoint_with_commands():
    """Test setting checkpoint after executing commands."""
    hm = HistoryManager()
    obj = MockObj()

    cmd1 = ChangePropertyCommand(obj, "value", 1)
    hm.execute(cmd1)

    assert hm.is_at_checkpoint() is False

    hm.set_checkpoint()
    assert hm.is_at_checkpoint() is True


def test_checkpoint_after_undo():
    """Test that undoing past checkpoint returns to not at checkpoint."""
    hm = HistoryManager()
    obj = MockObj()

    cmd1 = ChangePropertyCommand(obj, "value", 1)
    hm.execute(cmd1)

    hm.set_checkpoint()
    assert hm.is_at_checkpoint() is True
    assert obj.value == 1

    hm.undo()
    assert obj.value == 0
    assert hm.is_at_checkpoint() is False


def test_checkpoint_after_redo():
    """Test that redoing to checkpoint returns to at checkpoint."""
    hm = HistoryManager()
    obj = MockObj()

    cmd1 = ChangePropertyCommand(obj, "value", 1)
    hm.execute(cmd1)

    hm.set_checkpoint()
    assert hm.is_at_checkpoint() is True

    hm.undo()
    assert hm.is_at_checkpoint() is False

    hm.redo()
    assert hm.is_at_checkpoint() is True


def test_clear_checkpoint():
    """Test clearing the checkpoint."""
    hm = HistoryManager()
    obj = MockObj()

    cmd1 = ChangePropertyCommand(obj, "value", 1)
    hm.execute(cmd1)

    hm.set_checkpoint()
    assert hm.is_at_checkpoint() is True

    hm.clear_checkpoint()
    assert hm.is_at_checkpoint() is True


def test_checkpoint_with_multiple_commands():
    """Test checkpoint behavior with multiple commands."""
    hm = HistoryManager()
    obj = MockObj()

    cmd1 = ChangePropertyCommand(obj, "value", 1)
    hm.execute(cmd1)

    cmd2 = ChangePropertyCommand(obj, "value", 2)
    hm.execute(cmd2)

    hm.set_checkpoint()
    assert hm.is_at_checkpoint() is True
    assert obj.value == 2

    hm.undo()
    assert hm.is_at_checkpoint() is False
    assert obj.value == 0

    hm.redo()
    assert hm.is_at_checkpoint() is True
    assert obj.value == 2


def test_checkpoint_after_new_command():
    """Test that executing new command after checkpoint moves away."""
    hm = HistoryManager()
    obj = MockObj()

    cmd1 = ChangePropertyCommand(obj, "value", 1)
    hm.execute(cmd1)

    hm.set_checkpoint()
    assert hm.is_at_checkpoint() is True

    cmd2 = ChangePropertyCommand(obj, "value", 2)
    hm.execute(cmd2)

    assert hm.is_at_checkpoint() is True
    assert obj.value == 2


def test_checkpoint_clears_on_history_clear():
    """Test that clearing history also clears checkpoint."""
    hm = HistoryManager()
    obj = MockObj()

    cmd1 = ChangePropertyCommand(obj, "value", 1)
    hm.execute(cmd1)

    hm.set_checkpoint()
    assert hm.is_at_checkpoint() is True

    hm.clear()
    assert hm.is_at_checkpoint() is True


def test_checkpoint_with_transactions():
    """Test checkpoint behavior with transactions."""
    hm = HistoryManager()
    obj = MockObj()

    with hm.transaction("Batch") as t:
        t.execute(ChangePropertyCommand(obj, "value", 1))
        t.execute(ChangePropertyCommand(obj, "value", 2))

    hm.set_checkpoint()
    assert hm.is_at_checkpoint() is True
    assert obj.value == 2

    hm.undo()
    assert hm.is_at_checkpoint() is False
    assert obj.value == 0

    hm.redo()
    assert hm.is_at_checkpoint() is True
    assert obj.value == 2
