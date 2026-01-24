from rayforge.core.undo import HistoryManager
from rayforge.core.undo.command import Command
from rayforge.core.undo.property_cmd import ChangePropertyCommand


class MockObj:
    def __init__(self):
        self.value = 0
        self.other_value = 0


class SkipUndoCommand(Command):
    """Command that should be skipped from undo stack."""

    def __init__(self, skip=False):
        super().__init__()
        self._skip = skip
        self.executed = False

    def execute(self):
        self.executed = True

    def undo(self):
        pass

    def should_skip_undo(self):
        return self._skip


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


def test_should_skip_undo_not_skipped():
    """Test that commands with should_skip_undo=False are added to history."""
    hm = HistoryManager()
    cmd = SkipUndoCommand(skip=False)
    hm.execute(cmd)

    assert hm.can_undo() is True
    assert len(hm.undo_stack) == 1
    assert cmd.executed is True


def test_should_skip_undo_skipped():
    """Test commands with should_skip_undo=True are not added to history."""
    hm = HistoryManager()
    cmd = SkipUndoCommand(skip=True)
    hm.execute(cmd)

    assert hm.can_undo() is False
    assert len(hm.undo_stack) == 0
    assert cmd.executed is True


def test_should_skip_undo_mixed_commands():
    """Test history with mix of skip and non-skip commands."""
    hm = HistoryManager()
    obj = MockObj()

    cmd1 = ChangePropertyCommand(obj, "value", 1)
    hm.execute(cmd1)

    skip_cmd = SkipUndoCommand(skip=True)
    hm.execute(skip_cmd)

    cmd2 = ChangePropertyCommand(obj, "other_value", 2)
    hm.execute(cmd2)

    assert len(hm.undo_stack) == 2
    assert hm.can_undo() is True
    assert obj.value == 1
    assert obj.other_value == 2


def test_should_skip_undo_with_transactions():
    """Test that skip-undo commands in transactions are handled correctly."""
    hm = HistoryManager()
    obj = MockObj()

    with hm.transaction("Mixed") as t:
        t.execute(ChangePropertyCommand(obj, "value", 1))
        t.execute(SkipUndoCommand(skip=True))
        t.execute(ChangePropertyCommand(obj, "value", 2))

    assert len(hm.undo_stack) == 1
    assert hm.can_undo() is True
    assert obj.value == 2


def test_should_skip_undo_all_skipped_in_transaction():
    """Test transaction with only skip-undo commands."""
    hm = HistoryManager()

    with hm.transaction("All Skip") as t:
        t.execute(SkipUndoCommand(skip=True))
        t.execute(SkipUndoCommand(skip=True))

    assert len(hm.undo_stack) == 1
    assert hm.can_undo() is True
