from rayforge.core.undo.command import Command


class MockCommand(Command):
    def __init__(self, name=None, on_change_callback=None):
        super().__init__(name, on_change_callback)
        self.executed = False
        self.undone = False

    def execute(self):
        self.executed = True

    def undo(self):
        self.undone = True


def test_command_initialization():
    cmd = MockCommand(name="test_cmd")
    assert cmd.name == "test_cmd"
    assert cmd.on_change_callback is None
    assert cmd.timestamp > 0


def test_command_with_callback():
    callback_called = []

    def callback():
        callback_called.append(True)

    cmd = MockCommand(name="cmd", on_change_callback=callback)
    assert cmd.on_change_callback is callback


def test_command_execute():
    cmd = MockCommand()
    cmd.execute()
    assert cmd.executed is True


def test_command_undo():
    cmd = MockCommand()
    cmd.undo()
    assert cmd.undone is True


def test_can_coalesce_with_default():
    cmd1 = MockCommand()
    cmd2 = MockCommand()
    assert cmd1.can_coalesce_with(cmd2) is False


def test_coalesce_with_default():
    cmd1 = MockCommand()
    cmd2 = MockCommand()
    assert cmd1.coalesce_with(cmd2) is False


def test_command_abstract_methods():
    from abc import ABC

    assert issubclass(Command, ABC)


def test_should_skip_undo_default():
    """Test that default should_skip_undo returns False."""
    cmd = MockCommand()
    assert cmd.should_skip_undo() is False


class SkipUndoCommand(Command):
    def __init__(self, skip=False):
        super().__init__()
        self._skip = skip

    def execute(self):
        pass

    def undo(self):
        pass

    def should_skip_undo(self):
        return self._skip


def test_should_skip_undo_custom_true():
    """Test that custom command can override to return True."""
    cmd = SkipUndoCommand(skip=True)
    assert cmd.should_skip_undo() is True


def test_should_skip_undo_custom_false():
    """Test that custom command can override to return False."""
    cmd = SkipUndoCommand(skip=False)
    assert cmd.should_skip_undo() is False
