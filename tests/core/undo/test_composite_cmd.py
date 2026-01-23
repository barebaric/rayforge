from rayforge.core.undo.composite_cmd import CompositeCommand
from rayforge.core.undo.command import Command


class SimpleCommand(Command):
    def __init__(self, name=None, on_change_callback=None):
        super().__init__(name, on_change_callback)
        self.executed = False
        self.undone = False

    def execute(self):
        self.executed = True

    def undo(self):
        self.undone = True


class CoalescableCommand(Command):
    def __init__(self, name=None, on_change_callback=None):
        super().__init__(name, on_change_callback)
        self.executed = False
        self.undone = False
        self.coalesced = False

    def execute(self):
        self.executed = True

    def undo(self):
        self.undone = True

    def can_coalesce_with(self, next_command):
        return isinstance(next_command, CoalescableCommand)

    def coalesce_with(self, next_command):
        self.coalesced = True
        self.timestamp = next_command.timestamp
        return True


def test_composite_initialization():
    cmd1 = SimpleCommand(name="cmd1")
    cmd2 = SimpleCommand(name="cmd2")
    composite = CompositeCommand([cmd1, cmd2], "composite")
    assert composite.name == "composite"
    assert len(composite.commands) == 2


def test_composite_execute():
    cmd1 = SimpleCommand(name="cmd1")
    cmd2 = SimpleCommand(name="cmd2")
    composite = CompositeCommand([cmd1, cmd2], "composite")
    composite.execute()
    assert cmd1.executed is True
    assert cmd2.executed is True


def test_composite_undo():
    cmd1 = SimpleCommand(name="cmd1")
    cmd2 = SimpleCommand(name="cmd2")
    composite = CompositeCommand([cmd1, cmd2], "composite")
    composite.execute()
    composite.undo()
    assert cmd1.undone is True
    assert cmd2.undone is True


def test_composite_with_callback():
    callback_called = []

    def callback():
        callback_called.append(True)

    cmd1 = SimpleCommand(name="cmd1")
    cmd2 = SimpleCommand(name="cmd2")
    composite = CompositeCommand([cmd1, cmd2], "composite", callback)
    composite.execute()
    assert len(callback_called) == 1
    composite.undo()
    assert len(callback_called) == 2


def test_composite_cannot_coalesce_different_types():
    cmd1 = SimpleCommand(name="cmd1")
    cmd2 = SimpleCommand(name="cmd2")
    composite1 = CompositeCommand([cmd1], "comp1")
    composite2 = CompositeCommand([cmd2], "comp2")
    assert composite1.can_coalesce_with(composite2) is False


def test_composite_cannot_coalesce_different_counts():
    cmd1 = SimpleCommand(name="cmd1")
    cmd2 = SimpleCommand(name="cmd2")
    cmd3 = SimpleCommand(name="cmd3")
    composite1 = CompositeCommand([cmd1, cmd2], "comp1")
    composite2 = CompositeCommand([cmd3], "comp2")
    assert composite1.can_coalesce_with(composite2) is False


def test_composite_can_coalesce_same_types():
    cmd1 = CoalescableCommand(name="cmd1")
    cmd2 = CoalescableCommand(name="cmd2")
    composite1 = CompositeCommand([cmd1], "comp1")
    composite2 = CompositeCommand([cmd2], "comp2")
    assert composite1.can_coalesce_with(composite2) is True


def test_composite_coalesce_success():
    cmd1 = CoalescableCommand(name="cmd1")
    cmd2 = CoalescableCommand(name="cmd2")
    composite1 = CompositeCommand([cmd1], "comp1")
    composite2 = CompositeCommand([cmd2], "comp2")
    result = composite1.coalesce_with(composite2)
    assert result is True
    assert cmd1.coalesced is True


def test_composite_coalesce_failure():
    cmd1 = SimpleCommand(name="cmd1")
    cmd2 = SimpleCommand(name="cmd2")
    composite1 = CompositeCommand([cmd1], "comp1")
    composite2 = CompositeCommand([cmd2], "comp2")
    result = composite1.coalesce_with(composite2)
    assert result is False
