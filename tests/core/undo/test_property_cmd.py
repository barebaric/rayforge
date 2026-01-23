from rayforge.core.undo.property_cmd import ChangePropertyCommand


class MockTarget:
    def __init__(self):
        self.value = 0
        self._protected = 0

    def set_protected(self, value):
        self._protected = value

    def get_protected(self):
        return self._protected

    protected = property(get_protected, set_protected)


def test_property_initialization():
    target = MockTarget()
    cmd = ChangePropertyCommand(target, "value", 10, name="set_value")
    assert cmd.target is target
    assert cmd.property_name == "value"
    assert cmd.new_value == 10
    assert cmd.old_value == 0


def test_property_initialization_with_old_value():
    target = MockTarget()
    cmd = ChangePropertyCommand(
        target, "value", 10, old_value=5, name="set_value"
    )
    assert cmd.new_value == 10
    assert cmd.old_value == 5


def test_property_execute():
    target = MockTarget()
    cmd = ChangePropertyCommand(target, "value", 10, name="set_value")
    cmd.execute()
    assert target.value == 10


def test_property_undo():
    target = MockTarget()
    cmd = ChangePropertyCommand(target, "value", 10, name="set_value")
    cmd.execute()
    assert target.value == 10
    cmd.undo()
    assert target.value == 0


def test_property_with_setter():
    target = MockTarget()
    cmd = ChangePropertyCommand(
        target, "protected", 10, setter_method_name="set_protected"
    )
    cmd.execute()
    assert target._protected == 10
    cmd.undo()
    assert target._protected == 0


def test_property_with_callback():
    callback_called = []

    def callback():
        callback_called.append(True)

    target = MockTarget()
    cmd = ChangePropertyCommand(
        target, "value", 10, on_change_callback=callback
    )
    cmd.execute()
    assert len(callback_called) == 1
    cmd.undo()
    assert len(callback_called) == 2


def test_property_can_coalesce_same_target_property():
    target = MockTarget()
    cmd1 = ChangePropertyCommand(target, "value", 10, name="cmd1")
    cmd2 = ChangePropertyCommand(target, "value", 20, name="cmd2")
    assert cmd1.can_coalesce_with(cmd2) is True


def test_property_cannot_coalesce_different_target():
    target1 = MockTarget()
    target2 = MockTarget()
    cmd1 = ChangePropertyCommand(target1, "value", 10, name="cmd1")
    cmd2 = ChangePropertyCommand(target2, "value", 20, name="cmd2")
    assert cmd1.can_coalesce_with(cmd2) is False


def test_property_cannot_coalesce_different_property():
    target = MockTarget()
    cmd1 = ChangePropertyCommand(target, "value", 10, name="cmd1")
    cmd2 = ChangePropertyCommand(target, "_protected", 20, name="cmd2")
    assert cmd1.can_coalesce_with(cmd2) is False


def test_property_coalesce_success():
    target = MockTarget()
    cmd1 = ChangePropertyCommand(target, "value", 10, name="cmd1")
    cmd2 = ChangePropertyCommand(target, "value", 20, name="cmd2")
    result = cmd1.coalesce_with(cmd2)
    assert result is True
    assert cmd1.new_value == 20


def test_property_coalesce_failure():
    target1 = MockTarget()
    target2 = MockTarget()
    cmd1 = ChangePropertyCommand(target1, "value", 10, name="cmd1")
    cmd2 = ChangePropertyCommand(target2, "value", 20, name="cmd2")
    result = cmd1.coalesce_with(cmd2)
    assert result is False
    assert cmd1.new_value == 10


def test_property_undo_with_old_value_provided():
    target = MockTarget()
    target.value = 5
    cmd = ChangePropertyCommand(
        target, "value", 10, old_value=5, name="set_value"
    )
    cmd.execute()
    assert target.value == 10
    cmd.undo()
    assert target.value == 5
