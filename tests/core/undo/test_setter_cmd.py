from rayforge.core.undo.setter_cmd import SetterCommand


class MockTarget:
    def __init__(self):
        self.value = 0

    def set_value(self, val):
        self.value = val

    def set_multiple(self, a, b, c):
        self.value = a + b + c


def test_setter_initialization():
    target = MockTarget()
    cmd = SetterCommand(target, "set_value", (10,), (0,), name="set_value_cmd")
    assert cmd.target is target
    assert cmd.setter_method_name == "set_value"
    assert cmd.new_args == (10,)
    assert cmd.old_args == (0,)


def test_setter_execute():
    target = MockTarget()
    cmd = SetterCommand(target, "set_value", (10,), (0,), name="set_value_cmd")
    cmd.execute()
    assert target.value == 10


def test_setter_undo():
    target = MockTarget()
    cmd = SetterCommand(target, "set_value", (10,), (0,), name="set_value_cmd")
    cmd.execute()
    assert target.value == 10
    cmd.undo()
    assert target.value == 0


def test_setter_with_multiple_args():
    target = MockTarget()
    cmd = SetterCommand(
        target, "set_multiple", (1, 2, 3), (0, 0, 0), name="set_multi_cmd"
    )
    cmd.execute()
    assert target.value == 6
    cmd.undo()
    assert target.value == 0


def test_setter_with_callback():
    callback_called = []

    def callback():
        callback_called.append(True)

    target = MockTarget()
    cmd = SetterCommand(
        target, "set_value", (10,), (0,), callback, "set_value_cmd"
    )
    cmd.execute()
    assert len(callback_called) == 1
    cmd.undo()
    assert len(callback_called) == 2


def test_setter_can_coalesce_same_target_method():
    target = MockTarget()
    cmd1 = SetterCommand(target, "set_value", (10,), (0,), name="cmd1")
    cmd2 = SetterCommand(target, "set_value", (20,), (0,), name="cmd2")
    assert cmd1.can_coalesce_with(cmd2) is True


def test_setter_cannot_coalesce_different_target():
    target1 = MockTarget()
    target2 = MockTarget()
    cmd1 = SetterCommand(target1, "set_value", (10,), (0,), name="cmd1")
    cmd2 = SetterCommand(target2, "set_value", (20,), (0,), name="cmd2")
    assert cmd1.can_coalesce_with(cmd2) is False


def test_setter_cannot_coalesce_different_method():
    target = MockTarget()
    cmd1 = SetterCommand(target, "set_value", (10,), (0,), name="cmd1")
    cmd2 = SetterCommand(
        target, "set_multiple", (1, 2, 3), (0, 0, 0), name="cmd2"
    )
    assert cmd1.can_coalesce_with(cmd2) is False


def test_setter_coalesce_success():
    target = MockTarget()
    cmd1 = SetterCommand(target, "set_value", (10,), (0,), name="cmd1")
    cmd2 = SetterCommand(target, "set_value", (20,), (0,), name="cmd2")
    result = cmd1.coalesce_with(cmd2)
    assert result is True
    assert cmd1.new_args == (20,)


def test_setter_coalesce_failure():
    target1 = MockTarget()
    target2 = MockTarget()
    cmd1 = SetterCommand(target1, "set_value", (10,), (0,), name="cmd1")
    cmd2 = SetterCommand(target2, "set_value", (20,), (0,), name="cmd2")
    result = cmd1.coalesce_with(cmd2)
    assert result is False
    assert cmd1.new_args == (10,)


def test_setter_coalesce_multiple_args():
    target = MockTarget()
    cmd1 = SetterCommand(
        target, "set_multiple", (1, 2, 3), (0, 0, 0), name="cmd1"
    )
    cmd2 = SetterCommand(
        target, "set_multiple", (4, 5, 6), (0, 0, 0), name="cmd2"
    )
    result = cmd1.coalesce_with(cmd2)
    assert result is True
    assert cmd1.new_args == (4, 5, 6)
