from rayforge.core.undo.dict_cmd import DictItemCommand


def test_dict_item_initialization():
    target = {"key1": "value1"}
    cmd = DictItemCommand(target, "key1", "new_value", "test_cmd")
    assert cmd.target_dict is target
    assert cmd.key == "key1"
    assert cmd.new_value == "new_value"
    assert cmd.old_value == "value1"


def test_dict_item_execute():
    target = {"key1": "value1"}
    cmd = DictItemCommand(target, "key1", "new_value", "test_cmd")
    cmd.execute()
    assert target["key1"] == "new_value"


def test_dict_item_undo():
    target = {"key1": "value1"}
    cmd = DictItemCommand(target, "key1", "new_value", "test_cmd")
    cmd.execute()
    assert target["key1"] == "new_value"
    cmd.undo()
    assert target["key1"] == "value1"


def test_dict_item_undo_key_not_exists():
    target = {"key1": "value1"}
    cmd = DictItemCommand(target, "key2", "new_value", "test_cmd")
    assert cmd.old_value is None
    cmd.execute()
    assert target["key2"] == "new_value"
    cmd.undo()
    assert "key2" not in target


def test_dict_item_undo_different_value():
    target = {"key1": "value1"}
    cmd = DictItemCommand(target, "key1", "new_value", "test_cmd")
    cmd.execute()
    target["key1"] = "other_value"
    cmd.undo()
    assert target["key1"] == "value1"


def test_dict_item_with_callback():
    callback_called = []

    def callback():
        callback_called.append(True)

    target = {"key1": "value1"}
    cmd = DictItemCommand(target, "key1", "new_value", "test_cmd", callback)
    cmd.execute()
    assert len(callback_called) == 1
    cmd.undo()
    assert len(callback_called) == 2


def test_dict_item_can_coalesce_same_dict_key():
    target = {"key1": "value1"}
    cmd1 = DictItemCommand(target, "key1", "value2", "cmd1")
    cmd2 = DictItemCommand(target, "key1", "value3", "cmd2")
    assert cmd1.can_coalesce_with(cmd2) is True


def test_dict_item_cannot_coalesce_different_dict():
    target1 = {"key1": "value1"}
    target2 = {"key1": "value1"}
    cmd1 = DictItemCommand(target1, "key1", "value2", "cmd1")
    cmd2 = DictItemCommand(target2, "key1", "value3", "cmd2")
    assert cmd1.can_coalesce_with(cmd2) is False


def test_dict_item_cannot_coalesce_different_key():
    target = {"key1": "value1", "key2": "value2"}
    cmd1 = DictItemCommand(target, "key1", "value2", "cmd1")
    cmd2 = DictItemCommand(target, "key2", "value3", "cmd2")
    assert cmd1.can_coalesce_with(cmd2) is False


def test_dict_item_coalesce_success():
    target = {"key1": "value1"}
    cmd1 = DictItemCommand(target, "key1", "value2", "cmd1")
    cmd2 = DictItemCommand(target, "key1", "value3", "cmd2")
    result = cmd1.coalesce_with(cmd2)
    assert result is True
    assert cmd1.new_value == "value3"


def test_dict_item_coalesce_failure():
    target1 = {"key1": "value1"}
    target2 = {"key1": "value1"}
    cmd1 = DictItemCommand(target1, "key1", "value2", "cmd1")
    cmd2 = DictItemCommand(target2, "key1", "value3", "cmd2")
    result = cmd1.coalesce_with(cmd2)
    assert result is False
    assert cmd1.new_value == "value2"
