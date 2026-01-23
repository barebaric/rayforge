from rayforge.core.undo.list_cmd import ListItemCommand, ReorderListCommand


class MockOwner:
    def __init__(self):
        self.items = []

    def add_item(self, item):
        self.items.append(item)

    def remove_item(self, item):
        self.items.remove(item)


def test_list_item_initialization():
    owner = MockOwner()
    cmd = ListItemCommand(
        owner, "item1", "remove_item", "add_item", name="add_cmd"
    )
    assert cmd.owner_obj is owner
    assert cmd.item == "item1"
    assert cmd.name == "add_cmd"


def test_list_item_execute():
    owner = MockOwner()
    cmd = ListItemCommand(
        owner, "item1", "remove_item", "add_item", name="add_cmd"
    )
    cmd.execute()
    assert "item1" in owner.items


def test_list_item_undo():
    owner = MockOwner()
    cmd = ListItemCommand(
        owner, "item1", "remove_item", "add_item", name="add_cmd"
    )
    cmd.execute()
    assert "item1" in owner.items
    cmd.undo()
    assert "item1" not in owner.items


def test_list_item_with_callback():
    callback_called = []

    def callback():
        callback_called.append(True)

    owner = MockOwner()
    cmd = ListItemCommand(
        owner, "item1", "remove_item", "add_item", callback, "add_cmd"
    )
    cmd.execute()
    assert len(callback_called) == 1
    cmd.undo()
    assert len(callback_called) == 2


def test_reorder_list_initialization():
    class Target:
        def __init__(self):
            self.items = [1, 2, 3]

    target = Target()
    cmd = ReorderListCommand(target, "items", [3, 2, 1])
    assert cmd.target_obj is target
    assert cmd.list_property_name == "items"
    assert cmd.new_list == [3, 2, 1]
    assert cmd.old_list == [1, 2, 3]


def test_reorder_list_execute():
    class Target:
        def __init__(self):
            self.items = [1, 2, 3]

    target = Target()
    cmd = ReorderListCommand(target, "items", [3, 2, 1])
    cmd.execute()
    assert target.items == [3, 2, 1]


def test_reorder_list_undo():
    class Target:
        def __init__(self):
            self.items = [1, 2, 3]

    target = Target()
    cmd = ReorderListCommand(target, "items", [3, 2, 1])
    cmd.execute()
    assert target.items == [3, 2, 1]
    cmd.undo()
    assert target.items == [1, 2, 3]


def test_reorder_list_with_setter():
    class Target:
        def __init__(self):
            self._items = [1, 2, 3]

        def get_items(self):
            return self._items

        def set_items(self, value):
            self._items = value

        items = property(get_items, set_items)

    target = Target()
    cmd = ReorderListCommand(target, "items", [3, 2, 1], "set_items")
    cmd.execute()
    assert target.items == [3, 2, 1]
    cmd.undo()
    assert target.items == [1, 2, 3]


def test_reorder_list_with_callback():
    callback_called = []

    def callback():
        callback_called.append(True)

    class Target:
        def __init__(self):
            self.items = [1, 2, 3]

    target = Target()
    cmd = ReorderListCommand(
        target, "items", [3, 2, 1], on_change_callback=callback
    )
    cmd.execute()
    assert len(callback_called) == 1
    cmd.undo()
    assert len(callback_called) == 2


def test_reorder_list_copy_lists():
    class Target:
        def __init__(self):
            self.items = [1, 2, 3]

    target = Target()
    cmd = ReorderListCommand(target, "items", [3, 2, 1])
    cmd.new_list.append(4)
    assert cmd.new_list == [3, 2, 1, 4]
    cmd.old_list.append(0)
    assert cmd.old_list == [1, 2, 3, 0]
    cmd.execute()
    assert target.items == [3, 2, 1, 4]
