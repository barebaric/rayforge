from rayforge.core.menu_registry import MenuRegistry, menu_registry


class TestMenuRegistry:
    """Tests for MenuRegistry class."""

    def test_register_menu_item(self):
        registry = MenuRegistry()
        registry.register(
            item_id="test.action",
            label="Test Action",
            action="win.test_action",
            menu="Tools",
        )

        items = registry.get_items_for_menu("Tools")
        assert len(items) == 1
        assert items[0].item_id == "test.action"
        assert items[0].label == "Test Action"
        assert items[0].action == "win.test_action"

    def test_register_with_addon_name(self):
        registry = MenuRegistry()
        registry.register(
            item_id="addon.action",
            label="Addon Action",
            action="win.addon_action",
            menu="Tools",
            addon_name="my_addon",
        )

        items = registry.get_items_for_menu("Tools")
        assert len(items) == 1
        assert items[0].addon_name == "my_addon"

    def test_unregister_menu_item(self):
        registry = MenuRegistry()
        registry.register(
            item_id="test.action",
            label="Test",
            action="win.test",
            menu="Tools",
        )

        result = registry.unregister("test.action")
        assert result is True
        assert len(registry.get_items_for_menu("Tools")) == 0

    def test_unregister_nonexistent(self):
        registry = MenuRegistry()
        result = registry.unregister("nonexistent")
        assert result is False

    def test_unregister_all_from_addon(self):
        registry = MenuRegistry()
        registry.register(
            item_id="addon.action1",
            label="Action 1",
            action="win.action1",
            menu="Tools",
            addon_name="my_addon",
        )
        registry.register(
            item_id="addon.action2",
            label="Action 2",
            action="win.action2",
            menu="Tools",
            addon_name="my_addon",
        )
        registry.register(
            item_id="other.action",
            label="Other",
            action="win.other",
            menu="Tools",
            addon_name="other_addon",
        )

        count = registry.unregister_all_from_addon("my_addon")
        assert count == 2
        items = registry.get_items_for_menu("Tools")
        assert len(items) == 1
        assert items[0].item_id == "other.action"

    def test_items_sorted_by_priority(self):
        registry = MenuRegistry()
        registry.register(
            item_id="low",
            label="Low",
            action="win.low",
            menu="Tools",
            priority=200,
        )
        registry.register(
            item_id="high",
            label="High",
            action="win.high",
            menu="Tools",
            priority=50,
        )
        registry.register(
            item_id="medium",
            label="Medium",
            action="win.medium",
            menu="Tools",
            priority=100,
        )

        items = registry.get_items_for_menu("Tools")
        assert items[0].item_id == "high"
        assert items[1].item_id == "medium"
        assert items[2].item_id == "low"

    def test_get_items_for_nonexistent_menu(self):
        registry = MenuRegistry()
        items = registry.get_items_for_menu("Nonexistent")
        assert items == []

    def test_global_registry_exists(self):
        assert menu_registry is not None
