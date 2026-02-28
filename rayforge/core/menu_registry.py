import logging
from typing import Dict, List, Optional, Set

logger = logging.getLogger(__name__)


class MenuItem:
    """Represents a menu item."""

    def __init__(
        self,
        item_id: str,
        label: str,
        action: str,
        menu: str = "Tools",
        priority: int = 100,
        addon_name: Optional[str] = None,
    ):
        self.item_id = item_id
        self.label = label
        self.action = action
        self.menu = menu
        self.priority = priority
        self.addon_name = addon_name

    def __repr__(self):
        return f"MenuItem({self.item_id}, {self.menu}/{self.label})"


class MenuRegistry:
    """
    Registry for menu items.

    Allows plugins to register menu items that will be added to
    the application's menu bar.
    """

    def __init__(self):
        self._items: Dict[str, MenuItem] = {}
        self._addon_items: Dict[str, Set[str]] = {}
        self._menu_sections: Dict[str, str] = {}

    def register(
        self,
        item_id: str,
        label: str,
        action: str,
        menu: str = "Tools",
        priority: int = 100,
        addon_name: Optional[str] = None,
    ) -> None:
        """
        Register a menu item.

        Args:
            item_id: Unique identifier for the menu item.
            label: Display label for the menu item.
            action: Action name (e.g., 'win.my_action').
            menu: Menu to add the item to (e.g., 'Tools', 'File', 'Edit').
            priority: Lower values appear first. Default 100.
            addon_name: Name of the addon registering this item.
        """
        if item_id in self._items:
            logger.warning(
                f"Menu item '{item_id}' already registered, overwriting"
            )
            if addon_name:
                old_addon = self._items[item_id].addon_name
                if old_addon and old_addon in self._addon_items:
                    self._addon_items[old_addon].discard(item_id)

        item = MenuItem(
            item_id=item_id,
            label=label,
            action=action,
            menu=menu,
            priority=priority,
            addon_name=addon_name,
        )
        self._items[item_id] = item

        if addon_name:
            if addon_name not in self._addon_items:
                self._addon_items[addon_name] = set()
            self._addon_items[addon_name].add(item_id)

        logger.debug(f"Registered menu item '{item_id}' in menu '{menu}'")

    def unregister(self, item_id: str) -> bool:
        """
        Unregister a menu item.

        Args:
            item_id: The ID of the menu item to unregister.

        Returns:
            True if the item was unregistered, False if not found.
        """
        if item_id not in self._items:
            return False

        item = self._items[item_id]
        if item.addon_name and item.addon_name in self._addon_items:
            self._addon_items[item.addon_name].discard(item_id)

        del self._items[item_id]
        logger.debug(f"Unregistered menu item '{item_id}'")
        return True

    def unregister_all_from_addon(self, addon_name: str) -> int:
        """
        Unregister all menu items registered by a specific addon.

        Args:
            addon_name: The name of the addon.

        Returns:
            The number of items unregistered.
        """
        if addon_name not in self._addon_items:
            return 0

        item_ids = self._addon_items.pop(addon_name)
        count = 0
        for item_id in item_ids:
            if item_id in self._items:
                del self._items[item_id]
                count += 1
        logger.debug(
            f"Unregistered {count} menu items from addon '{addon_name}'"
        )
        return count

    def get_items_for_menu(self, menu: str) -> List[MenuItem]:
        """
        Get all menu items for a specific menu, sorted by priority.

        Args:
            menu: The menu name (e.g., 'Tools', 'File').

        Returns:
            List of MenuItem objects sorted by priority.
        """
        items = [item for item in self._items.values() if item.menu == menu]
        return sorted(items, key=lambda x: x.priority)

    def get_all_items(self) -> Dict[str, MenuItem]:
        """
        Return a copy of all registered menu items.

        Returns:
            Dictionary mapping item IDs to MenuItem objects.
        """
        return self._items.copy()

    def register_menu_section(self, menu: str, section_id: str) -> None:
        """
        Register a section ID for a menu where addon items should be added.

        Args:
            menu: The menu name.
            section_id: The section identifier within that menu.
        """
        self._menu_sections[menu] = section_id

    def get_menu_section(self, menu: str) -> Optional[str]:
        """
        Get the section ID for a menu where addon items should be added.

        Args:
            menu: The menu name.

        Returns:
            The section ID, or None if not set.
        """
        return self._menu_sections.get(menu)


menu_registry = MenuRegistry()
