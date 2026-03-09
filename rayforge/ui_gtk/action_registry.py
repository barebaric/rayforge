import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Set

from blinker import Signal
from gi.repository import Gio, Gtk


logger = logging.getLogger(__name__)


@dataclass
class MenuPlacement:
    menu_id: str = "tools"
    priority: int = 100


@dataclass
class ToolbarPlacement:
    group: str = "main"
    priority: int = 100


@dataclass
class ActionInfo:
    action: Gio.SimpleAction
    action_name: str
    label: Optional[str] = None
    icon_name: Optional[str] = None
    shortcut: Optional[str] = None
    addon_name: Optional[str] = None
    menu: Optional[MenuPlacement] = None
    toolbar: Optional[ToolbarPlacement] = None


class ActionRegistry:
    """
    Registry for window actions with optional menu and toolbar placement.

    Tracks which actions belong to which addon so they can be
    properly removed when an addon is disabled.
    """

    def __init__(self):
        self._actions: Dict[str, ActionInfo] = {}
        self._addon_actions: Dict[str, Set[str]] = {}
        self._window: Optional[Gtk.ApplicationWindow] = None
        self.changed = Signal()

    def set_window(self, window: Gtk.ApplicationWindow) -> None:
        """Set the window to which actions will be added."""
        self._window = window

    @property
    def window(self) -> Optional[Gtk.ApplicationWindow]:
        """Get the window associated with this registry."""
        return self._window

    def register(
        self,
        action_name: str,
        action: Gio.SimpleAction,
        addon_name: Optional[str] = None,
        label: Optional[str] = None,
        icon_name: Optional[str] = None,
        shortcut: Optional[str] = None,
        menu: Optional[MenuPlacement] = None,
        toolbar: Optional[ToolbarPlacement] = None,
    ) -> None:
        """
        Register an action with the window and track it by addon.

        Args:
            action_name: Name of the action (without 'win.' prefix).
            action: The Gio.SimpleAction instance.
            addon_name: Name of the addon registering this action.
            label: Display label for UI (menu, toolbar).
            icon_name: Icon name for toolbar items.
            shortcut: Keyboard shortcut (e.g., "<Ctrl><Alt>p").
            menu: Menu placement info if this action should appear in a menu.
            toolbar: Toolbar placement info if this action should appear in
                the toolbar.
        """
        if self._window is None:
            logger.warning("Cannot register action: window not set")
            return

        if action_name in self._actions:
            logger.warning(
                f"Action '{action_name}' already registered, replacing"
            )
            old_addon = self._actions[action_name].addon_name
            if old_addon and old_addon in self._addon_actions:
                self._addon_actions[old_addon].discard(action_name)

        self._window.add_action(action)

        info = ActionInfo(
            action=action,
            action_name=action_name,
            label=label,
            icon_name=icon_name,
            shortcut=shortcut,
            addon_name=addon_name or "",
            menu=menu,
            toolbar=toolbar,
        )
        self._actions[action_name] = info

        if addon_name:
            if addon_name not in self._addon_actions:
                self._addon_actions[addon_name] = set()
            self._addon_actions[addon_name].add(action_name)

        logger.debug(
            f"Registered action '{action_name}' for addon '{addon_name}'"
        )
        self.changed.send(self)

    def unregister(self, action_name: str) -> bool:
        """
        Unregister an action from the window.

        Args:
            action_name: The name of the action to unregister.

        Returns:
            True if the action was unregistered, False if not found.
        """
        if action_name not in self._actions:
            return False

        if self._window is not None:
            self._window.remove_action(action_name)

        info = self._actions[action_name]
        if info.addon_name and info.addon_name in self._addon_actions:
            self._addon_actions[info.addon_name].discard(action_name)

        del self._actions[action_name]
        logger.debug(f"Unregistered action '{action_name}'")
        self.changed.send(self)
        return True

    def unregister_all_from_addon(self, addon_name: str) -> int:
        """
        Unregister all actions registered by a specific addon.

        Args:
            addon_name: The name of the addon.

        Returns:
            The number of actions unregistered.
        """
        if addon_name not in self._addon_actions:
            return 0

        action_names = self._addon_actions.pop(addon_name)
        count = 0
        for action_name in action_names:
            if action_name in self._actions:
                if self._window is not None:
                    self._window.remove_action(action_name)
                del self._actions[action_name]
                count += 1
        logger.debug(f"Unregistered {count} actions from addon '{addon_name}'")
        if count > 0:
            self.changed.send(self)
        return count

    def get(self, action_name: str) -> Optional[ActionInfo]:
        """
        Get action info by name.

        Args:
            action_name: The name of the action.

        Returns:
            The ActionInfo object, or None if not found.
        """
        return self._actions.get(action_name)

    def get_menu_items(self, menu_id: str) -> List[ActionInfo]:
        """
        Get actions with menu placement for a specific menu, sorted by
        priority.

        Args:
            menu_id: The menu identifier (e.g., 'tools', 'arrange').

        Returns:
            List of ActionInfo objects sorted by priority.
        """
        items = [
            info
            for info in self._actions.values()
            if info.menu and info.menu.menu_id == menu_id and info.label
        ]
        return sorted(items, key=lambda x: x.menu.priority if x.menu else 100)

    def get_toolbar_items(self, group: str) -> List[ActionInfo]:
        """
        Get actions with toolbar placement for a specific group, sorted by
        priority.

        Args:
            group: The toolbar group identifier (e.g., 'main', 'arrange').

        Returns:
            List of ActionInfo objects sorted by priority.
        """
        items = [
            info
            for info in self._actions.values()
            if info.toolbar and info.toolbar.group == group
        ]
        return sorted(
            items, key=lambda x: x.toolbar.priority if x.toolbar else 100
        )

    def get_all_with_shortcuts(self) -> List[ActionInfo]:
        """
        Get all actions that have keyboard shortcuts defined.

        Returns:
            List of ActionInfo objects with shortcuts.
        """
        return [info for info in self._actions.values() if info.shortcut]


action_registry = ActionRegistry()
