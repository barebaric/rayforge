import logging
from typing import Dict, Optional, Set

from gi.repository import Gio, Gtk


logger = logging.getLogger(__name__)


class ActionRegistry:
    """
    Registry for window actions registered by addons.

    Tracks which actions belong to which addon so they can be
    properly removed when an addon is disabled.
    """

    def __init__(self):
        self._actions: Dict[str, str] = {}
        self._addon_actions: Dict[str, Set[str]] = {}
        self._window: Optional[Gtk.ApplicationWindow] = None

    def set_window(self, window: Gtk.ApplicationWindow) -> None:
        """Set the window to which actions will be added."""
        self._window = window

    def register(
        self,
        action_name: str,
        action: Gio.SimpleAction,
        addon_name: Optional[str] = None,
    ) -> None:
        """
        Register an action with the window and track it by addon.

        Args:
            action_name: Name of the action (without 'win.' prefix).
            action: The Gio.SimpleAction instance.
            addon_name: Name of the addon registering this action.
        """
        if self._window is None:
            logger.warning("Cannot register action: window not set")
            return

        full_name = action_name
        if full_name in self._actions:
            logger.warning(
                f"Action '{full_name}' already registered, replacing"
            )
            old_addon = self._actions[full_name]
            if old_addon and old_addon in self._addon_actions:
                self._addon_actions[old_addon].discard(full_name)

        self._window.add_action(action)
        self._actions[full_name] = addon_name or ""

        if addon_name:
            if addon_name not in self._addon_actions:
                self._addon_actions[addon_name] = set()
            self._addon_actions[addon_name].add(full_name)

        logger.debug(
            f"Registered action '{full_name}' for addon '{addon_name}'"
        )

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

        addon_name = self._actions[action_name]
        if addon_name and addon_name in self._addon_actions:
            self._addon_actions[addon_name].discard(action_name)

        del self._actions[action_name]
        logger.debug(f"Unregistered action '{action_name}'")
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
        return count


action_registry = ActionRegistry()
