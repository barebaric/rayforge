from typing import Dict, Optional, Set, Type


class CommandRegistry:
    """
    Registry for editor command classes.

    Allows addons to register command handlers that extend the
    DocEditor functionality. Commands are registered by name and
    instantiated with the editor instance.
    """

    def __init__(self):
        self._command_classes: Dict[str, Type] = {}
        self._addon_items: Dict[str, Set[str]] = {}

    def register(
        self,
        command_name: str,
        command_class: Type,
        addon_name: Optional[str] = None,
    ) -> None:
        """
        Register a command class.

        Args:
            command_name: The name to use for this command.
            command_class: The command class (takes DocEditor in __init__).
            addon_name: Optional name of addon registering this command.
        """
        self._command_classes[command_name] = command_class
        if addon_name:
            if addon_name not in self._addon_items:
                self._addon_items[addon_name] = set()
            self._addon_items[addon_name].add(command_name)

    def unregister(self, command_name: str) -> bool:
        """
        Unregister a command class by name.

        Args:
            command_name: The name of command to unregister.

        Returns:
            True if command was unregistered, False if not found.
        """
        if command_name in self._command_classes:
            del self._command_classes[command_name]
            for addon_name, items in self._addon_items.items():
                items.discard(command_name)
            return True
        return False

    def unregister_all_from_addon(self, addon_name: str) -> int:
        """
        Unregister all commands registered by a specific addon.

        Args:
            addon_name: The name of addon.

        Returns:
            The number of commands unregistered.
        """
        if addon_name not in self._addon_items:
            return 0
        items = self._addon_items.pop(addon_name)
        count = 0
        for command_name in items:
            if command_name in self._command_classes:
                del self._command_classes[command_name]
                count += 1
        return count

    def get(self, command_name: str) -> Optional[Type]:
        """
        Look up a command class by name.

        Args:
            command_name: The name of command.

        Returns:
            The command class, or None if not found.
        """
        return self._command_classes.get(command_name)

    def all_commands(self) -> Dict[str, Type]:
        """
        Return a copy of all registered command classes.

        Returns:
            Dictionary mapping command names to command classes.
        """
        return self._command_classes.copy()


command_registry = CommandRegistry()
