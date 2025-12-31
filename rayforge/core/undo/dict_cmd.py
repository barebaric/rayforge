"""
Provides a command for changing a value within a dictionary.
"""

from typing import Dict, Any, Optional, Callable, cast
from .command import Command


class DictItemCommand(Command):
    """
    An undoable command that changes a value for a specific key in a
    dictionary.
    """

    def __init__(
        self,
        target_dict: Dict[str, Any],
        key: str,
        new_value: Any,
        name: str,
        on_change_callback: Optional[Callable[[], Any]] = None,
    ):
        """
        Initializes the command.

        Args:
            target_dict: The dictionary to modify.
            key: The key whose value will be changed.
            new_value: The new value to set for the key.
            name: The user-facing name for this command.
            on_change_callback: An optional function to call after the
                dictionary is modified.
        """
        super().__init__(name, on_change_callback)
        self.target_dict = target_dict
        self.key = key
        self.new_value = new_value
        self.old_value = self.target_dict.get(self.key)

    def execute(self) -> None:
        """Sets the new value in the dictionary."""
        self.target_dict[self.key] = self.new_value
        if self.on_change_callback:
            self.on_change_callback()

    def undo(self) -> None:
        """Restores the old value in the dictionary."""
        if (
            self.key in self.target_dict
            and self.target_dict[self.key] == self.new_value
        ):
            self.target_dict[self.key] = self.old_value
        elif self.old_value is not None:
            self.target_dict[self.key] = self.old_value
        else:  # old value was none, so the key didn't exist
            if self.key in self.target_dict:
                del self.target_dict[self.key]

        if self.on_change_callback:
            self.on_change_callback()

    def can_coalesce_with(self, next_command: Command) -> bool:
        return (
            isinstance(next_command, DictItemCommand)
            and self.target_dict is next_command.target_dict
            and self.key == next_command.key
        )

    def coalesce_with(self, next_command: Command) -> bool:
        """
        Merges another DictItemCommand if it affects the same dictionary key.
        """
        if not self.can_coalesce_with(next_command):
            return False

        # mypy check for next_command type is done in can_coalesce_with
        self.new_value = cast(DictItemCommand, next_command).new_value
        self.timestamp = next_command.timestamp
        return True
