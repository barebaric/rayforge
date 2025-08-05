from typing import Any, Optional, Callable
from .command import Command


class ChangePropertyCommand(Command):
    """A command to change a single property on an object."""

    def __init__(
        self,
        target: Any,
        property_name: str,
        new_value: Any,
        setter_method_name: Optional[str] = None,
        on_change_callback: Optional[Callable[[], None]] = None,
        name: Optional[str] = None,
    ):
        super().__init__(name, on_change_callback)
        self.target = target
        self.property_name = property_name
        self.new_value = new_value
        self.setter_method_name = setter_method_name
        self.old_value = getattr(self.target, self.property_name)

    def _set_property(self, value: Any) -> None:
        if self.setter_method_name:
            setter_func = getattr(self.target, self.setter_method_name)
            setter_func(value)
        else:
            setattr(self.target, self.property_name, value)

    def execute(self) -> None:
        self._set_property(self.new_value)
        if self.on_change_callback:
            self.on_change_callback()

    def undo(self) -> None:
        self._set_property(self.old_value)
        if self.on_change_callback:
            self.on_change_callback()

    def can_coalesce_with(self, next_command: Command) -> bool:
        return (
            isinstance(next_command, ChangePropertyCommand) and
            self.target is next_command.target and
            self.property_name == next_command.property_name
        )

    def coalesce_with(self, next_command: Command) -> bool:
        """
        Merges another ChangePropertyCommand if it affects the same
        property.
        """
        if not self.can_coalesce_with(next_command):
            return False

        self.new_value = next_command.new_value  # type: ignore
        self.timestamp = next_command.timestamp
        return True
