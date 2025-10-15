from typing import Any, Optional, Callable, Tuple
from .command import Command


class SetterCommand(Command):
    """
    A generic command to call a setter method with arbitrary arguments.
    """

    def __init__(
        self,
        target: Any,
        setter_method_name: str,
        new_args: Tuple[Any, ...],
        old_args: Tuple[Any, ...],
        on_change_callback: Optional[Callable[[], None]] = None,
        name: Optional[str] = None,
    ):
        super().__init__(name, on_change_callback)
        self.target = target
        self.setter_method_name = setter_method_name
        self.setter_method = getattr(self.target, self.setter_method_name)
        self.new_args = new_args
        self.old_args = old_args

    def execute(self) -> None:
        """Executes the setter with the new arguments."""
        self.setter_method(*self.new_args)
        if self.on_change_callback:
            self.on_change_callback()

    def undo(self) -> None:
        """Executes the setter with the old arguments to revert."""
        self.setter_method(*self.old_args)
        if self.on_change_callback:
            self.on_change_callback()

    def can_coalesce_with(self, next_command: Command) -> bool:
        return (
            isinstance(next_command, SetterCommand)
            and self.target is next_command.target
            and self.setter_method_name == next_command.setter_method_name
        )

    def coalesce_with(self, next_command: Command) -> bool:
        """
        Merges another SetterCommand if it affects the same object and
        method.
        """
        if not self.can_coalesce_with(next_command):
            return False

        # The new arguments become the value from the incoming command.
        self.new_args = next_command.new_args  # type: ignore
        # The timestamp is updated to the newer command's time.
        self.timestamp = next_command.timestamp
        return True
