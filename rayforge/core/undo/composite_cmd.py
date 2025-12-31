from typing import List, Optional, Callable
from .command import Command


class CompositeCommand(Command):
    """
    A command that groups several other commands into a single transaction.
    """

    def __init__(
        self,
        commands: List[Command],
        name: str,
        on_change_callback: Optional[Callable[[], None]] = None,
    ):
        super().__init__(name, on_change_callback)
        self.commands = commands

    def execute(self) -> None:
        """Executes all child commands in order."""
        for cmd in self.commands:
            cmd.execute()
        if self.on_change_callback:
            self.on_change_callback()

    def undo(self) -> None:
        """Undoes all child commands in reverse order."""
        for cmd in reversed(self.commands):
            cmd.undo()
        if self.on_change_callback:
            self.on_change_callback()

    def can_coalesce_with(self, next_command: Command) -> bool:
        if not isinstance(next_command, CompositeCommand):
            return False

        # Both composites must have the same number of child commands.
        if len(self.commands) != len(next_command.commands):
            return False

        # Check if every child command can coalesce with its counterpart.
        for i, cmd in enumerate(self.commands):
            if not cmd.can_coalesce_with(next_command.commands[i]):
                return False

        return True

    def coalesce_with(self, next_command: Command) -> bool:
        """
        Merges another CompositeCommand if all their respective child
        commands can be coalesced.
        """
        if not self.can_coalesce_with(next_command):
            return False

        # Since we know it's possible, now perform the merge.
        for i, cmd in enumerate(self.commands):
            cmd.coalesce_with(next_command.commands[i])  # type: ignore

        self.timestamp = next_command.timestamp
        return True
