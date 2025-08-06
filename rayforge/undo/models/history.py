from __future__ import annotations
from typing import List, Optional, Iterator
from blinker import Signal
from contextlib import contextmanager
from .command import Command
from .composite_cmd import CompositeCommand

# Maximum time in seconds between two commands to be considered for coalescing.
COALESCE_THRESHOLD = 0.5


class _TransactionContextProxy:
    """
    A helper object yielded by the HistoryManager's transaction context
    manager.
    It proxies execute/add calls to the manager, ensuring they are handled
    within the current transaction.
    """

    def __init__(self, manager: HistoryManager):
        self._manager = manager

    def set_label(self, name: str) -> None:
        """Sets the display name for the transaction (e.g., for the UI)."""
        self._manager.transaction_name = name

    def execute(self, command: Command) -> None:
        """Executes a command and adds it to the transaction."""
        self._manager.execute(command)

    def add(self, command: Command) -> None:
        """Adds a command that has already been executed to the transaction."""
        self._manager.add(command)


class HistoryManager:
    """
    Manages the undo/redo history using a transactional command pattern.
    Supports both explicit transactions (for multi-part actions) and
    automatic coalescing (for rapid, identical actions).
    """

    def __init__(self):
        self.undo_stack: List[Command] = []
        self.redo_stack: List[Command] = []
        self.changed = Signal()

        # State for explicit, manual transactions
        self.in_transaction: bool = False
        self.transaction_commands: List[Command] = []
        self.transaction_name: str = ""

    def execute(self, command: Command):
        """
        Executes a command and adds it to the history, possibly coalescing
        it with the previous command.
        """
        command.execute()
        self.add(command)

    def add(self, command: Command):
        """
        Adds a command that has already been executed to the history,
        possibly coalescing it with the previous command.
        """
        if self.in_transaction:
            self.transaction_commands.append(command)
            return
        self._add_to_history(command)

    def _add_to_history(self, command: Command):
        """
        Adds a command to the undo stack, handling the coalescing logic.
        This is the single entry point for a command to be placed on the
        undo stack.
        """
        last_command = self.undo_stack[-1] if self.undo_stack else None

        if last_command:
            time_delta = command.timestamp - last_command.timestamp
            # Try to coalesce if the new command is similar and recent.
            if time_delta < COALESCE_THRESHOLD and last_command.coalesce_with(
                command
            ):
                # The last command was successfully updated.
                self.changed.send(self, command=last_command)
                return

        # If we couldn't coalesce, add the new command to the stack.
        self.undo_stack.append(command)
        self.redo_stack.clear()
        self.changed.send(self, command=command)

    @contextmanager
    def transaction(
        self, name: str = "Transaction"
    ) -> Iterator[_TransactionContextProxy]:
        """
        Provides a context manager for grouping commands into a single
        transaction.

        If the transaction completes successfully, the commands are grouped
        into a single history entry. If only one command is executed, it is
        "unwrapped" and added directly. Otherwise, commands are bundled into
        a CompositeCommand.
        The transaction's name will be applied to the final command.

        If an exception occurs, all commands executed within the transaction
        are undone, and the transaction is aborted.

        Usage:
            with history_manager.transaction("My Changes") as t:
                # t.set_label("A better name") is also possible
                t.execute(SetterCommand(...))
        """
        self.begin_transaction(name)
        try:
            yield _TransactionContextProxy(self)
            self.end_transaction()
        except Exception:
            # An exception occurred. Undo any commands that were executed.
            for cmd in reversed(self.transaction_commands):
                try:
                    cmd.undo()
                except Exception:
                    # Best effort: log this secondary error. For now, we
                    # continue.
                    pass
            self.abort_transaction()
            # The state has changed due to the undos, so we signal.
            self.changed.send(self, command=None)
            raise  # Re-raise the original exception

    def begin_transaction(self, name: str = "Transaction"):
        """
        Starts an explicit transaction. All subsequent commands executed will
        be grouped together until end_transaction() is called.
        """
        if self.in_transaction:
            # Nested transactions are not supported; raise an error to prevent
            # unexpected behavior.
            raise RuntimeError(
                "Cannot start a new transaction while another is already"
                " active."
            )

        self.in_transaction = True
        self.transaction_commands = []
        self.transaction_name = name

    def end_transaction(self):
        """
        Ends the current transaction, creates a CompositeCommand, and adds
        it to the history, allowing it to be coalesced.
        """
        if not self.in_transaction:
            return

        self.in_transaction = False
        if not self.transaction_commands:
            return

        # If only one command is in the transaction, it gets "unwrapped".
        # Otherwise, they are bundled into a CompositeCommand.
        final_command = self._coalesce_commands(self.transaction_commands)

        if final_command:
            final_command.name = self.transaction_name
            # Add the composite/unwrapped command to history via the proper
            # channel.
            self._add_to_history(final_command)

    def abort_transaction(self):
        """
        Aborts the current transaction, discarding any commands that were
        added since it began. NOTE: This does not undo the commands itself,
        as that is handled by the context manager's exception block.
        """
        if not self.in_transaction:
            return
        self.in_transaction = False
        self.transaction_commands = []
        self.transaction_name = ""

    def _coalesce_commands(self, commands: List[Command]) -> Optional[Command]:
        """
        Internal helper to optimize a list of commands from an explicit
        transaction. If there's only one command, it returns it directly.
        Otherwise, it wraps them in a CompositeCommand.
        """
        if not commands:
            return None
        if len(commands) == 1:
            return commands[0]

        # Unlike automatic coalescing, here we group different commands
        # into a single CompositeCommand.
        return CompositeCommand(commands, self.transaction_name)

    def undo(self):
        """Undoes the last action."""
        if not self.can_undo():
            return
        command = self.undo_stack.pop()
        command.undo()
        self.redo_stack.append(command)
        self.changed.send(self, command=command)

    def redo(self):
        """Redoes the last undone action."""
        if not self.can_redo():
            return
        command = self.redo_stack.pop()
        command.execute()
        self.undo_stack.append(command)
        self.changed.send(self, command=command)

    def undo_to(self, target_command: Command):
        """Undoes all actions up to and including the target command."""
        while self.can_undo():
            command_to_undo = self.undo_stack[-1]
            self.undo()
            if command_to_undo is target_command:
                break

    def redo_to(self, target_command: Command):
        """Redoes all actions up to and including the target command."""
        while self.can_redo():
            command_to_redo = self.redo_stack[-1]
            self.redo()
            if command_to_redo is target_command:
                break

    def can_undo(self) -> bool:
        """Returns True if there are actions to undo."""
        return bool(self.undo_stack)

    def can_redo(self) -> bool:
        """Returns True if there are actions to redo."""
        return bool(self.redo_stack)

    def clear(self):
        """Clears all undo and redo history."""
        self.undo_stack.clear()
        self.redo_stack.clear()
        self.in_transaction = False
        self.transaction_commands.clear()
        self.changed.send(self, command=None)
