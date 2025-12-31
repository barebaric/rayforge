from __future__ import annotations
import time
from abc import ABC, abstractmethod
from typing import Optional, Callable


class Command(ABC):
    """
    Abstract base class for an undoable action.
    """

    def __init__(
        self,
        name: Optional[str] = None,
        on_change_callback: Optional[Callable[[], None]] = None,
    ):
        self.name = name
        self.on_change_callback = on_change_callback
        self.timestamp: float = time.time()

    @abstractmethod
    def execute(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def undo(self) -> None:
        raise NotImplementedError

    def can_coalesce_with(self, next_command: Command) -> bool:
        """
        Checks if the 'next_command' can be merged into this one without
        modifying the state of either command.

        Args:
            next_command: The incoming command to check.

        Returns: True if merging is possible, False otherwise.
        """
        return False

    def coalesce_with(self, next_command: Command) -> bool:
        """
        Attempts to merge the 'next_command' into this one. If successful,
        this command's state is updated with the newer command's state,
        and it returns True. Otherwise, it returns False.

        Args:
            next_command: The incoming command to potentially merge.

        Returns:
            True if merging was successful, False otherwise.
        """
        return False
