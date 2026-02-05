"""Progress context abstraction for the compute layer.

This module provides a testable interface for progress reporting and
cancellation checking in the compute layer, separate from the complex
ExecutionContext used in the stage layer.
"""

from abc import ABC, abstractmethod
from typing import Callable, Optional


class ProgressContext(ABC):
    """Abstract base class for progress reporting in the compute layer.

    This interface provides the same capabilities as ExecutionContext
    but is designed for testability - no threading, no queues, no
    complex state management.
    """

    @abstractmethod
    def is_cancelled(self) -> bool:
        """Check if the operation has been cancelled."""
        pass

    @abstractmethod
    def set_progress(self, progress: float) -> None:
        """Set progress as an absolute value.

        The value is normalized against the context's total.
        Example: If total=200, set_progress(20) reports 0.1 progress.
        """
        pass

    @abstractmethod
    def set_message(self, message: str) -> None:
        """Set a descriptive status message."""
        pass

    @abstractmethod
    def set_total(self, total: float) -> None:
        """Set the total value for progress normalization.

        If total <= 0, it's treated as 1.0 (already normalized).
        """
        pass

    @abstractmethod
    def sub_context(
        self,
        base_progress: float,
        progress_range: float,
        total: float = 1.0,
    ) -> "ProgressContext":
        """Create a sub-context for hierarchical progress reporting.

        Args:
            base_progress: The normalized (0.0-1.0) progress in the parent
                           when the sub-task begins.
            progress_range: The fraction (0.0-1.0) of the parent's progress
                           that this sub-task represents.
            total: The total number of steps for the new sub-context.
                   Defaults to 1.0 (already normalized).

        Returns:
            A new ProgressContext configured as a sub-context.
        """
        pass

    @abstractmethod
    def flush(self) -> None:
        """Immediately send any pending updates."""
        pass


class CallbackProgressContext(ProgressContext):
    """ProgressContext implementation that uses callbacks for reporting.

    This is used in production where the runner layer adapts
    ExecutionContext to this interface.
    """

    def __init__(
        self,
        is_cancelled_func: Callable[[], bool],
        progress_callback: Optional[Callable[[float], None]] = None,
        message_callback: Optional[Callable[[str], None]] = None,
        _parent: Optional["CallbackProgressContext"] = None,
        _base_progress: float = 0.0,
        _progress_range: float = 1.0,
    ):
        self._is_cancelled_func = is_cancelled_func
        self._progress_callback = progress_callback
        self._message_callback = message_callback
        self._parent = _parent
        self._base = _base_progress
        self._range = _progress_range
        self._total = 1.0

    def is_cancelled(self) -> bool:
        return self._is_cancelled_func()

    def set_total(self, total: float) -> None:
        if total <= 0:
            self._total = 1.0
        else:
            self._total = float(total)

    def set_progress(self, progress: float) -> None:
        if self._progress_callback:
            normalized = progress / self._total
            global_progress = self._base + (normalized * self._range)
            self._progress_callback(global_progress)

    def set_message(self, message: str) -> None:
        if self._message_callback:
            self._message_callback(message)

    def sub_context(
        self,
        base_progress: float,
        progress_range: float,
        total: float = 1.0,
    ) -> "CallbackProgressContext":
        new_base = self._base + (base_progress * self._range)
        new_range = self._range * progress_range
        return CallbackProgressContext(
            is_cancelled_func=self._is_cancelled_func,
            progress_callback=self._progress_callback,
            message_callback=self._message_callback,
            _parent=self,
            _base_progress=new_base,
            _progress_range=new_range,
        )

    def flush(self) -> None:
        pass


class NoOpProgressContext(ProgressContext):
    """No-op ProgressContext for cases where progress is not needed.

    Useful for simple operations or when running in non-interactive mode.
    """

    def is_cancelled(self) -> bool:
        return False

    def set_progress(self, progress: float) -> None:
        pass

    def set_message(self, message: str) -> None:
        pass

    def set_total(self, total: float) -> None:
        pass

    def sub_context(
        self,
        base_progress: float,
        progress_range: float,
        total: float = 1.0,
    ) -> "ProgressContext":
        return self

    def flush(self) -> None:
        pass
