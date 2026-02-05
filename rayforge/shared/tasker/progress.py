"""Progress context abstraction for task execution.

This module provides a unified hierarchy for progress reporting and
cancellation checking. The base class ProgressContext defines the
interface, with ThrottledProgressContext extending it for contexts
that require throttling or debouncing of progress updates.
"""

from abc import ABC, abstractmethod
from typing import Callable


class ProgressContext(ABC):
    """Abstract base class for progress reporting and cancellation.

    This class provides the foundation for all progress reporting
    contexts, including normalization, sub-contexting, and cancellation
    checking. Subclasses must implement the abstract methods to provide
    specific reporting mechanisms.
    """

    def __init__(
        self,
        base_progress: float = 0.0,
        progress_range: float = 1.0,
        total: float = 100.0,
    ):
        """Initialize the progress context.

        Args:
            base_progress: The normalized (0.0-1.0) progress when this
                           context begins relative to its parent.
            progress_range: The fraction (0.0-1.0) of the parent's
                           progress that this context represents.
            total: The total number of steps for progress normalization.
                   Defaults to 100.0.
        """
        self._base = base_progress
        self._range = progress_range
        self._total = 1.0
        self.set_total(total)

    @abstractmethod
    def is_cancelled(self) -> bool:
        """Check if the operation has been cancelled.

        Returns:
            True if the operation should be cancelled, False otherwise.
        """
        pass

    def set_progress(self, progress: float) -> None:
        """Set progress as an absolute value.

        The value is normalized based on the context's base, range,
        and total, then reported via _report_normalized_progress.

        Args:
            progress: The absolute progress value to set.
        """
        normalized_progress = progress / self._total
        self._report_normalized_progress(normalized_progress)

    @abstractmethod
    def set_message(self, message: str) -> None:
        """Set a descriptive status message.

        Args:
            message: The status message to display.
        """
        pass

    def set_total(self, total: float) -> None:
        """Set the total value for progress normalization.

        If total <= 0, it's treated as 1.0 (already normalized).

        Args:
            total: The total number of steps for progress calculation.
        """
        if total <= 0:
            self._total = 1.0
        else:
            self._total = float(total)

    def sub_context(
        self,
        base_progress: float,
        progress_range: float,
        total: float,
    ) -> "ProgressContext":
        """Create a sub-context for hierarchical progress reporting.

        Args:
            base_progress: The normalized (0.0-1.0) progress in the parent
                           when the sub-task begins.
            progress_range: The fraction (0.0-1.0) of the parent's
                           progress that this sub-task represents.
            total: The total number of steps for the new sub-context.

        Returns:
            A new ProgressContext configured as a sub-context.
        """
        return self._create_sub_context(base_progress, progress_range, total)

    @abstractmethod
    def flush(self) -> None:
        """Immediately send any pending updates."""
        pass

    @abstractmethod
    def _report_normalized_progress(self, progress: float) -> None:
        """Report a normalized (0.0-1.0) progress value.

        Subclasses must implement this to provide the specific reporting
        mechanism.

        Args:
            progress: The normalized progress value (0.0-1.0).
        """
        pass

    @abstractmethod
    def _create_sub_context(
        self,
        base_progress: float,
        progress_range: float,
        total: float,
    ) -> "ProgressContext":
        """Factory method for creating sub-contexts.

        Subclasses must implement this to return an appropriate sub-context
        instance.

        Args:
            base_progress: The normalized (0.0-1.0) progress in the parent
                           when the sub-task begins.
            progress_range: The fraction (0.0-1.0) of the parent's
                           progress that this sub-task represents.
            total: The total number of steps for the new sub-context.

        Returns:
            A new ProgressContext configured as a sub-context.
        """
        pass


class ThrottledProgressContext(ProgressContext):
    """Abstract marker class for contexts requiring throttling.

    This class extends ProgressContext but serves as a marker for
    contexts that need progress updates to be throttled or debounced.
    All abstract methods remain abstract, requiring concrete subclasses
    to implement them.
    """

    pass


class NoOpProgressContext(ProgressContext):
    """No-op implementation of ProgressContext.

    This class provides a silent implementation that does nothing for all
    operations. Useful for testing or when progress reporting is not
    needed.
    """

    def __init__(
        self,
        base_progress: float = 0.0,
        progress_range: float = 1.0,
        total: float = 100.0,
    ):
        """Initialize the no-op progress context."""
        super().__init__(base_progress, progress_range, total)

    def is_cancelled(self) -> bool:
        """Check if the operation has been cancelled.

        Returns:
            Always False for no-op context.
        """
        return False

    def set_progress(self, progress: float) -> None:
        """Set progress as an absolute value (no-op).

        Args:
            progress: The absolute progress value (ignored).
        """
        pass

    def set_message(self, message: str) -> None:
        """Set a descriptive status message (no-op).

        Args:
            message: The status message (ignored).
        """
        pass

    def set_total(self, total: float) -> None:
        """Set the total value for progress normalization (no-op).

        Args:
            total: The total number of steps (ignored).
        """
        pass

    def sub_context(
        self,
        base_progress: float,
        progress_range: float,
        total: float,
    ) -> "ProgressContext":
        """Create a sub-context (returns new NoOpProgressContext).

        Args:
            base_progress: The normalized (0.0-1.0) progress in the parent
                           when the sub-task begins (ignored).
            progress_range: The fraction (0.0-1.0) of the parent's
                           progress that this sub-task represents
                           (ignored).
            total: The total number of steps for the new sub-context
                   (ignored).

        Returns:
            A new NoOpProgressContext instance.
        """
        return NoOpProgressContext(base_progress, progress_range, total)

    def flush(self) -> None:
        """Immediately send any pending updates (no-op)."""
        pass

    def _report_normalized_progress(self, progress: float) -> None:
        """Report a normalized progress value (no-op).

        Args:
            progress: The normalized progress value (ignored).
        """
        pass

    def _create_sub_context(
        self,
        base_progress: float,
        progress_range: float,
        total: float,
    ) -> "ProgressContext":
        """Factory method for creating sub-contexts.

        Args:
            base_progress: The normalized (0.0-1.0) progress in the parent
                           when the sub-task begins.
            progress_range: The fraction (0.0-1.0) of the parent's
                           progress that this sub-task represents.
            total: The total number of steps for the new sub-context.

        Returns:
            A new NoOpProgressContext instance.
        """
        return NoOpProgressContext(base_progress, progress_range, total)


class CallbackProgressContext(ProgressContext):
    """ProgressContext implementation that uses callbacks for reporting.

    This class provides a concrete implementation that delegates progress
    reporting, message setting, and cancellation checking to provided
    callback functions.
    """

    def __init__(
        self,
        is_cancelled_func: Callable[[], bool],
        progress_callback: Callable[[float], None],
        message_callback: Callable[[str], None],
        base_progress: float = 0.0,
        progress_range: float = 1.0,
        total: float = 100.0,
    ):
        """Initialize the callback progress context.

        Args:
            is_cancelled_func: Function that returns True if cancelled.
            progress_callback: Function called with normalized progress.
            message_callback: Function called with status messages.
            base_progress: The normalized (0.0-1.0) progress when this
                           context begins relative to its parent.
            progress_range: The fraction (0.0-1.0) of the parent's
                           progress that this context represents.
            total: The total number of steps for progress normalization.
        """
        super().__init__(base_progress, progress_range, total)
        self._is_cancelled_func = is_cancelled_func
        self._progress_callback = progress_callback
        self._message_callback = message_callback

    def is_cancelled(self) -> bool:
        """Check if the operation has been cancelled.

        Returns:
            The result of calling the is_cancelled_func callback.
        """
        return self._is_cancelled_func()

    def set_message(self, message: str) -> None:
        """Set a descriptive status message.

        Args:
            message: The status message to send via callback.
        """
        self._message_callback(message)

    def flush(self) -> None:
        """Immediately send any pending updates (no-op for callbacks)."""
        pass

    def _report_normalized_progress(self, progress: float) -> None:
        """Report a normalized progress value via callback.

        Args:
            progress: The normalized progress value (0.0-1.0).
        """
        scaled_progress = self._base + (progress * self._range)
        self._progress_callback(scaled_progress)

    def _create_sub_context(
        self,
        base_progress: float,
        progress_range: float,
        total: float,
    ) -> "ProgressContext":
        """Factory method for creating sub-contexts.

        Args:
            base_progress: The normalized (0.0-1.0) progress in the parent
                           when the sub-task begins.
            progress_range: The fraction (0.0-1.0) of the parent's
                           progress that this sub-task represents.
            total: The total number of steps for the new sub-context.

        Returns:
            A new CallbackProgressContext instance with the same callbacks.
        """
        return CallbackProgressContext(
            self._is_cancelled_func,
            self._progress_callback,
            self._message_callback,
            base_progress,
            progress_range,
            total,
        )
