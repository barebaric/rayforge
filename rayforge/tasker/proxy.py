"""
This module defines the base class for execution contexts and a proxy
for reporting progress from subprocesses. It provides a framework for
managing task execution, including progress reporting, message handling,
and sub-contexting.

WARNING: This file MUST NOT have any imports that cause any other
parts of the application to be initialized. It is designed to
be used during subprocess bootstrapping, where no other parts
of the application should be imported or initialized.
We can also not import any GTK or Adw classes here,
as this would cause the GTK main loop to be initialized,
which is not safe during bootstrapping.
In other words, we cannot use GLib.idle_add or similar.
"""

import abc
import logging
from queue import Full
from multiprocessing.queues import Queue


class BaseExecutionContext(abc.ABC):
    """
    Abstract base class for execution contexts.

    Provides common functionality for progress reporting, including
    normalization and sub-contexting. Subclasses must implement the
    specific reporting mechanism (e.g., via a queue or a debounced
    callback).
    """

    def __init__(
        self,
        base_progress: float = 0.0,
        progress_range: float = 1.0,
        total: float = 1.0,
    ):
        self._base = base_progress
        self._range = progress_range
        self._total = 1.0  # Default total for normalization
        self.set_total(total)
        self.task = None  # Add task attribute

    @abc.abstractmethod
    def _report_normalized_progress(self, progress: float):
        """
        Abstract method for handling a 0.0-1.0 progress value.
        Subclasses must implement this to either send the progress to a
        queue or schedule a debounced update.
        """
        pass

    def set_total(self, total: float):
        """
        Sets or updates the total value for this context's progress
        calculations.
        """
        if total <= 0:
            self._total = 1.0
        else:
            self._total = float(total)

    def set_progress(self, progress: float):
        """
        Sets the progress as an absolute value. This value is automatically
        normalized against the context's total.
        Example: If total=200, set_progress(20) reports 0.1 progress.
        """
        # The code calling this might already be sending normalized progress
        # if it doesn't call set_total. In that case, total is 1.0, and this
        # works.
        normalized_progress = progress / self._total
        self._report_normalized_progress(normalized_progress)

    @abc.abstractmethod
    def set_message(self, message: str):
        """Sets a descriptive message."""
        pass

    def sub_context(
        self,
        base_progress: float,
        progress_range: float,
        total: float = 1.0,
        **kwargs,
    ) -> "BaseExecutionContext":
        """
        Creates a sub-context that reports progress within a specified
        range of this context's progress.

        Args:
            base_progress: The normalized (0.0-1.0) progress in the parent
                           when the sub-task begins.
            progress_range: The fraction (0.0-1.0) of the parent's progress
                           that this sub-task represents.
            total: The total number of steps for the new sub-context.
                   Defaults to 1.0, treating progress as already normalized.
            **kwargs: Additional arguments for specific subclass constructors
                      (e.g., `check_cancelled` for ExecutionContext).

        Returns:
            A new execution context instance configured as a sub-context.
        """
        new_base = self._base + (base_progress * self._range)
        new_range = self._range * progress_range
        return self._create_sub_context(new_base, new_range, total, **kwargs)

    @abc.abstractmethod
    def _create_sub_context(
        self,
        base_progress: float,
        progress_range: float,
        total: float,
        **kwargs,
    ) -> "BaseExecutionContext":
        """
        Abstract factory method for creating a sub-context of the
        correct type.
        """
        pass

    @abc.abstractmethod
    def is_cancelled(self) -> bool:
        """Checks if the operation has been cancelled."""
        pass

    @abc.abstractmethod
    def flush(self):
        """
        Immediately sends any pending updates.
        """
        pass


class ExecutionContextProxy(BaseExecutionContext):
    """
    A pickleable proxy for reporting progress from a subprocess via a queue.
    """

    def __init__(
        self,
        progress_queue: Queue,
        base_progress=0.0,
        progress_range=1.0,
        parent_log_level: int = logging.DEBUG,
    ):
        super().__init__(base_progress, progress_range, total=1.0)
        self._queue = progress_queue
        self.parent_log_level = parent_log_level

    def _report_normalized_progress(self, progress: float):
        """
        Reports a 0.0-1.0 progress value, scaled to the proxy's
        range.
        """
        # Clamp to a valid range before scaling
        progress = max(0.0, min(1.0, progress))
        scaled_progress = self._base + (progress * self._range)
        try:
            self._queue.put_nowait(("progress", scaled_progress))
        except Full:
            pass  # If the queue is full, we drop the update.

    def set_message(self, message: str):
        try:
            self._queue.put_nowait(("message", message))
        except Full:
            pass

    def send_event(self, name: str, data: dict):
        """Sends a named event with a data payload to the parent."""
        try:
            self._queue.put_nowait(("event", (name, data)))
        except Full:
            pass

    def _create_sub_context(
        self,
        base_progress: float,
        progress_range: float,
        total: float,
        **kwargs,
    ) -> "ExecutionContextProxy":
        """
        Creates a sub-context that reports progress within a specified range.
        """
        # The new proxy gets its own total for its own progress calculations
        new_proxy = ExecutionContextProxy(
            self._queue, base_progress, progress_range
        )
        new_proxy.set_total(total)
        return new_proxy

    def is_cancelled(self) -> bool:
        """
        Provides a compatible API with ExecutionContext. The parent TaskManager
        is responsible for terminating the process on cancellation.
        """
        return False

    def flush(self):
        """
        Provides a compatible API with ExecutionContext. In the proxy,
        messages are sent immediately, so there is nothing to flush.
        """
        pass
