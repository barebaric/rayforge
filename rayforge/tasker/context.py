"""
ExecutionContext module for managing task execution context.
"""

import abc
import logging
import threading
from typing import Optional, Callable
from queue import Full
from multiprocessing.queues import Queue
from ..util.glib import idle_add


logger = logging.getLogger(__name__)


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
        self, progress_queue: Queue, base_progress=0.0, progress_range=1.0
    ):
        super().__init__(base_progress, progress_range, total=1.0)
        self._queue = progress_queue

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


class ExecutionContext(BaseExecutionContext):
    def __init__(
        self,
        update_callback: Optional[
            Callable[[Optional[float], Optional[str]], None]
        ] = None,
        check_cancelled: Optional[Callable[[], bool]] = None,
        debounce_interval_ms: int = 100,
        # Internal args for sub-contexting
        _parent_context: Optional["ExecutionContext"] = None,
        _base_progress: float = 0.0,
        _progress_range: float = 1.0,
        _total: float = 1.0,
    ):
        super().__init__(_base_progress, _progress_range, _total)
        self._parent_context = _parent_context

        if self._parent_context:
            # This is a sub-context. It doesn't own resources.
            self._root_context = self._parent_context._get_root()
            self._check_cancelled = (
                check_cancelled or self._root_context.is_cancelled
            )
            # These are only used by the root context
            self._update_callback = None
            self._debounce_interval_sec = 0
            self._update_timer = None
            self._pending_progress = None
            self._pending_message = None
            self._lock = None
        else:
            # This is a root context. Initialize resources.
            self._root_context = self
            self._update_callback = update_callback
            self._check_cancelled = check_cancelled or (lambda: False)
            self._debounce_interval_sec = debounce_interval_ms / 1000.0
            self._update_timer: Optional[threading.Timer] = None
            self._pending_progress: Optional[float] = None
            self._pending_message: Optional[str] = None
            self._lock = threading.Lock()

    def _get_root(self) -> "ExecutionContext":
        """Returns the root context in the chain."""
        return self._root_context

    def _fire_update(self):
        """Called by the timer to schedule a UI update."""
        assert self._lock is not None, (
            "_fire_update() called on a non-root context"
        )
        with self._lock:
            if self._update_timer is None:
                return
            progress = self._pending_progress
            message = self._pending_message
            self._pending_progress = None
            self._pending_message = None
            self._update_timer = None

        if self._update_callback and not self.is_cancelled():
            idle_add(self._update_callback, progress, message)

    def _schedule_update(self):
        """(Re)schedules the update timer for the root context."""
        assert self._lock is not None, (
            "_schedule_update() called on a non-root context"
        )
        if self._update_timer is None:
            self._update_timer = threading.Timer(
                self._debounce_interval_sec, self._fire_update
            )
            self._update_timer.start()

    def _update_root_state(
        self, progress: Optional[float] = None, message: Optional[str] = None
    ):
        """
        Sets pending state on the root and schedules an update.
        """
        assert self._lock is not None, (
            "_update_root_state() called on a non-root context"
        )
        with self._lock:
            if progress is not None:
                self._pending_progress = progress
            if message is not None:
                self._pending_message = message
            self._schedule_update()

    def _report_normalized_progress(self, progress: float):
        """
        The core logic for handling 0.0-1.0 progress values.
        This calculates the final global progress and reports it to the root.
        """
        progress = max(0.0, min(1.0, progress))
        global_progress = self._base + (progress * self._range)
        self._get_root()._update_root_state(progress=global_progress)

    def is_cancelled(self) -> bool:
        """Checks if the operation has been cancelled."""
        return self._check_cancelled()

    def set_message(self, message: str):
        """Sets a descriptive message."""
        self._get_root()._update_root_state(message=message)

    def flush(self):
        """Immediately sends the last known values to the UI."""
        root = self._get_root()
        if self is not root:
            root.flush()
            return

        assert self._lock is not None, "flush() called on a non-root context"
        with self._lock:
            if self._update_timer:
                self._update_timer.cancel()
                self._update_timer = None
            progress = self._pending_progress
            message = self._pending_message
            self._pending_progress = None
            self._pending_message = None

        if (
            self._update_callback
            and not self.is_cancelled()
            and (progress is not None or message is not None)
        ):
            idle_add(self._update_callback, progress, message)

    def _create_sub_context(
        self,
        base_progress: float,
        progress_range: float,
        total: float,
        **kwargs,
    ) -> "ExecutionContext":
        """
        Creates a sub-context that reports progress within a specified
        range of this context's progress.
        """
        return ExecutionContext(
            _parent_context=self,
            _base_progress=base_progress,
            _progress_range=progress_range,
            _total=total,
            check_cancelled=kwargs.get("check_cancelled"),
        )
