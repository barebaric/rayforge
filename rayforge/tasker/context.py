"""
ExecutionContext module for managing task execution context.
"""
import logging
import threading
from typing import Optional, Callable
from queue import Full
from ..util.glib import idle_add


logger = logging.getLogger(__name__)


class ExecutionContextProxy:
    """
    A pickleable proxy for reporting progress from a subprocess via a queue.
    """

    def __init__(self, progress_queue, base_progress=0.0, progress_range=1.0):
        self._queue = progress_queue
        self._base = base_progress
        self._range = progress_range
        self._total = 1.0  # Default total for normalization
        self.task = None  # Add task attribute

    def _report_normalized_progress(self, progress: float):
        """
        (Internal) Reports a 0.0-1.0 progress value, scaled to the proxy's
        range.
        """

        # Clamp to a valid range before scaling
        progress = max(0.0, min(1.0, progress))
        scaled_progress = self._base + (progress * self._range)
        try:
            self._queue.put_nowait(("progress", scaled_progress))
        except Full:
            pass  # If the queue is full, we drop the update.

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

    def set_message(self, message: str):
        try:
            self._queue.put_nowait(("message", message))
        except Full:
            pass

    def sub_context(
        self, base_progress, progress_range, total: float = 1.0, **kwargs
    ) -> "ExecutionContextProxy":
        """
        Creates a sub-context that reports progress within a specified range.
        """
        new_base = self._base + (base_progress * self._range)
        new_range = self._range * progress_range
        # The new proxy gets its own total for its own progress calculations
        new_proxy = ExecutionContextProxy(self._queue, new_base, new_range)
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


class ExecutionContext:
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
        self._parent_context = _parent_context
        self._total = float(_total) if _total > 0 else 1.0
        self.task = None  # Add task attribute

        if self._parent_context:
            # This is a sub-context. It doesn't own any resources.
            # It just delegates to the parent.
            self._update_callback = None
            self._check_cancelled = (
                check_cancelled or self._parent_context.is_cancelled
            )
            self._debounce_interval_sec = 0  # not used
            self._update_timer = None
            self._pending_progress = None
            self._pending_message = None
            self._lock = None  # not needed
            self._base_progress = _base_progress
            self._progress_range = _progress_range
        else:
            # This is a root context. Initialize as before.
            self._update_callback = update_callback
            self._check_cancelled = check_cancelled or (lambda: False)
            self._debounce_interval_sec = debounce_interval_ms / 1000.0
            self._update_timer: Optional[threading.Timer] = None
            self._pending_progress: Optional[float] = None
            self._pending_message: Optional[str] = None
            self._lock = threading.Lock()
            # Root context spans the full range by definition.
            self._base_progress = 0.0
            self._progress_range = 1.0

    def _fire_update(self):
        """(Internal) Called by the timer to schedule a UI update."""
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
        """(Internal) (Re)schedules the update timer."""
        # Only schedule an update if a timer is not already running.
        # This ensures updates go out roughly every `_debounce_interval_sec`
        # instead of being perpetually postponed.
        if self._update_timer is None:
            self._update_timer = threading.Timer(
                self._debounce_interval_sec, self._fire_update
            )
            self._update_timer.start()

    def _report_normalized_progress(self, normalized_progress: float):
        """
        (Internal) The core logic for handling 0.0-1.0 progress values.
        This is how sub-contexts communicate with their parents.
        """
        # Clamp to a valid range.
        normalized_progress = max(0.0, min(1.0, normalized_progress))

        if self._parent_context:
            # This is a sub-context. Calculate progress in parent's scale
            # and delegate the call up the chain.
            parent_progress = (
                self._base_progress
                + normalized_progress * self._progress_range
            )
            self._parent_context._report_normalized_progress(parent_progress)
        else:
            # This is the root context. Schedule the debounced UI update.
            assert self._lock is not None, (
                "_report_normalized_progress called on a non-root context"
            )
            with self._lock:
                self._pending_progress = normalized_progress
                self._schedule_update()

    def set_total(self, total: float):
        """
        Sets or updates the total value for this context's progress
        calculations.
        Useful for the root context after it has been created.
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
        normalized_progress = progress / self._total
        self._report_normalized_progress(normalized_progress)

    def is_cancelled(self) -> bool:
        """Checks if the operation has been cancelled."""
        return self._check_cancelled()

    def set_message(self, message: str):
        """Sets a descriptive message."""
        if self._parent_context:
            # Delegate message setting up to the root context.
            self._parent_context.set_message(message)
            return

        assert self._lock is not None, (
            "set_message() called on a non-root context"
        )
        with self._lock:
            self._pending_message = message
            self._schedule_update()

    def flush(self):
        """Immediately sends the last known values to the UI."""
        if self._parent_context:
            self._parent_context.flush()
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

    def sub_context(
        self,
        base_progress: float,
        progress_range: float,
        total: float = 1.0,
        check_cancelled: Optional[Callable[[], bool]] = None,
    ) -> "ExecutionContext":
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
            check_cancelled: An optional, more specific cancellation check.

        Returns:
            A new ExecutionContext instance configured as a sub-context.
        """
        return ExecutionContext(
            _parent_context=self,
            _base_progress=base_progress,
            _progress_range=progress_range,
            _total=total,
            check_cancelled=check_cancelled,
        )
