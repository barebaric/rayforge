"""
ExecutionContext module for managing task execution context.
"""

import logging
import threading
from typing import Optional, Callable
from ..util.glib import idle_add
from .proxy import BaseExecutionContext


logger = logging.getLogger(__name__)


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
