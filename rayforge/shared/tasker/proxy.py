"""
Proxy for reporting progress from subprocesses via a queue.

This module provides ExecutionContextProxy, which extends
ThrottledProgressContext to enable progress reporting from subprocesses
through a queue. It includes throttling to prevent flooding the IPC
queue and provides event sending capabilities.

WARNING: This file MUST NOT have any imports that cause any other
parts of the application to be initialized. It is designed to
be used during subprocess bootstrapping, where no other parts
of the application should be imported or initialized.
We can also not import any GTK or Adw classes here,
as this would cause the GTK main loop to be initialized,
which is not safe during bootstrapping.
In other words, we cannot use GLib.idle_add or similar.
"""

import logging
import time
from queue import Full
from multiprocessing.queues import Queue
from typing import Any, Optional

from rayforge.shared.tasker.progress import (
    ThrottledProgressContext,
)


class ExecutionContextProxy(ThrottledProgressContext):
    """Pickleable proxy for reporting progress from a subprocess via a queue.

    Extends ThrottledProgressContext to enable progress reporting from
    subprocesses through an IPC queue. Progress updates are throttled to
    prevent flooding the queue. Supports event sending for subprocess
    communication with the parent process.
    """

    # Report progress at most ~10 times per second to prevent flooding the UI.
    PROGRESS_REPORT_INTERVAL_S = 0.1

    def __init__(
        self,
        progress_queue: Queue,
        base_progress: float = 0.0,
        progress_range: float = 1.0,
        parent_log_level: int = logging.DEBUG,
        adoption_signals: Any = None,
        task_id: int = 0,
    ):
        super().__init__(base_progress, progress_range, total=1.0)
        self._queue = progress_queue
        self._adoption_signals = adoption_signals
        self._task_id = task_id
        self.parent_log_level = parent_log_level
        self._last_progress_report_time = 0.0
        self._last_reported_progress: Optional[float] = None
        self.task = None

    def _report_normalized_progress(self, progress: float):
        """
        Reports a 0.0-1.0 progress value, scaled to the proxy's
        range. This is throttled to prevent flooding the IPC queue.
        """
        # Clamp to a valid range before scaling
        progress = max(0.0, min(1.0, progress))
        scaled_progress = self._base + (progress * self._range)
        self._last_reported_progress = scaled_progress

        current_time = time.monotonic()
        if (
            current_time - self._last_progress_report_time
            < self.PROGRESS_REPORT_INTERVAL_S
        ):
            return  # Not enough time has passed, skip sending the update.
        self._last_progress_report_time = current_time

        try:
            self._queue.put_nowait(("progress", scaled_progress))
        except Full:
            pass  # If the queue is full, we drop the update.

    def set_message(self, message: str):
        try:
            self._queue.put_nowait(("message", message))
        except Full:
            pass

    def send_event(self, name: str, data: Optional[dict] = None):
        """Sends a named event with a data payload to the parent."""
        try:
            self._queue.put_nowait(
                ("event", (name, data if data is not None else {}))
            )
        except Full:
            pass

    def sub_context(
        self,
        base_progress: float = 0.0,
        progress_range: float = 1.0,
        total: float = 1.0,
        **kwargs,
    ) -> "ExecutionContextProxy":
        """
        Creates a sub-context that reports progress within a specified range.
        """
        new_base = self._base + (base_progress * self._range)
        new_range = self._range * progress_range
        return self._create_sub_context(new_base, new_range, total, **kwargs)

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
            self._queue,
            base_progress,
            progress_range,
            adoption_signals=self._adoption_signals,
            task_id=self._task_id,
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
        Immediately sends any pending updates. This ensures the final
        progress value is always sent, bypassing the throttle.
        """
        if self._last_reported_progress is None:
            return

        try:
            self._queue.put_nowait(("progress", self._last_reported_progress))
            # Reset to avoid duplicate flushes if called multiple times.
            self._last_reported_progress = None
        except Full:
            pass

    def send_event_and_wait(
        self,
        name: str,
        data: Optional[dict] = None,
        timeout: float = 5.0,
        logger: Optional[logging.Logger] = None,
    ) -> bool:
        """
        Sends a named event and waits for adoption acknowledgment.

        This is used for events that carry shared memory handles. On Windows,
        shared memory is destroyed when all handles are closed, so the worker
        must wait for the main process to adopt before closing its handle.

        Args:
            name: The event name.
            data: Optional data payload.
            timeout: Maximum time to wait for acknowledgment in seconds.
            logger: Optional logger for debug messages.

        Returns:
            True if acknowledgment was received and successful, False if NACKed
            or timed out.
        """
        self.send_event(name, data)

        if self._adoption_signals is None:
            return True

        signal_key = f"{self._task_id}:{name}"
        deadline = time.monotonic() + timeout

        while time.monotonic() < deadline:
            if signal_key in self._adoption_signals:
                result = self._adoption_signals.pop(signal_key)
                return bool(result)
            time.sleep(0.01)

        if logger:
            logger.warning(
                f"Timeout waiting for adoption signal for {signal_key}"
            )
        return False
