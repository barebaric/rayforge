import asyncio
from asyncio.exceptions import CancelledError
from blinker import Signal
import logging
from typing import Optional, Callable, Coroutine
import threading
from .util.glib import idle_add

logger = logging.getLogger(__name__)


class ExecutionContext:
    """
    An object that holds the execution context for a task.
    It is thread-safe and performs debouncing in a background
    thread using a single threading.Timer, minimizing load on the
    GLib main loop.
    It supports sub-contexts for cleanly managing nested operations.
    """

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
        # Use the provided total, ensuring it's valid.
        self._total = float(_total) if _total > 0 else 1.0

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
        if self._update_timer:
            self._update_timer.cancel()
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


class Task:
    def __init__(
        self, coro: Callable[..., Coroutine], *args, key=None, **kwargs
    ):
        self.coro = coro
        self.args = args
        self.kwargs = kwargs
        self.key = key if key is not None else id(self)
        self._task = None  # The asyncio.Task executing self.coro
        self._status = "pending"
        self._progress = 0.0
        self._message: Optional[str] = None
        self._cancel_requested = False  # Flag for early cancellation
        self.status_changed = Signal()

    def update(
        self, progress: Optional[float] = None, message: Optional[str] = None
    ):
        """
        Updates task progress and/or message. This method is designed to be
        called from the main thread (e.g., via idle_add) and emits a
        single signal for any change.
        """
        updated = False
        if progress is not None and self._progress != progress:
            self._progress = progress
            updated = True
        if message is not None and self._message != message:
            self._message = message
            updated = True

        if updated:
            self._emit_status_changed()

    async def run(self, context: ExecutionContext):
        """
        Run the task and update its status. The wrapped coroutine is
        responsible for reporting progress via the provided context.
        """
        logger.debug(f"Task {self.key}: Entering run method.")

        # Early cancellation check
        if self._cancel_requested:
            logger.debug(
                f"Task {self.key}: Cancellation requested before coro start."
            )
            self._status = "canceled"
            self._emit_status_changed()
            raise CancelledError("Task cancelled before coro execution")

        # Start execution
        self._status = "running"
        self._emit_status_changed()  # Emit running status
        logger.debug(
            f"Task {self.key}: Creating internal asyncio.Task for coro."
        )

        # Wrap the coroutine in a Task.
        self._task = asyncio.create_task(
            self.coro(context, *self.args, **self.kwargs)
        )

        # Await Coroutine Completion
        try:
            logger.debug(f"Task {self.key}: Awaiting internal asyncio.Task.")
            await self._task
            # If await completes without CancelledError or other Exception:
            logger.debug(f"Task {self.key}: Coro completed successfully.")
            self._status = "completed"
            self._progress = 1.0
        except asyncio.CancelledError:
            # This catches cancellation of self._task (the coro)
            logger.warning(
                f"Task {self.key}: Internal asyncio.Task was cancelled."
            )
            self._status = "canceled"
            # Propagate so the outer _run_task knows about the cancellation
            raise
        except Exception:
            logger.exception(f"Task {self.key}: Coro failed with exception.")
            self._status = "failed"
            # Re-raise so the TaskManager can see and log it.
            raise
        finally:
            logger.debug(
                f"Task {self.key}: Run method finished "
                f"with status '{self._status}'."
            )
            # Emit final status and progress
            self._emit_status_changed()

    def _emit_status_changed(self):
        """Emit status_changed signal from the main thread."""
        self.status_changed.send(self)

    def get_progress(self):
        """Get the current progress of the task."""
        return self._progress

    def get_status(self):
        """Get the current lifecycle status of the task."""
        return self._status

    def get_message(self) -> Optional[str]:
        """Get the current user-facing message for the task."""
        return self._message

    def result(self):
        return self._task.result()

    def cancel(self):
        """
        Request cancellation of the task.
        Sets a flag to prevent starting if not already started,
        and attempts to cancel the underlying asyncio.Task if it exists.
        """
        logger.debug(f"Task {self.key}: Cancel method called.")
        self._cancel_requested = True  # Set flag regardless of current state

        task_to_cancel = self._task
        if task_to_cancel and not task_to_cancel.done():
            logger.info(
                f"Task {self.key}: Attempting to cancel "
                f"running internal asyncio.Task."
            )
            task_to_cancel.cancel()
        elif task_to_cancel:
            logger.debug(
                f"Task {self.key}: Internal asyncio.Task already done."
            )
        else:
            logger.debug(
                f"Task {self.key}: Internal asyncio.Task not yet "
                f"created, flag set."
            )

    def is_cancelled(self) -> bool:
        """Checks if cancellation has been requested for this task."""
        return self._cancel_requested


class TaskManager:
    def __init__(self):
        self._tasks = {}
        self._progress_map = {}  # Stores progress of all current tasks
        self.tasks_updated = Signal()
        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(
            target=self._run_event_loop, args=(self._loop,), daemon=True
        )
        self._thread.start()

    def _run_event_loop(self, loop):
        """Run the asyncio event loop in a background thread."""
        asyncio.set_event_loop(loop)
        loop.run_forever()

    def add_task(self, task: Task, when_done: Optional[Callable] = None):
        """Add a task to the manager."""
        # If the manager was idle, this is a new batch of work.
        if not self._tasks:
            self._progress_map.clear()

        old_task = self._tasks.get(task.key)
        if old_task:
            logger.info(
                f"TaskManager: Found existing task key '{task.key}'. "
                f"Attempting cancellation."
            )
            old_task.cancel()
        else:
            logger.info(f"TaskManager: Adding new task key '{task.key}'.")

        self._tasks[task.key] = task
        self._progress_map[task.key] = 0.0  # Register new task in the batch
        task.status_changed.connect(self._on_task_updated)

        # Emit signal immediately when a new task is added
        self._emit_tasks_updated()

        asyncio.run_coroutine_threadsafe(
            self._run_task(task, when_done), self._loop
        )

    def add_coroutine(
        self,
        coro: Callable[..., Coroutine],
        *args,
        key=None,
        when_done: Optional[Callable] = None,
        **kwargs,
    ):
        """
        Add a raw coroutine to the manager.
        The coroutine will be wrapped in a Task object internally.
        It is expected that the coroutine accepts an ExecutionContext
        as its first argument, followed by any other *args and **kwargs.
        """
        task = Task(coro, *args, key=key, **kwargs)
        self.add_task(task, when_done)

    async def _run_task(self, task: Task, when_done: Optional[Callable]):
        """Run the task and clean up when done."""
        context = ExecutionContext(
            update_callback=task.update,
            check_cancelled=task.is_cancelled,
        )
        try:
            await task.run(context)
        except Exception:
            # This is the master error handler for all background tasks.
            logger.error(
                f"Unhandled exception in managed task '{task.key}':",
                exc_info=True,
            )
        finally:
            context.flush()
            self._cleanup_task(task)
            self._emit_tasks_updated()
            if when_done:
                idle_add(when_done, task)

    def _cleanup_task(self, task: Task):
        """
        Clean up a completed task.
        """
        current_task_in_dict = self._tasks.get(task.key)
        if current_task_in_dict is task:  # Check object identity
            logger.debug(
                f"TaskManager: Cleaning up task '{task.key}' "
                f"(status: {task.get_status()})."
            )
            del self._tasks[task.key]
        else:
            # This task finished, but it's no longer the active one
            # for this key in the dictionary (it was replaced).
            # Don't remove the newer task.
            logger.debug(
                f"TaskManager: Skipping cleanup for finished task "
                f"'{task.key}' (status: {task.get_status()}) as it was "
                f"already replaced in the manager."
            )

    def _on_task_updated(self, task):
        """Handle task status or progress changes."""
        if task.key in self._progress_map:
            self._progress_map[task.key] = task.get_progress()
        self._emit_tasks_updated()

    def _emit_tasks_updated(self):
        """Emit a single consolidated signal from the main thread."""
        progress = self.get_overall_progress()
        tasks = list(self._tasks.values())
        idle_add(
            self.tasks_updated.send,
            self,
            tasks=tasks,
            progress=progress
        )

    def get_overall_progress(self):
        """Calculate the overall progress of all tasks."""
        if not self._progress_map:
            return 1.0

        total_progress = sum(self._progress_map.values())
        return total_progress / len(self._progress_map)

    def shutdown(self):
        """Cancel all tasks and stop the event loop."""
        for task in self._tasks.values():
            task.cancel()
        self._tasks.clear()
        self._progress_map.clear()
        self._loop.call_soon_threadsafe(self._loop.stop)
        self._thread.join()


task_mgr = TaskManager()
