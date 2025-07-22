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
    thread using threading.Timer, minimizing load on the GLib main loop.
    """

    def __init__(
        self,
        progress_callback: Optional[Callable[[float], None]] = None,
        check_cancelled: Callable[[], bool] = lambda: False,
        message_callback: Optional[Callable[[str], None]] = None,
        debounce_interval_ms: int = 100,
    ):
        self._progress_callback = progress_callback
        self._check_cancelled = check_cancelled
        self._message_callback = message_callback
        self._debounce_interval_sec = debounce_interval_ms / 1000.0

        # These timers are for debouncing.
        self._progress_timer: Optional[threading.Timer] = None
        self._message_timer: Optional[threading.Timer] = None

        self._pending_progress: Optional[float] = None
        self._pending_message: Optional[str] = None
        self._lock = threading.Lock()

    def _fire_progress(self):
        """
        Called by the threading.Timer in a background thread.
        Schedules the actual UI update on the main loop.
        """
        with self._lock:
            if self._pending_progress is not None and self._progress_callback:
                if not self.is_cancelled():
                    # The ONLY interaction with the main loop.
                    idle_add(self._progress_callback, self._pending_progress)
            self._pending_progress = None
            self._progress_timer = None

    def _fire_message(self):
        """
        Called by the threading.Timer in a background thread.
        Schedules the actual UI update on the main loop.
        """
        with self._lock:
            if self._pending_message is not None and self._message_callback:
                if not self.is_cancelled():
                    # The ONLY interaction with the main loop.
                    idle_add(self._message_callback, self._pending_message)
            self._pending_message = None
            self._message_timer = None

    def set_progress(self, progress: float):
        """
        Sets the progress. All debouncing logic happens in background threads.
        """
        with self._lock:
            if self._progress_timer:
                self._progress_timer.cancel()

            self._pending_progress = progress
            self._progress_timer = threading.Timer(
                self._debounce_interval_sec, self._fire_progress
            )
            self._progress_timer.start()

    def is_cancelled(self) -> bool:
        """Checks if the operation has been cancelled."""
        return self._check_cancelled()

    def set_message(self, message: str):
        """
        Sets a descriptive message. All debouncing logic happens in
        background threads.
        """
        with self._lock:
            if self._message_timer:
                self._message_timer.cancel()

            self._pending_message = message
            self._message_timer = threading.Timer(
                self._debounce_interval_sec, self._fire_message
            )
            self._message_timer.start()

    def flush(self):
        """
        Cancels any pending debounced calls and immediately sends the
        last known values to the main loop for UI update.
        """
        with self._lock:
            # Flush progress
            if self._progress_timer:
                self._progress_timer.cancel()
                self._progress_timer = None
            if self._pending_progress is not None and self._progress_callback:
                if not self.is_cancelled():
                    idle_add(self._progress_callback, self._pending_progress)
            self._pending_progress = None

            # Flush message
            if self._message_timer:
                self._message_timer.cancel()
                self._message_timer = None
            if self._pending_message is not None and self._message_callback:
                if not self.is_cancelled():
                    idle_add(self._message_callback, self._pending_message)
            self._pending_message = None


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
        self.progress_changed = Signal()

    async def run(self, context: ExecutionContext):
        """
        Run the task and update its status. The wrapped coroutine is
        responsible for reporting progress via the provided context.
        """
        logger.debug(f"Task {self.key}: Entering run method.")

        # --- Early Cancellation Check ---
        if self._cancel_requested:
            logger.debug(
                f"Task {self.key}: Cancellation requested before coro start."
            )
            self._status = "canceled"
            self._emit_status_changed()
            # Ensure status update even if cancelled early
            self._emit_progress_changed()  # Ensure progress update
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
            self._emit_progress_changed()

    def _emit_status_changed(self):
        """Emit status_changed signal from the main thread."""
        self.status_changed.send(self)

    def _emit_progress_changed(self):
        """Emit progress_changed signal from the main thread."""
        self.progress_changed.send(self)

    def set_progress(self, progress: float):
        """Sets the progress of the task and emits a signal."""
        self._progress = progress
        self._emit_progress_changed()

    def get_progress(self):
        """Get the current progress of the task."""
        return self._progress

    def get_status(self):
        """Get the current lifecycle status of the task."""
        return self._status

    def set_message(self, message: str):
        """Sets the user-facing message and emits the status_changed signal."""
        self._message = message
        self._emit_status_changed()

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
        self.overall_progress_changed = Signal()
        self.running_tasks_changed = Signal()
        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(
            target=self._run_event_loop,
            args=(self._loop,),
            daemon=True
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
        task.status_changed.connect(self._on_task_status_changed)
        task.progress_changed.connect(self._on_task_progress_changed)

        # Emit signals immediately when a new task is added
        self._emit_overall_progress_changed()
        self._emit_running_tasks_changed()

        asyncio.run_coroutine_threadsafe(
            self._run_task(task, when_done),
            self._loop
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
            progress_callback=task.set_progress,
            check_cancelled=task.is_cancelled,
            message_callback=task.set_message,
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
            self._emit_overall_progress_changed()
            self._emit_running_tasks_changed()
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

    def _on_task_status_changed(self, task):
        """Handle task status changes."""
        self._emit_overall_progress_changed()
        self._emit_running_tasks_changed()

    def _on_task_progress_changed(self, task):
        """Handle task progress changes."""
        if task.key in self._progress_map:
            self._progress_map[task.key] = task.get_progress()
        self._emit_overall_progress_changed()

    def _emit_overall_progress_changed(self):
        """Emit overall_progress_changed signal from the main thread."""
        progress = self.get_overall_progress()
        idle_add(
            self.overall_progress_changed.send,
            self,
            progress=progress
        )

    def _emit_running_tasks_changed(self):
        """Emit running_tasks_changed signal from the main thread."""
        idle_add(
            self.running_tasks_changed.send,
            self,
            tasks=list(self._tasks.values()),
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
