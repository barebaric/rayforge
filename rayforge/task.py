import asyncio
from asyncio.exceptions import CancelledError
from blinker import Signal
import logging
from typing import Optional, Callable, Coroutine
import threading
from .util.glib import idle_add

logger = logging.getLogger(__name__)


class Task:
    def __init__(self, coro, key=None, monitor_callback=None):
        self.coro = coro
        self.key = key if key is not None else id(self)
        self.monitor_callback = monitor_callback
        self._task = None  # The asyncio.Task executing self.coro
        self._status = "pending"
        self._progress = 0.0
        self._cancel_requested = False  # Flag for early cancellation
        self.status_changed = Signal()
        self.progress_changed = Signal()

    async def run(self):
        """Run the task and update its status and progress."""
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

        # --- Start Execution ---
        self._status = "running"
        self._emit_status_changed()  # Emit running status
        logger.debug(
            f"Task {self.key}: Creating internal asyncio.Task for coro."
        )
        self._task = asyncio.create_task(self.coro)

        # --- Start Progress Monitor (if applicable) ---
        monitor_task = None
        if self.monitor_callback:
            monitor_task = asyncio.create_task(self._monitor_progress())

        # --- Await Coroutine Completion ---
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
            # Optionally propagate? Depends on desired behavior for _run_task
            # For now, we just log and set status. The finally block runs.
        finally:
            # --- Cleanup ---
            logger.debug(
                f"Task {self.key}: Run method finished "
                f"with status '{self._status}'."
            )
            # Ensure progress monitor is stopped if it was started
            if monitor_task and not monitor_task.done():
                monitor_task.cancel()
            # Emit final status and progress
            self._emit_status_changed()
            self._emit_progress_changed()

    async def _monitor_progress(self):
        """Monitor progress using the provided callback."""
        while not self._task.done():
            if self.monitor_callback:
                self._progress = self.monitor_callback(self)
                self._emit_progress_changed()
            await asyncio.sleep(0.1)

    def _emit_status_changed(self):
        """Emit status_changed signal from the main thread."""
        self.status_changed.send(self)

    def _emit_progress_changed(self):
        """Emit progress_changed signal from the main thread."""
        self.progress_changed.send(self)

    def get_status(self):
        """Get the current status of the task."""
        return self._status

    def get_progress(self):
        """Get the current progress of the task."""
        return self._progress

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


class TaskManager:
    def __init__(self):
        self._tasks = {}
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
        coro: Coroutine,
        key=None,
        monitor_callback=None,
        when_done: Optional[Callable] = None
    ):
        """
        Add a raw coroutine to the manager.
        The coroutine will be wrapped in a Task object internally.
        """
        task = Task(coro, key=key, monitor_callback=monitor_callback)
        self.add_task(task, when_done)

    async def _run_task(self, task: Task, when_done: Optional[Callable]):
        """Run the task and clean up when done."""
        # If task.run() raises CancelledError
        # (because task._task was cancelled),
        # that exception will propagate and cancel this _run_task coroutine.
        # The finally block will still execute for cleanup.
        try:
            await task.run()
        finally:
            self._cleanup_task(task)
            self._emit_overall_progress_changed()
            self._emit_running_tasks_changed()
            if when_done:
                idle_add(when_done, task)

    def _cleanup_task(self, task: Task):
        """
        Clean up a completed task. Only remove the task from the dictionary
        if it is the *same object* currently associated with the key.
        This prevents a cancelled task from inadvertently removing a newer
        task that replaced it due to rapid additions.
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
            tasks=list(self._tasks.values())
        )

    def get_overall_progress(self):
        """Calculate the overall progress of all tasks."""
        if not self._tasks:
            return 1.0
        total_progress = sum(
            task.get_progress() for task in self._tasks.values()
        )
        return total_progress / len(self._tasks)

    def shutdown(self):
        """Cancel all tasks and stop the event loop."""
        for task in self._tasks.values():
            task.cancel()
        self._tasks.clear()
        self._loop.call_soon_threadsafe(self._loop.stop)
        self._thread.join()


task_mgr = TaskManager()
