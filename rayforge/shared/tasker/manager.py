"""
TaskManager module for managing task execution.
"""

from __future__ import annotations
import asyncio
import logging
import threading
from typing import (
    Any,
    Callable,
    Coroutine,
    Dict,
    Iterator,
    Optional,
)
from blinker import Signal
from ..util.glib import idle_add
from .context import ExecutionContext
from .task import Task
from .pool import WorkerPoolManager


logger = logging.getLogger(__name__)


class TaskManager:
    def __init__(
        self, main_thread_scheduler: Optional[Callable] = None
    ) -> None:
        logger.debug("Initializing TaskManager")
        self._tasks: Dict[Any, Task] = {}
        self._progress_map: Dict[
            Any, float
        ] = {}  # Stores progress of all current tasks
        # Stores callbacks for tasks running in the pool
        self._pooled_task_callbacks: Dict[Any, Callable[[Task], None]] = {}

        self._lock = threading.RLock()
        self.tasks_updated: Signal = Signal()
        self._loop: asyncio.AbstractEventLoop = asyncio.new_event_loop()
        self._thread: threading.Thread = threading.Thread(
            target=self._run_event_loop, args=(self._loop,), daemon=True
        )
        self._main_thread_scheduler = main_thread_scheduler or idle_add
        self._thread.start()

        # Initialize the worker pool
        self._pool = WorkerPoolManager()
        self._connect_pool_signals()

    def _connect_pool_signals(self):
        """Connects to signals emitted by the WorkerPoolManager."""
        self._pool.task_completed.connect(self._on_pool_task_completed)
        self._pool.task_failed.connect(self._on_pool_task_failed)
        self._pool.task_progress_updated.connect(self._on_pool_task_progress)
        self._pool.task_message_updated.connect(self._on_pool_task_message)
        self._pool.task_event_received.connect(self._on_pool_task_event)

    def __len__(self) -> int:
        """Return the number of active tasks."""
        with self._lock:
            return len(self._tasks)

    def __iter__(self) -> Iterator[Task]:
        """Return an iterator over the active tasks."""
        with self._lock:
            # Return an iterator over a copy of the tasks to prevent
            # "RuntimeError: dictionary changed size during iteration"
            # if tasks are added/removed while iterating.
            return iter(list(self._tasks.values()))

    def has_tasks(self) -> bool:
        """Return True if there are any active tasks, False otherwise."""
        with self._lock:
            return bool(self._tasks)

    def _run_event_loop(self, loop: asyncio.AbstractEventLoop) -> None:
        """Run the asyncio event loop in a background thread."""
        asyncio.set_event_loop(loop)
        loop.run_forever()

    def add_task(
        self, task: Task, when_done: Optional[Callable[[Task], None]] = None
    ) -> None:
        """Add an asyncio-based task to the manager."""
        with self._lock:
            # If the manager was idle, this is a new batch of work.
            if not self._tasks:
                self._progress_map.clear()

            old_task = self._tasks.get(task.key)
            if old_task:
                logger.debug(
                    f"TaskManager: Found existing task key '{task.key}'. "
                    f"Attempting cancellation."
                )
                self.cancel_task(old_task.key)
            else:
                logger.debug(f"TaskManager: Adding new task key '{task.key}'.")

            self._tasks[task.key] = task
            self._progress_map[task.key] = 0.0
            task.status_changed.connect(self._on_task_updated)

            # Emit signal immediately when a new task is added
            self._emit_tasks_updated_unsafe()

    def add_coroutine(
        self,
        coro: Callable[..., Coroutine[Any, Any, Any]],
        *args: Any,
        key: Optional[Any] = None,
        when_done: Optional[Callable[[Task], None]] = None,
        **kwargs: Any,
    ) -> None:
        """
        Add a raw coroutine to the manager.
        The coroutine will be wrapped in a Task object internally.
        It is expected that the coroutine accepts an ExecutionContext
        as its first argument, followed by any other *args and **kwargs.
        """
        task = Task(coro, *args, key=key, **kwargs)
        self.add_task(task, when_done)

        # Coroutines use the asyncio event loop
        asyncio.run_coroutine_threadsafe(
            self._run_task(task, when_done), self._loop
        )

    def schedule_on_main_thread(
        self, callback: Callable[..., Any], *args: Any, **kwargs: Any
    ) -> None:
        """
        Schedules a callable to be executed on the main thread's event loop.

        This is the designated way for background threads or task callbacks to
        safely interact with the main thread (e.g., for UI updates).
        """
        self._main_thread_scheduler(callback, *args, **kwargs)

    async def run_in_executor(
        self, func: Callable[..., Any], *args: Any
    ) -> Any:
        """
        Runs a synchronous function in a separate thread using asyncio's
        default executor and returns the result. This is useful for offloading
        blocking, CPU-bound work from an async coroutine.
        """
        # The first argument 'None' tells asyncio to use its default
        # ThreadPoolExecutor.
        return await self._loop.run_in_executor(None, func, *args)

    def run_thread(
        self,
        func: Callable[..., Any],
        *args: Any,
        key: Optional[Any] = None,
        when_done: Optional[Callable[[Task], None]] = None,
        **kwargs: Any,
    ) -> Task:
        """
        Creates, configures, and schedules a task to run a synchronous function
        in a background thread.
        """

        async def thread_wrapper(
            context: ExecutionContext, *args: Any, **kwargs: Any
        ) -> Any:
            # This is running inside the TaskManager's event loop thread.
            # We use run_in_executor to move the blocking call to a *different*
            # thread (from the default thread pool executor), ensuring the
            # TaskManager's own event loop is not blocked.
            result = await self.run_in_executor(func, *args, **kwargs)
            return result

        # We create a task with the async wrapper.
        # The original sync function's args/kwargs are passed through.
        task = Task(thread_wrapper, *args, key=key, **kwargs)
        self.add_task(task, when_done)

        # Schedule the async wrapper to be run.
        asyncio.run_coroutine_threadsafe(
            self._run_task(task, when_done), self._loop
        )
        return task

    def run_process(
        self,
        func: Callable[..., Any],
        *args: Any,
        key: Optional[Any] = None,
        when_done: Optional[Callable[[Task], None]] = None,
        when_event: Optional[Callable[[Task, str, dict], None]] = None,
        **kwargs: Any,
    ) -> Task:
        """
        Creates, configures, and schedules a task to run in the worker pool.
        """
        logger.debug(f"Creating task for worker pool {key}")

        # Define a no-op async placeholder. The Task object requires a
        # coroutine, but we won't be running it via asyncio.
        async def _noop_coro(*_args, **_kwargs):
            pass

        # We pass the *real* function and args to the Task object just for
        # bookkeeping, even though the Task object itself won't execute them.
        task = Task(_noop_coro, func, *args, key=key, **kwargs)

        if when_event:
            task.event_received.connect(when_event, weak=False)

        with self._lock:
            # If the manager was idle, this is a new batch of work.
            if not self._tasks:
                self._progress_map.clear()

            old_task = self._tasks.get(task.key)
            if old_task:
                logger.debug(
                    f"TaskManager: Found existing task key '{task.key}'. "
                    f"Attempting cancellation."
                )
                self.cancel_task(old_task.key)

            self._tasks[task.key] = task
            self._progress_map[task.key] = 0.0
            if when_done:
                self._pooled_task_callbacks[task.key] = when_done

            task.status_changed.connect(self._on_task_updated)
            self._emit_tasks_updated_unsafe()

        # Manually set status to running and notify
        task._status = "running"
        task._emit_status_changed()

        # Submit the actual work to the pool
        self._pool.submit(task.key, task.id, func, *args, **kwargs)

        return task

    def cancel_task(self, key: Any) -> None:
        """
        Cancels a running task by its key. This is the authoritative method
        for initiating a cancellation.
        """
        with self._lock:
            task = self._tasks.get(key)
            if not task or task.is_final():
                return

            logger.debug(f"TaskManager: Cancelling task with key '{key}'.")

            # Check if this is a pooled task by checking if it has a callback
            # registered in the pooled task dictionary.
            is_pooled = key in self._pooled_task_callbacks

            # Set the internal cancelled flag on the Task object.
            # For asyncio tasks, this will also cancel the underlying future.
            task.cancel()

            # For pooled tasks, we perform immediate finalization.
            if is_pooled:
                # Tell the pool to ignore any future messages from this ID.
                self._pool.cancel(key, task.id)

                # Immediately finalize the task as 'canceled'.
                task._status = "canceled"
                task._emit_status_changed()

                # Get the callback and clean up immediately.
                when_done = self._pooled_task_callbacks.pop(key, None)
                self._cleanup_task(task)
                if when_done:
                    # Schedule the callback to run on the main thread,
                    # ensuring consistent behavior with completed tasks.
                    self._main_thread_scheduler(when_done, task)

            # For non-pooled (asyncio/thread) tasks,
            # _run_task will handle the cleanup and callback when the
            # CancelledError is caught.

    async def _run_task(
        self, task: Task, when_done: Optional[Callable[[Task], None]]
    ) -> None:
        """Run an asyncio task and clean up when done."""
        context = ExecutionContext(
            update_callback=task.update,
            check_cancelled=task.is_cancelled,
        )
        context.task = task
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
            if when_done:
                self._main_thread_scheduler(when_done, task)

    # === Worker Pool Signal Handlers (runs on listener thread) ===

    def _on_pool_task_completed(self, sender, key, task_id, result):
        self._main_thread_scheduler(
            self._finalize_pooled_task,
            key,
            task_id,
            "completed",
            result=result,
        )

    def _on_pool_task_failed(self, sender, key, task_id, error):
        self._main_thread_scheduler(
            self._finalize_pooled_task, key, task_id, "failed", error=error
        )

    def _on_pool_task_progress(self, sender, key, task_id, progress):
        self._main_thread_scheduler(
            self._update_pooled_task, key, task_id, progress=progress
        )

    def _on_pool_task_message(self, sender, key, task_id, message):
        self._main_thread_scheduler(
            self._update_pooled_task, key, task_id, message=message
        )

    def _on_pool_task_event(self, sender, key, task_id, event_name, data):
        # Schedule the event dispatch on the main thread
        self._main_thread_scheduler(
            self._dispatch_pooled_task_event, key, task_id, event_name, data
        )

    # === Main Thread Update Methods for Pooled Tasks ===

    def _update_pooled_task(
        self,
        key: Any,
        task_id: int,
        progress: Optional[float] = None,
        message: Optional[str] = None,
    ):
        """Updates a Task object from the main thread."""
        with self._lock:
            task = self._tasks.get(key)
        if task and task.id == task_id:
            task.update(progress, message)
        else:
            logger.debug(
                f"Ignoring progress/message for stale task instance for key "
                f"'{key}' (id: {task_id})."
            )

    def _dispatch_pooled_task_event(
        self, key: Any, task_id: int, event_name: str, data: dict
    ):
        """Dispatches a task event from the main thread."""
        with self._lock:
            task = self._tasks.get(key)

        if task and task.id == task_id:
            logger.debug(
                f"TaskManager: Dispatching event '{event_name}' for task "
                f"'{task.key}'."
            )
            task.event_received.send(task, event_name=event_name, data=data)
        else:
            logger.debug(
                f"Ignoring event '{event_name}' for stale task instance for "
                f"key '{key}' (id: {task_id})."
            )

    def _finalize_pooled_task(
        self,
        key: Any,
        task_id: int,
        status: str,
        result: Any = None,
        error: Optional[str] = None,
    ):
        """Finalizes a pooled task from the main thread."""
        with self._lock:
            task = self._tasks.get(key)
            when_done = self._pooled_task_callbacks.pop(key, None)

        if not task or task.id != task_id:
            logger.debug(
                f"Received result for stale/unknown task instance for key "
                f"'{key}'. Ignoring."
            )
            # Re-add the callback for the *new* active task if it exists.
            if task and when_done:
                self._pooled_task_callbacks[key] = when_done
            return

        # If a cancellation happened, the status will already be 'canceled'.
        # We should not overwrite it.
        if not task.is_cancelled():
            task._status = status
            if status == "completed":
                task._progress = 1.0
                task._task_result = result
            elif status == "failed":
                # We got a string traceback, wrap it in an Exception
                logger.error(f"Task {key} failed in worker pool:\n{error}")
                task._task_exception = Exception(error)

        # Emit one final, authoritative signal for all outcomes.
        task._emit_status_changed()

        # This block must run for all outcomes: completed, failed,
        # and cancelled to ensure cleanup and user notification.
        self._cleanup_task(task)
        if when_done:
            when_done(task)

    def _cleanup_task(self, task: Task) -> None:
        """
        Clean up a completed task.
        """
        with self._lock:
            current_task_in_dict = self._tasks.get(task.key)
            if current_task_in_dict is task:
                logger.debug(
                    f"TaskManager: Cleaning up task '{task.key}' "
                    f"(status: {task.get_status()})."
                )
                del self._tasks[task.key]
                # Ensure callback is removed if it exists
                self._pooled_task_callbacks.pop(task.key, None)

                # DO NOT delete from _progress_map. The final progress
                # value (usually 1.0) must be kept for accurate
                # overall progress calculation until the next batch starts.
                # The map is cleared when a new batch begins.
            else:
                # This task finished, but it's no longer the active one
                # for this key in the dictionary (it was replaced).
                logger.debug(
                    f"TaskManager: Skipping cleanup for finished task "
                    f"'{task.key}' (status: {task.get_status()}) as it was "
                    f"already replaced in the manager."
                )
            self._emit_tasks_updated_unsafe()

    def _on_task_updated(self, task: Task) -> None:
        """Handle task status changes. This method is thread-safe."""
        with self._lock:
            if task.key in self._progress_map:
                self._progress_map[task.key] = task.get_progress()
            self._emit_tasks_updated_unsafe()

    def _emit_tasks_updated_unsafe(self) -> None:
        """
        Emit a signal with current state. Must be called with the lock held.
        """
        progress = self.get_overall_progress_unsafe()
        tasks = list(self._tasks.values())
        self._main_thread_scheduler(
            self.tasks_updated.send, self, tasks=tasks, progress=progress
        )

    def get_overall_progress(self) -> float:
        """Calculate overall progress. This method is thread-safe."""
        with self._lock:
            return self.get_overall_progress_unsafe()

    def get_overall_progress_unsafe(self) -> float:
        """Calculate overall progress. Assumes lock is held."""
        if not self._tasks:
            # If there are no active tasks, progress is 100%
            return 1.0
        if not self._progress_map:
            # This can happen briefly if tasks are added but the map isn't
            # populated yet.
            return 0.0
        return sum(self._progress_map.values()) / len(self._progress_map)

    def shutdown(self) -> None:
        """
        Cancel all tasks, shut down the worker pool, and stop the event loop.
        This method is thread-safe.
        """
        try:
            with self._lock:
                tasks_to_cancel = list(self._tasks.values())

            logger.debug(
                f"Shutting down. Cancelling {len(tasks_to_cancel)} tasks"
            )
            for task in tasks_to_cancel:
                self.cancel_task(task.key)

            # Shut down the worker pool. This will wait for workers to exit.
            self._pool.shutdown()

            # Stop the asyncio loop
            if self._loop.is_running():
                self._loop.call_soon_threadsafe(self._loop.stop)
            self._thread.join(timeout=1.0)
            logger.debug("TaskManager shutdown complete.")
        except KeyboardInterrupt:
            logger.debug(
                "TaskManager shutdown interrupted by user. "
                "Suppressing traceback."
            )
            pass


class TaskManagerProxy:
    """
    A lazy-initializing proxy for the TaskManager singleton.

    This object can be safely created at the module level. The real
    TaskManager instance (with its threads and processes) is only created
    when one of its methods is accessed for the first time. This avoids
    the multiprocessing `RuntimeError` on systems that use 'spawn'.
    """

    def __init__(self):
        self._instance: Optional[TaskManager] = None
        self._lock = threading.Lock()

    def _get_instance(self) -> TaskManager:
        """
        Lazily creates the TaskManager instance in a thread-safe manner.
        """
        if self._instance is None:
            with self._lock:
                # Double-check lock to prevent race conditions
                if self._instance is None:
                    logger.debug(
                        "First use of TaskManager detected. "
                        "Initializing the real instance."
                    )
                    self._instance = TaskManager()
        return self._instance

    def __getattr__(self, name: str) -> Any:
        """
        Delegates attribute access to the real TaskManager instance,
        creating it on first access.
        """
        # Forward the call to the real instance.
        return getattr(self._get_instance(), name)
