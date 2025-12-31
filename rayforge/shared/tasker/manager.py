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
        self,
        main_thread_scheduler: Optional[Callable] = None,
        worker_initializer: Optional[Callable[..., None]] = None,
        worker_initargs: tuple = (),
    ) -> None:
        logger.debug("Initializing TaskManager")
        self._tasks: Dict[Any, Task] = {}
        # A holding area for recently replaced/cancelled tasks to
        # catch in-flight messages.
        self._zombie_tasks: Dict[int, Task] = {}
        self._progress_map: Dict[
            Any, float
        ] = {}  # Stores progress of all current tasks

        self._lock = threading.RLock()
        self.tasks_updated: Signal = Signal()
        self.loop: asyncio.AbstractEventLoop = asyncio.new_event_loop()
        self._thread: threading.Thread = threading.Thread(
            target=self._run_event_loop, args=(self.loop,), daemon=True
        )
        self._main_thread_scheduler = main_thread_scheduler or idle_add
        self._thread.start()

        # Initialize the worker pool
        self._pool = WorkerPoolManager(
            initializer=worker_initializer, initargs=worker_initargs
        )
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

    def _add_or_replace_task_unsafe(
        self,
        task: Task,
    ):
        """
        Atomically adds a task, replacing any existing task with the same key.
        MUST be called with the lock held.
        """
        # If the manager was idle, this is a new batch of work.
        if not self._tasks and not self._zombie_tasks:
            self._progress_map.clear()

        # Check for and handle an existing task with the same key.
        old_task = self._tasks.get(task.key)
        if old_task:
            logger.debug(
                f"TaskManager: Replacing existing task key '{task.key}'."
            )
            self.cancel_task(old_task.key)

        # Add the new task.
        logger.debug(f"TaskManager: Adding new task key '{task.key}'.")
        self._tasks[task.key] = task
        self._progress_map[task.key] = 0.0

        task.status_changed.connect(self._on_task_updated)
        self._emit_tasks_updated_unsafe()

    def add_task(
        self, task: Task, when_done: Optional[Callable[[Task], None]] = None
    ) -> None:
        """Add an asyncio-based task to the manager."""
        # For asyncio tasks, the when_done is handled by the _run_task wrapper.
        # We store it on the task object for consistency.
        if when_done:
            task.when_done_callback = when_done
        with self._lock:
            self._add_or_replace_task_unsafe(task)

        # Coroutines use the asyncio event loop
        asyncio.run_coroutine_threadsafe(
            self._run_task(task, task.when_done_callback), self.loop
        )

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
        task = Task(coro, *args, key=key, when_done=when_done, **kwargs)
        self.add_task(task)

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
        return await self.loop.run_in_executor(None, func, *args)

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
        task = Task(
            thread_wrapper, *args, key=key, when_done=when_done, **kwargs
        )
        self.add_task(task)
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
        task = Task(
            _noop_coro,
            func,
            *args,
            key=key,
            when_done=when_done,
            task_type="process",
            **kwargs,
        )

        if when_event:
            task.event_received.connect(when_event, weak=False)

        with self._lock:
            self._add_or_replace_task_unsafe(task)

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

            # Set the internal cancelled flag on the Task object.
            # For asyncio tasks, this will also cancel the underlying future.
            task.cancel()

            # For pooled tasks, we just notify the pool.
            if task.task_type == "process":
                self._pool.cancel(key, task.id)
                if task.get_status() != "canceled":
                    task._status = "canceled"
                    task._emit_status_changed()

                # Move the task to the zombie dictionary to await final
                # message.
                del self._tasks[key]
                self._zombie_tasks[task.id] = task
                self._emit_tasks_updated_unsafe()

    def get_task(self, key: Any) -> Optional[Task]:
        """Retrieves a task by its key."""
        with self._lock:
            return self._tasks.get(key)

    async def _run_task(
        self, task: Task, when_done: Optional[Callable[[Task], None]]
    ) -> None:
        """Run an asyncio task and clean up when done."""
        context = ExecutionContext(
            update_callback=task.update,
            check_cancelled=task.is_cancelled,
            scheduler=self.schedule_on_main_thread,
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
            if not (task and task.id == task_id):
                task = self._zombie_tasks.get(task_id)

        if task:
            task.update(progress, message)
        else:
            logger.debug(
                f"Ignoring progress/message for stale/unknown task instance "
                f"for key '{key}' (id: {task_id})."
            )

    def _dispatch_pooled_task_event(
        self, key: Any, task_id: int, event_name: str, data: dict
    ):
        """Dispatches a task event from the main thread."""
        with self._lock:
            # First, check if the event is for the currently active task.
            task = self._tasks.get(key)
            if not (task and task.id == task_id):
                # If not, check if it's for a recently replaced (zombie) task.
                task = self._zombie_tasks.get(task_id)

        if task:
            logger.debug(
                f"TaskManager: Dispatching event '{event_name}' for task "
                f"'{task.key}' (task_id={task_id})."
            )
            task.event_received.send(task, event_name=event_name, data=data)
        else:
            logger.warning(
                f"Received event '{event_name}' for unknown or fully cleaned "
                f"up task key '{key}' (id: {task_id}). Ignoring."
            )

    def _finalize_pooled_task(
        self,
        key: Any,
        task_id: int,
        status: str,
        result: Any = None,
        error: Optional[str] = None,
    ):
        """
        Finalizes a pooled task from the main thread. This is the single
        source of truth for completing a pooled task's lifecycle.
        """
        logger.debug(
            f"Attempting to finalize pooled task '{key}' "
            f"(id: {task_id}) with status '{status}'"
        )
        with self._lock:
            task = None
            # Find the task in either the active or zombie dictionaries
            active_task = self._tasks.get(key)

            if active_task and active_task.id == task_id:
                # This is the currently active task for this key.
                # It is now finished.
                logger.debug(
                    f"Finalizing ACTIVE task '{key}' (id: {task_id})."
                )
                task = active_task
                del self._tasks[key]
            else:
                # It's not the active task. See if it's a zombie.
                zombie_task = self._zombie_tasks.get(task_id)
                if zombie_task:
                    logger.debug(
                        f"Finalizing ZOMBIE task '{key}' (id: {task_id})."
                    )
                    task = zombie_task
                    del self._zombie_tasks[task_id]

            if not task:
                logger.debug(
                    f"Received final message for unknown/cleaned-up task "
                    f"instance for key '{key}' (id: {task_id}). Ignoring."
                )
                return

        # Now that we have the correct task instance and it has been removed
        # from tracking, we can process its completion.

        # Set the final status, unless it was already cancelled.
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

        # Call the user's callback if it was stored on the task.
        when_done = task.when_done_callback
        if when_done:
            logger.debug(
                f"Invoking when_done callback for task '{key}' "
                f"(id: {task.id})."
            )
            when_done(task)
        else:
            logger.debug(
                f"No when_done callback to invoke for task '{key}' "
                f"(id: {task.id})."
            )

        with self._lock:
            self._emit_tasks_updated_unsafe()

    def _cleanup_task(self, task: Task) -> None:
        """
        Clean up a completed asyncio task. This is NOT used for pooled tasks.
        """
        with self._lock:
            current_task_in_dict = self._tasks.get(task.key)
            # Only remove the task from the dictionary if it's the one we
            # expect. This prevents a stale task's cleanup from removing a
            # newer, active task.
            if current_task_in_dict is task:
                logger.debug(
                    f"Cleaning up (asyncio) task '{task.key}' "
                    f"(status: {task.get_status()})."
                )
                del self._tasks[task.key]

                # DO NOT delete from _progress_map. The final progress
                # value (usually 1.0) must be kept for accurate
                # overall progress calculation until the next batch starts.
                # The map is cleared when a new batch begins.
            else:
                # This task finished, but it's no longer the active one
                # for this key in the dictionary (it was replaced).
                logger.debug(
                    f"Skipping cleanup for finished (asyncio) task "
                    f"'{task.key}' (status: {task.get_status()}) as it was "
                    f"already replaced in the manager."
                )
            self._emit_tasks_updated_unsafe()

    def _on_task_updated(self, task: Task) -> None:
        """Handle task status changes. This method is thread-safe."""
        with self._lock:
            # Only update progress if the task is still the active one for
            # its key
            active_task = self._tasks.get(task.key)
            if active_task is task:
                self._progress_map[task.key] = task.get_progress()
            self._emit_tasks_updated_unsafe()

    def _emit_tasks_updated_unsafe(self) -> None:
        """
        Emit a signal with current state. Must be called with the lock held.
        """
        progress = self.get_overall_progress_unsafe()
        tasks = list(self._tasks.values())
        self._main_thread_scheduler(
            lambda: self.tasks_updated.send(
                self, tasks=tasks, progress=progress
            )
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

        # Ensure progress map only contains keys for active tasks
        active_keys = self._tasks.keys()
        total_progress = sum(
            self._progress_map.get(k, 0.0) for k in active_keys
        )

        return total_progress / len(active_keys) if active_keys else 1.0

    def shutdown(self) -> None:
        """
        Cancel all tasks, shut down the worker pool, and stop the event loop.
        This method is thread-safe.
        """
        try:
            logger.info("Shutdown started")
            with self._lock:
                tasks_to_cancel = list(self._tasks.values())

            logger.info(f"Active tasks at shutdown: {len(tasks_to_cancel)}")
            logger.info("Cancelling all active tasks...")
            for task in tasks_to_cancel:
                status = task.get_status()
                progress = task.get_progress()
                logger.info(
                    f"  Task '{task.key}': status={status}, "
                    f"progress={progress:.1f}%"
                )
                self.cancel_task(task.key)

            # Shut down the worker pool. This will wait for workers to exit.
            self._pool.shutdown()

            logger.info("Stopping asyncio event loop...")
            # Stop the asyncio loop
            if self.loop.is_running():
                self.loop.call_soon_threadsafe(self.loop.stop)
            logger.debug("Joining thread...")
            self._thread.join(timeout=1.0)
            if self._thread.is_alive():
                logger.warning("thread shutdown timed out, ignoring")
            logger.info("TaskManager shutdown complete.")
        except KeyboardInterrupt:
            logger.debug(
                "TaskManager shutdown interrupted by user. "
                "Suppressing traceback."
            )
            pass

    def wait_until_settled(self, timeout: int) -> bool:
        """
        Wait until all tasks have completed or until timeout is reached.

        This is a thread-safe, non-blocking-loop implementation.

        Args:
            timeout: Maximum time to wait in milliseconds.

        Returns:
            True if all tasks completed before timeout, False if timeout was
            reached.
        """
        # If already settled, return immediately.
        if not self.has_tasks():
            return True

        settled_event = threading.Event()
        timeout_seconds = timeout / 1000.0

        def on_update(sender, tasks, **kwargs):
            """Signal handler that checks if the manager is idle."""
            if not tasks:
                # The manager is now idle. Set the event and disconnect.
                settled_event.set()
                self.tasks_updated.disconnect(on_update)

        # Connect the handler. We don't use a weak reference because we
        # need to guarantee disconnection.
        self.tasks_updated.connect(on_update, weak=False)

        # Wait for the event to be set by the callback. This is a blocking
        # call, but it does NOT block the main_thread_scheduler's event loop,
        # allowing the cleanup callbacks to run and trigger our on_update.
        event_was_set = settled_event.wait(timeout=timeout_seconds)

        # Always try to disconnect in case of a timeout to prevent leaks.
        try:
            self.tasks_updated.disconnect(on_update)
        except Exception:
            # It might have already been disconnected, which is fine.
            pass

        return event_was_set


class TaskManagerProxy:
    """
    A lazy-initializing proxy for the TaskManager singleton.

    This object can be safely created at the module level. The real
    TaskManager instance (with its threads and processes) is only created
    when one of its methods is accessed for the first time. This avoids
    the multiprocessing `RuntimeError` on systems that use 'spawn'.

    To provide worker initialization arguments, call the `initialize` method
    once at application startup before using the task manager.
    """

    def __init__(self):
        self._instance: Optional[TaskManager] = None
        self._lock = threading.Lock()
        self._init_kwargs: Dict[str, Any] = {}

    def initialize(self, **kwargs: Any) -> None:
        """
        Provides configuration for the TaskManager before it is created.
        This must be called before any other TaskManager methods are used.

        Example:
            task_mgr.initialize(
                worker_initializer=some_func,
                worker_initargs=(arg1, arg2)
            )

        Raises:
            RuntimeError: If called after the TaskManager has been created.
        """
        with self._lock:
            if self._instance is not None:
                raise RuntimeError("TaskManager has already been initialized.")
            self._init_kwargs = kwargs

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
                    self._instance = TaskManager(**self._init_kwargs)
        return self._instance

    def __getattr__(self, name: str) -> Any:
        """
        Delegates attribute access to the real TaskManager instance,
        creating it on first access.
        """
        # Forward the call to the real instance.
        return getattr(self._get_instance(), name)
