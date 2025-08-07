"""
TaskManager module for managing task execution.
"""

from __future__ import annotations
import asyncio
import logging
import threading
import time
from multiprocessing import get_context
from multiprocessing.context import SpawnProcess
from multiprocessing.queues import Queue
from queue import Empty
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
from .process import process_target_wrapper
from .task import Task, CancelledError


logger = logging.getLogger(__name__)


class TaskManager:
    def __init__(self) -> None:
        logger.debug("Initializing TaskManager")
        self._tasks: Dict[Any, Task] = {}
        self._progress_map: Dict[
            Any, float
        ] = {}  # Stores progress of all current tasks
        self._lock = threading.RLock()
        self.tasks_updated: Signal = Signal()
        self._loop: asyncio.AbstractEventLoop = asyncio.new_event_loop()
        self._thread: threading.Thread = threading.Thread(
            target=self._run_event_loop, args=(self._loop,), daemon=True
        )
        self._thread.start()

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
        """Add a task to the manager."""
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
                old_task.cancel()
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
        Creates, configures, and schedules a task to run in a separate
        process.
        """
        logger.debug(f"Creating task for subprocess {key}")

        # Define an async placeholder that matches the required type signature.
        async def _process_placeholder(*_args, **_kwargs):
            pass

        task = Task(_process_placeholder, func, *args, key=key, **kwargs)

        # Connect the event handler BEFORE scheduling the task. This is the
        # key to ensuring stability. The handler itself is never pickled.
        if when_event:
            task.event_received.connect(when_event)

        self.add_task(task, when_done)

        # Schedule the creation and start of the process on the main GTK
        # thread. This will execute after the current call stack unwinds.
        idle_add(self._start_process_on_main_thread, task, when_done)

        return task

    def cancel_task(self, key: Any) -> None:
        """Cancels a running task by its key."""
        with self._lock:
            task = self._tasks.get(key)
            if task:
                logger.debug(f"TaskManager: Cancelling task with key '{key}'.")
                task.cancel()

    async def _run_task(
        self, task: Task, when_done: Optional[Callable[[Task], None]]
    ) -> None:
        """Run the task and clean up when done."""
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
                idle_add(when_done, task)

    def _start_process_on_main_thread(
        self, task: Task, when_done: Optional[Callable[[Task], None]]
    ) -> None:
        """
        Creates and starts the subprocess on the main thread to avoid
        deadlocks.
        Then, it launches a simple thread to monitor the process.
        """
        # Get a fresh context on the main thread.
        mp_context = get_context("spawn")
        queue: Queue[tuple[str, Any]] = mp_context.Queue()

        # Unpack the real function and args from the task object
        user_func, user_args, user_kwargs = (
            task.args[0],
            task.args[1:],
            task.kwargs,
        )

        log_level = logging.getLogger().getEffectiveLevel()
        process_args = (queue, log_level, user_func, user_args, user_kwargs)

        try:
            if task.is_cancelled():
                # Task was cancelled before we even got a chance to run.
                raise CancelledError("Task cancelled before process start.")

            process = mp_context.Process(
                target=process_target_wrapper, args=process_args, daemon=True
            )

            process.start()
            logger.debug(
                f"Task {task.key}: Started subprocess with PID {process.pid}"
            )

            # Now that the process is started, launch the monitor thread.
            monitor_thread = threading.Thread(
                target=self._monitor_subprocess_lifecycle,
                args=(task, when_done, process, queue),
                daemon=True,
            )
            monitor_thread.start()

        except Exception as e:
            # Handle failures during the startup phase
            logger.error(
                f"Task {task.key}: Failed to start process on main thread",
                exc_info=True,
            )
            task._status = "failed"
            task._task_exception = e
            task.status_changed.send(task)
            self._cleanup_task(task)
            if when_done:
                idle_add(when_done, task)

    def _monitor_subprocess_lifecycle(
        self,
        task: Task,
        when_done: Optional[Callable[[Task], None]],
        process: SpawnProcess,
        queue: Queue[tuple[str, Any]],
    ) -> None:
        """
        Synchronously monitors a subprocess lifecycle in a dedicated thread.
        """
        context = ExecutionContext(
            update_callback=task.update,
            check_cancelled=task.is_cancelled,
        )
        context.task = task
        state: Dict[str, Any] = {"result": None, "error": None}

        try:
            # Synchronous monitoring loop
            while process.is_alive():
                self._drain_process_queue(queue, context, state)
                if state.get("error"):
                    break  # Error reported by child
                if task.is_cancelled():
                    raise CancelledError("Task cancelled by parent.")
                time.sleep(0.1)

            self._drain_process_queue(queue, context, state)  # Final drain
            self._check_process_result(process, state, task.key)

            task._status = "completed"
            task._progress = 1.0
            task._task_result = state.get("result")

        except CancelledError as e:
            logger.warning(f"Task {task.key}: Process task was cancelled: {e}")
            task._status = "canceled"
            task._task_exception = e
        except Exception as e:
            logger.error(
                f"Task {task.key}: Process monitor thread failed.",
                exc_info=True,
            )
            task._status = "failed"
            task._task_exception = e
        finally:
            self._cleanup_process_resources(process, task.key)
            context.flush()
            # Manually trigger final status update, since run() wasn't used
            task.status_changed.send(task)
            self._cleanup_task(task)
            if when_done:
                idle_add(when_done, task)

    def _handle_process_queue_message(
        self,
        msg: tuple[str, Any],
        context: ExecutionContext,
        state: Dict[str, Any],
    ) -> None:
        """
        Process a single message from the subprocess queue.

        Args:
            msg: The (type, value) tuple from the queue.
            context: The ExecutionContext for progress reporting.
            state: A mutable dictionary to store 'result' and 'error'.
        """
        msg_type, value = msg
        if msg_type == "progress":
            context._report_normalized_progress(value)
        elif msg_type == "message":
            context.set_message(value)
        elif msg_type == "event":
            if context.task:
                event_name, data = value
                logger.debug(
                    f"TaskManager: Received event '{event_name}' for task "
                    f"'{context.task.key}'. Dispatching via idle_add."
                )
                # Fire the event signal on the Task object.
                # This needs to be done on the main thread.
                idle_add(
                    context.task.event_received.send,
                    context.task,
                    event_name=event_name,
                    data=data,
                )
        elif msg_type == "done":
            state["result"] = value
            if context.task:
                logger.debug(f"Task {context.task.key}: Received 'done'.")
        elif msg_type == "error":
            state["error"] = value
            if context.task:
                logger.error(
                    f"Task {context.task.key}: 'error' from subprocess:"
                    f"\n{value}"
                )

    def _drain_process_queue(
        self,
        queue: Queue[tuple[str, Any]],
        context: ExecutionContext,
        state: Dict[str, Any],
    ) -> None:
        """Drain all pending messages from the subprocess queue."""
        try:
            while True:
                msg = queue.get_nowait()
                self._handle_process_queue_message(msg, context, state)
        except Empty:
            pass

    def _check_process_result(
        self, process: SpawnProcess, state: Dict[str, Any], task_key: Any
    ) -> None:
        """
        Check for errors after a subprocess has finished.

        Args:
            process: The completed multiprocessing.Process object.
            state: A dictionary containing the final 'result' and 'error'.
            task_key: The key of the task for logging/error messages.

        Raises:
            Exception: If the subprocess reported an error or exited with a
                       non-zero status code.
        """
        if state["error"]:
            msg = (
                f"Subprocess for task '{task_key}' failed.\n"
                f"--- Subprocess Traceback ---\n{state['error']}"
            )
            raise Exception(msg)

        if process.exitcode != 0:
            msg = (
                f"Subprocess for task '{task_key}' terminated "
                f"unexpectedly with exit code {process.exitcode}."
            )
            raise Exception(msg)

    def _cleanup_process_resources(
        self, process: SpawnProcess, task_key: Any
    ) -> None:
        """
        Ensure a subprocess is terminated and its resources are closed.

        Args:
            process: The multiprocessing.Process to clean up.
            task_key: The key of the task for logging.
        """
        if process.is_alive():
            logger.warning(
                "Task %s: Terminating subprocess %s.", task_key, process.pid
            )
            process.terminate()
            process.join(timeout=1.0)

            if process.is_alive():
                logger.error(
                    "Task %s: Subprocess %s did not die. Killing.",
                    task_key,
                    process.pid,
                )
                process.kill()
                process.join(timeout=1.0)

        process.close()
        logger.debug("Task %s: Subprocess resources cleaned up.", task_key)

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
                # DO NOT delete from _progress_map. The final progress
                # value (usually 1.0) must be kept for accurate
                # overall progress calculation until the next batch starts.
                # The map is cleared in add_task() when a new batch begins.
            else:
                # This task finished, but it's no longer the active one
                # for this key in the dictionary (it was replaced).
                # Don't remove the newer task.
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
        idle_add(self.tasks_updated.send, self, tasks=tasks, progress=progress)

    def get_overall_progress(self) -> float:
        """Calculate overall progress. This method is thread-safe."""
        with self._lock:
            return self.get_overall_progress_unsafe()

    def get_overall_progress_unsafe(self) -> float:
        """Calculate overall progress. Assumes lock is held."""
        if not self._progress_map:
            return 1.0
        return sum(self._progress_map.values()) / len(self._progress_map)

    def shutdown(self) -> None:
        """
        Cancel all tasks and stop the event loop.
        This method is thread-safe.
        """
        with self._lock:
            tasks_to_cancel = list(self._tasks.values())

        logger.debug(f"Shutting down. Cancelling {len(tasks_to_cancel)} tasks")
        for task in tasks_to_cancel:
            task.cancel()

        # Wait a moment for cancellations to propagate before stopping the loop
        # This is not strictly necessary but can help with cleaner shutdown.
        if tasks_to_cancel:
            time.sleep(0.2)  # Give threads time to see cancellation

        self._loop.call_soon_threadsafe(self._loop.stop)
        self._thread.join()
        logger.debug("TaskManager shutdown complete.")
