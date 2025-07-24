"""
TaskManager module for managing task execution.
"""

from __future__ import annotations
import asyncio
import logging
import threading  # Keep this import
import traceback
from multiprocessing import get_context, Process
from multiprocessing.context import BaseContext
from multiprocessing.queues import Queue
from queue import Empty, Full
from typing import (
    Any,
    Callable,
    Coroutine,
    Dict,
    List,
    Optional,
)

from blinker import Signal

from ..util.glib import idle_add
from .context import ExecutionContext
from .task import Task


logger = logging.getLogger(__name__)


# This wrapper needs to be a top-level function to be pickleable by
# multiprocessing
def _process_target_wrapper(
    # The type of queue object will be determined by the multiprocessing
    # context.
    queue: Queue[tuple[str, Any]],
    user_func: Callable[..., Any],
    user_args: tuple[Any, ...],
    user_kwargs: dict[str, Any],
) -> None:
    """
    A wrapper that runs in the subprocess, calling the user's function
    and communicating status/results back to the parent via a queue.
    """
    from .context import ExecutionContextProxy

    proxy = ExecutionContextProxy(queue)
    try:
        result = user_func(proxy, *user_args, **user_kwargs)
        queue.put_nowait(("done", result))
    except Exception:
        error_info = traceback.format_exc()
        try:
            queue.put(("error", error_info), block=True, timeout=1.0)
        except Full:
            logger.error(
                f"Could not report exception to parent process:\n{error_info}"
            )


class TaskManager:
    def __init__(self) -> None:
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
        # Get the spawn context for safe subprocess creation
        self._mp_context: BaseContext = get_context("spawn")
        self._thread.start()

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
                logger.info(
                    f"TaskManager: Found existing task key '{task.key}'. "
                    f"Attempting cancellation."
                )
                old_task.cancel()
            else:
                logger.info(f"TaskManager: Adding new task key '{task.key}'.")

            self._tasks[task.key] = task
            self._progress_map[task.key] = 0.0
            task.status_changed.connect(self._on_task_updated)

            # Emit signal immediately when a new task is added
            self._emit_tasks_updated()

        asyncio.run_coroutine_threadsafe(
            self._run_task(task, when_done), self._loop
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
        task = Task(coro, *args, key=key, **kwargs)
        self.add_task(task, when_done)

    def run_process(
        self,
        func: Callable[..., Any],
        *args: Any,
        key: Optional[Any] = None,
        when_done: Optional[Callable[[Task], None]] = None,
        **kwargs: Any,
    ) -> None:
        task = Task(self._process_runner, func, *args, key=key, **kwargs)
        self.add_task(task, when_done)

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
        elif msg_type == "done":
            state["result"] = value
            logger.debug(
                "Task %s: Received 'done' from subprocess.", context.task.key
            )
        elif msg_type == "error":
            state["error"] = value
            logger.error(
                "Task %s: 'error' from subprocess:\n%s",
                context.task.key,
                value,
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

    async def _monitor_and_drain_queue(
        self,
        process: Process,
        queue: Queue[tuple[str, Any]],
        context: ExecutionContext,
        state: Dict[str, Any],
    ) -> None:
        """
        Monitor a subprocess and drain its queue until it exits.

        Args:
            process: The multiprocessing.Process to monitor.
            queue: The queue to drain.
            context: The ExecutionContext for progress reporting.
            state: A mutable dictionary to check for early error exit.
        """
        task_key = context.task.key
        while process.is_alive():
            self._drain_process_queue(queue, context, state)
            if state["error"]:
                logger.warning(
                    "Task %s: Error from subprocess, stopping monitor.",
                    task_key,
                )
                break
            await asyncio.sleep(0.1)

        logger.debug(
            "Task %s: Process %s ended. Final queue drain.",
            task_key,
            process.pid,
        )
        self._drain_process_queue(queue, context, state)

    def _check_process_result(
        self, process: Process, state: Dict[str, Any], task_key: Any
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
        self, process: Process, task_key: Any
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

    async def _process_runner(
        self,
        context: ExecutionContext,
        user_func: Callable[..., Any],
        *user_args: Any,
        **user_kwargs: Any,
    ) -> Any:
        """
        Runs a function in a separate process and monitors it.

        This coroutine creates and manages a subprocess, communicating with
        it via a queue to report progress, messages, results, and errors.
        It handles normal completion, failure, and cancellation.
        """
        task_key = context.task.key
        queue: Queue[tuple[str, Any]] = self._mp_context.Queue()
        process_args = (queue, user_func, user_args, user_kwargs)
        process: Process = self._mp_context.Process(
            target=_process_target_wrapper, args=process_args, daemon=True
        )
        # State dict to share status between helper methods.
        state: Dict[str, Any] = {"result": None, "error": None}

        try:
            process.start()
            logger.debug(
                "Task %s: Started subprocess with PID %s",
                task_key,
                process.pid,
            )

            await self._monitor_and_drain_queue(process, queue, context, state)

            self._check_process_result(process, state, task_key)

            logger.debug(
                "Task %s: Subprocess %s finished successfully.",
                task_key,
                process.pid,
            )
            return state["result"]
        except asyncio.CancelledError:
            logger.warning(
                "Task %s: Coroutine cancelled, cleaning up subprocess %s.",
                task_key,
                process.pid,
            )
            # The finally block handles the actual termination.
            raise
        finally:
            self._cleanup_process_resources(process, task_key)

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
            self._emit_tasks_updated()

    def _on_task_updated(self, task: Task) -> None:
        """Handle task status or progress changes."""
        with self._lock:  # <-- FIX: Protect shared state
            if task.key in self._progress_map:
                self._progress_map[task.key] = task.get_progress()
            self._emit_tasks_updated()

    def _emit_tasks_updated(self) -> None:
        """Emit a single consolidated signal. Assumes lock is already held."""
        # This method is now always called from a block that holds the lock
        progress = self.get_overall_progress()  # This will acquire the lock
        tasks: List[Task] = list(self._tasks.values())
        # Release the lock before calling idle_add
        idle_add(self.tasks_updated.send, self, tasks=tasks, progress=progress)

    def get_overall_progress(self) -> float:
        """Calculate the overall progress of all tasks."""
        with self._lock:  # <-- FIX: Protect shared state
            if not self._progress_map:
                return 1.0

            total_progress = sum(self._progress_map.values())
            return total_progress / len(self._progress_map)

    def shutdown(self) -> None:
        """Cancel all tasks and stop the event loop."""
        with self._lock:  # <-- FIX: Protect shared state
            for task in self._tasks.values():
                task.cancel()
            self._tasks.clear()
            self._progress_map.clear()
        self._loop.call_soon_threadsafe(self._loop.stop)
        self._thread.join()
