import asyncio
from asyncio.exceptions import CancelledError
from blinker import Signal
import logging
from typing import Optional, Callable, Coroutine
import threading
import traceback
from multiprocessing import get_context
from queue import Empty, Full
from .util.glib import idle_add


logger = logging.getLogger(__name__)


# This wrapper needs to be a top-level function to be pickleable by
# multiprocessing
def _process_target_wrapper(
    # The type of queue object will be determined by the multiprocessing
    # context.
    queue,
    user_func: Callable,
    user_args: tuple,
    user_kwargs: dict,
):
    """
    A wrapper that runs in the subprocess, calling the user's function
    and communicating status/results back to the parent via a queue.
    """
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


class ExecutionContextProxy:
    """
    A pickleable proxy for reporting progress from a subprocess via a queue.
    """

    def __init__(self, progress_queue, base_progress=0.0, progress_range=1.0):
        self._queue = progress_queue
        self._base = base_progress
        self._range = progress_range
        self._total = 1.0  # Default total for normalization

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
        # Get the spawn context for safe subprocess creation
        self._mp_context = get_context("spawn")
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
        self._progress_map[task.key] = 0.0
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

    def run_process(
        self,
        func: Callable,
        *args,
        key=None,
        when_done: Optional[Callable] = None,
        **kwargs,
    ):
        task = Task(self._process_runner, func, *args, key=key, **kwargs)
        self.add_task(task, when_done)

    async def _run_task(self, task: Task, when_done: Optional[Callable]):
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
            self._emit_tasks_updated()
            if when_done:
                idle_add(when_done, task)

    def _handle_process_queue_message(self, msg, context, state):
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
            logger.debug("Task %s: Received 'done' from subprocess.",
                         context.task.key)
        elif msg_type == "error":
            state["error"] = value
            logger.error("Task %s: 'error' from subprocess:\n%s",
                         context.task.key, value)

    def _drain_process_queue(self, queue, context, state):
        """Drain all pending messages from the subprocess queue."""
        try:
            while True:
                msg = queue.get_nowait()
                self._handle_process_queue_message(msg, context, state)
        except Empty:
            pass

    async def _monitor_and_drain_queue(self, process, queue, context, state):
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
                    task_key
                )
                break
            await asyncio.sleep(0.1)

        logger.debug("Task %s: Process %s ended. Final queue drain.",
                     task_key, process.pid)
        self._drain_process_queue(queue, context, state)

    def _check_process_result(self, process, state, task_key):
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

    def _cleanup_process_resources(self, process, task_key):
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
                    task_key, process.pid
                )
                process.kill()
                process.join(timeout=1.0)

        process.close()
        logger.debug("Task %s: Subprocess resources cleaned up.", task_key)

    async def _process_runner(
        self,
        context: ExecutionContext,
        user_func: Callable,
        *user_args,
        **user_kwargs,
    ):
        """
        Runs a function in a separate process and monitors it.

        This coroutine creates and manages a subprocess, communicating with
        it via a queue to report progress, messages, results, and errors.
        It handles normal completion, failure, and cancellation.
        """
        task_key = context.task.key
        queue = self._mp_context.Queue()
        process_args = (queue, user_func, user_args, user_kwargs)
        process = self._mp_context.Process(
            target=_process_target_wrapper, args=process_args, daemon=True
        )
        # State dict to share status between helper methods.
        state = {"result": None, "error": None}

        try:
            process.start()
            logger.debug(
                "Task %s: Started subprocess with PID %s",
                task_key, process.pid
            )

            await self._monitor_and_drain_queue(
                process, queue, context, state
            )

            self._check_process_result(process, state, task_key)

            logger.debug(
                "Task %s: Subprocess %s finished successfully.",
                task_key, process.pid
            )
            return state["result"]
        except asyncio.CancelledError:
            logger.warning(
                "Task %s: Coroutine cancelled, cleaning up subprocess %s.",
                task_key, process.pid
            )
            # The finally block handles the actual termination.
            raise
        finally:
            self._cleanup_process_resources(process, task_key)

    def _cleanup_task(self, task: Task):
        """
        Clean up a completed task.
        """
        current_task_in_dict = self._tasks.get(task.key)
        if current_task_in_dict is task:
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
        idle_add(self.tasks_updated.send, self, tasks=tasks, progress=progress)

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
