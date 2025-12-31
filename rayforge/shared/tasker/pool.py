"""
Defines the WorkerPoolManager, a class for managing a pool of long-lived
worker processes to execute tasks efficiently.
"""

import logging
import os
import threading
import traceback
import builtins
from queue import Empty
from multiprocessing import get_context
from multiprocessing.process import BaseProcess
from multiprocessing.queues import Queue as MpQueue
from typing import Any, Callable, List, Set, Optional, Tuple
from blinker import Signal
from .proxy import ExecutionContextProxy

logger = logging.getLogger(__name__)

# A poison pill message to signal workers to shut down.
_WORKER_POISON_PILL = None
# A sentinel message to signal the result listener thread to shut down.
# Use a string for safe comparison across threads/processes.
_LISTENER_SENTINEL = "__listener_sentinel__"
# Message type for worker shutdown info
_SHUTDOWN_INFO_MSG = "__shutdown_info__"


class _TaggedQueue:
    """
    A wrapper around a multiprocessing queue that tags every message
    with a specific key before putting it on the underlying queue.

    This allows a shared result queue to distinguish which message belongs
    to which task. It respects the interface of ExecutionContextProxy, which
    expects an object with a `put_nowait` method.
    """

    def __init__(self, queue: MpQueue, key: Any, task_id: int):
        self._queue = queue
        self._key = key
        self._task_id = task_id

    def put_nowait(self, msg: tuple[str, Any]):
        """Tags the message with the key and puts it on the real queue."""
        msg_type, value = msg
        try:
            self._queue.put_nowait((self._key, self._task_id, msg_type, value))
        except Exception:
            # This can happen if the queue is closed during shutdown.
            # It's safe to ignore.
            pass


def _worker_main_loop(
    task_queue: MpQueue,
    result_queue: MpQueue,
    log_level: int,
    initializer: Optional[Callable[..., None]],
    initargs: Tuple[Any, ...],
):
    """
    The main function for a worker process.

    It continuously fetches tasks from the task_queue, executes them, and
    reports results, progress, and events back to the main process via the
    result_queue.
    """
    # Set up a null translator for gettext in the subprocess.
    if not hasattr(builtins, "_"):
        setattr(builtins, "_", lambda s: s)

    # Force reconfiguration of logging for this new process.
    root_logger = logging.getLogger()
    if root_logger.handlers:
        for handler in root_logger.handlers[:]:
            root_logger.removeHandler(handler)
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    worker_logger = logging.getLogger(__name__)

    if initializer is not None:
        try:
            initializer(*initargs)
        except Exception:
            # If initialization fails, report it and exit immediately.
            error_info = traceback.format_exc()
            worker_logger.critical(
                f"Worker {os.getpid()} failed during initialization:\n"
                f"{error_info}"
            )
            # We can't easily report this back via normal channels since
            # we don't have a task ID yet, so we log critical and die.
            return

    worker_logger.info(f"Worker process {os.getpid()} started and ready.")
    last_task_key = None

    while True:
        try:
            job = task_queue.get()
        except (EOFError, OSError):
            worker_logger.warning(
                f"Worker {os.getpid()}: Task queue connection lost. Exiting."
            )
            break
        except KeyboardInterrupt:
            # Gracefully exit if the worker is interrupted while waiting
            break

        if job is _WORKER_POISON_PILL:
            worker_logger.info(f"Worker {os.getpid()} received poison pill.")
            try:
                result_queue.put_nowait(
                    (
                        _SHUTDOWN_INFO_MSG,
                        0,
                        _SHUTDOWN_INFO_MSG,
                        (os.getpid(), last_task_key),
                    )
                )
            except (OSError, BrokenPipeError):
                pass
            break

        key, task_id, user_func, user_args, user_kwargs = job
        last_task_key = key
        worker_logger.debug(f"Worker {os.getpid()} starting task '{key}'.")

        # Wrap the result queue to automatically tag all messages from the
        # proxy with this task's unique key.
        tagged_queue = _TaggedQueue(result_queue, key, task_id)
        # The _TaggedQueue implements the necessary 'put_nowait' method
        # (duck typing), but isn't a Queue subclass. We ignore the type
        # checker warning here as the code is functionally correct.
        proxy = ExecutionContextProxy(
            tagged_queue,  # type: ignore
            parent_log_level=log_level,
        )

        try:
            result = user_func(proxy, *user_args, **user_kwargs)
            proxy.flush()  # Ensure the final progress is sent before "done"
            result_queue.put_nowait((key, task_id, "done", result))
        except Exception:
            error_info = traceback.format_exc()
            worker_logger.error(
                f"Worker {os.getpid()} task '{key}' failed:\n{error_info}"
            )
            # Also flush on error to send any last-known state
            proxy.flush()
            result_queue.put_nowait((key, task_id, "error", error_info))
        worker_logger.debug(f"Worker {os.getpid()} finished task '{key}'.")


class WorkerPoolManager:
    """
    Manages a pool of persistent worker processes to avoid the overhead of
    spawning a new process for every task.
    """

    def __init__(
        self,
        num_workers: int | None = None,
        initializer: Optional[Callable[..., None]] = None,
        initargs: Tuple[Any, ...] = (),
    ):
        if num_workers is None:
            num_workers = os.cpu_count() or 1
        logger.info(
            f"Initializing WorkerPoolManager with {num_workers} workers."
        )

        mp_context = get_context("spawn")
        self._task_queue: MpQueue = mp_context.Queue()
        self._result_queue: MpQueue = mp_context.Queue()
        self._workers: List[BaseProcess] = []
        self._cancelled_task_ids: Set[int] = set()
        self._lock = threading.Lock()
        self._worker_shutdown_info: dict[int, tuple[int, Any | None]] = {}

        # Signals for the TaskManager to subscribe to
        self.task_event_received = Signal()
        self.task_completed = Signal()
        self.task_failed = Signal()
        self.task_progress_updated = Signal()
        self.task_message_updated = Signal()

        log_level = logging.getLogger().getEffectiveLevel()

        for _ in range(num_workers):
            process = mp_context.Process(
                target=_worker_main_loop,
                # Pass initializer and initargs to the target function
                args=(
                    self._task_queue,
                    self._result_queue,
                    log_level,
                    initializer,
                    initargs,
                ),
                daemon=True,
            )
            self._workers.append(process)
            process.start()

        self._listener_thread = threading.Thread(
            target=self._result_listener_loop, daemon=True
        )
        self._listener_thread.start()

    def submit(
        self,
        key: Any,
        task_id: int,
        target: Callable[..., Any],
        *args: Any,
        **kwargs: Any,
    ) -> None:
        """
        Submits a task to the worker pool for execution.

        Args:
            key: A unique identifier for the task.
            task_id: The unique ID of the Task object instance.
            target: The function to execute in the worker process.
            *args: Positional arguments for the target function.
            **kwargs: Keyword arguments for the target function.
        """
        logger.debug(
            f"Submitting task '{key}' (id: {task_id}) to worker pool."
        )
        with self._lock:
            # Before submitting, remove the ID from the cancelled set in case
            # it's a retry of a previously cancelled task ID. This is unlikely
            # with UUIDs but good practice.
            self._cancelled_task_ids.discard(task_id)
        job = (key, task_id, target, args, kwargs)
        self._task_queue.put(job)

    def cancel(self, key: Any, task_id: int):
        """
        Registers a task ID as cancelled. The listener thread will ignore
        any subsequent messages from this task ID.
        """
        logger.debug(f"Registering task '{key}' (id: {task_id}) as cancelled.")
        with self._lock:
            self._cancelled_task_ids.add(task_id)

    def _result_listener_loop(self):
        """

        Runs in a dedicated thread in the main process, listening for results
        from all workers and dispatching them as signals.
        """
        logger.debug("Result listener thread started.")
        while True:
            try:
                message = self._result_queue.get(timeout=0.1)
            except (EOFError, OSError):
                logger.warning(
                    "Result queue connection lost. Exiting listener."
                )
                break
            except KeyboardInterrupt:
                # Gracefully exit if the listener is interrupted while waiting
                break
            except Empty:
                continue

            # Use '==' for value comparison, as 'is' fails for objects
            # passed through a queue.
            if message == _LISTENER_SENTINEL:
                logger.debug("Result listener thread received sentinel.")
                break

            key, task_id, msg_type, value = message

            if msg_type == _SHUTDOWN_INFO_MSG:
                pid, last_task_key = value
                with self._lock:
                    self._worker_shutdown_info[pid] = (pid, last_task_key)
                logger.debug(
                    f"Received shutdown info from worker {pid}: "
                    f"last_task={last_task_key}"
                )
                continue

            # The 'event' message type is special because it may carry
            # resource handles (like shared memory). These must ALWAYS be
            # forwarded to the TaskManager so the receiving code has a
            # chance to adopt the resource, even if the task was cancelled
            # or is stale. This prevents resource leaks.
            if msg_type == "event":
                event_name, data = value
                self.task_event_received.send(
                    self,
                    key=key,
                    task_id=task_id,
                    event_name=event_name,
                    data=data,
                )
                continue

            # For all other message types, we can safely ignore them if the
            # task has been cancelled.
            with self._lock:
                if task_id in self._cancelled_task_ids:
                    # For a cancelled task, only process the final 'done' or
                    # 'error' message for cleanup. Ignore everything else.
                    if msg_type in ("done", "error"):
                        # It's the final message. Let it pass through for
                        # cleanup and remove the ID from the cancelled set.
                        self._cancelled_task_ids.remove(task_id)
                    else:
                        # It's an intermediate message. Ignore it.
                        logger.debug(
                            f"Ignoring message '{msg_type}' from cancelled "
                            f"task '{key}' (id: {task_id})."
                        )
                        continue

            if msg_type == "done":
                self.task_completed.send(
                    self, key=key, task_id=task_id, result=value
                )
            elif msg_type == "error":
                self.task_failed.send(
                    self, key=key, task_id=task_id, error=value
                )
            elif msg_type == "progress":
                self.task_progress_updated.send(
                    self, key=key, task_id=task_id, progress=value
                )
            elif msg_type == "message":
                self.task_message_updated.send(
                    self, key=key, task_id=task_id, message=value
                )
        logger.debug("Result listener thread finished.")

    def shutdown(self, timeout: float = 2.0):
        """
        Shuts down the worker pool, terminating all worker processes.
        """
        logger.info("Shutting down worker pool.")
        try:
            for worker in self._workers:
                pid = worker.pid
                status = "alive" if worker.is_alive() else "dead"
                logger.info(f"Worker PID {pid}: {status}")

            # 1. Signal workers to exit by sending a poison pill for each one.
            for _ in self._workers:
                try:
                    self._task_queue.put(_WORKER_POISON_PILL)
                except (OSError, BrokenPipeError):
                    pass  # Queue may already be closed if workers crashed

            # 2. Join worker processes with a timeout.
            # Capture PIDs before closing workers for shutdown summary.
            worker_pids = [w.pid for w in self._workers]
            for worker in self._workers:
                worker.join(timeout=timeout)
                if worker.is_alive():
                    logger.warning(
                        f"Worker process {worker.pid} did not exit cleanly. "
                        "Terminating."
                    )
                    worker.terminate()
                    worker.join(timeout=1.0)
                # Always close the process object to properly clean up
                # and prevent zombie processes.
                try:
                    worker.close()
                except ValueError:
                    # Process might already be closed or in an invalid state
                    # This can happen if the process was already terminated
                    # and cleaned up by the OS
                    pass

            # 3. Stop the result listener thread.
            try:
                self._result_queue.put(_LISTENER_SENTINEL)
            except (OSError, BrokenPipeError):
                pass
            self._listener_thread.join(timeout=1.0)

            # 4. Clean up queues.
            self._task_queue.close()
            self._result_queue.close()
            # It's important to join the queue's feeder thread.
            self._task_queue.join_thread()
            self._result_queue.join_thread()

            logger.debug("Worker shutdown summary")
            for pid in worker_pids:
                if pid in self._worker_shutdown_info:
                    _, last_task_key = self._worker_shutdown_info[pid]
                    logger.info(
                        f"Worker PID {pid}: last_task='{last_task_key}'"
                    )
                else:
                    logger.warning(
                        f"Worker PID {pid}: no shutdown info received "
                        "(may have crashed or not reported)"
                    )
            logger.info("Worker pool shutdown complete.")
        except KeyboardInterrupt:
            logger.debug(
                "Worker pool shutdown interrupted by user. "
                "Suppressing traceback."
            )
            # At this point, the main process is exiting anyway.
            # The daemon processes will be terminated by the OS. We can just
            # pass and allow the exit to proceed cleanly.
            pass
