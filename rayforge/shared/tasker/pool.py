"""
Defines the WorkerPoolManager, a class for managing a pool of long-lived
worker processes to execute tasks efficiently.
"""

import logging
import os
import threading
import traceback
import builtins
from multiprocessing import get_context
from multiprocessing.process import BaseProcess
from multiprocessing.queues import Queue as MpQueue
from typing import Any, Callable, List
from blinker import Signal
from .proxy import ExecutionContextProxy

logger = logging.getLogger(__name__)

# A poison pill message to signal workers to shut down.
_WORKER_POISON_PILL = None
# A sentinel message to signal the result listener thread to shut down.
# Use a string for safe comparison across threads/processes.
_LISTENER_SENTINEL = "__listener_sentinel__"


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
    task_queue: MpQueue, result_queue: MpQueue, log_level: int
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
    worker_logger.info(f"Worker process {os.getpid()} started.")

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
            break

        key, task_id, user_func, user_args, user_kwargs = job
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
            result_queue.put_nowait((key, task_id, "done", result))
        except Exception:
            error_info = traceback.format_exc()
            worker_logger.error(
                f"Worker {os.getpid()} task '{key}' failed:\n{error_info}"
            )
            result_queue.put_nowait((key, task_id, "error", error_info))
        worker_logger.debug(f"Worker {os.getpid()} finished task '{key}'.")


class WorkerPoolManager:
    """
    Manages a pool of persistent worker processes to avoid the overhead of
    spawning a new process for every task.
    """

    def __init__(self, num_workers: int | None = None):
        if num_workers is None:
            num_workers = os.cpu_count() or 1
        logger.info(
            f"Initializing WorkerPoolManager with {num_workers} workers."
        )

        mp_context = get_context("spawn")
        self._task_queue: MpQueue = mp_context.Queue()
        self._result_queue: MpQueue = mp_context.Queue()
        # Use a static type hint that the linter can understand.
        self._workers: List[BaseProcess] = []

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
                args=(self._task_queue, self._result_queue, log_level),
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
        logger.debug(f"Submitting task '{key}' to worker pool.")
        job = (key, task_id, target, args, kwargs)
        self._task_queue.put(job)

    def _result_listener_loop(self):
        """

        Runs in a dedicated thread in the main process, listening for results
        from all workers and dispatching them as signals.
        """
        logger.debug("Result listener thread started.")
        while True:
            try:
                message = self._result_queue.get()
            except (EOFError, OSError):
                logger.warning(
                    "Result queue connection lost. Exiting listener."
                )
                break
            except KeyboardInterrupt:
                # Gracefully exit if the listener is interrupted while waiting
                break

            # Use '==' for value comparison, as 'is' fails for objects
            # passed through a queue.
            if message == _LISTENER_SENTINEL:
                logger.debug("Result listener thread received sentinel.")
                break

            key, task_id, msg_type, value = message
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
            elif msg_type == "event":
                event_name, data = value
                self.task_event_received.send(
                    self,
                    key=key,
                    task_id=task_id,
                    event_name=event_name,
                    data=data,
                )
        logger.debug("Result listener thread finished.")

    def shutdown(self, timeout: float = 2.0):
        """
        Shuts down the worker pool, terminating all worker processes.
        """
        logger.info("Shutting down worker pool.")
        try:
            # 1. Signal workers to exit by sending a poison pill for each one.
            for _ in self._workers:
                try:
                    self._task_queue.put(_WORKER_POISON_PILL)
                except (OSError, BrokenPipeError):
                    pass  # Queue may already be closed if workers crashed

            # 2. Join worker processes with a timeout.
            for worker in self._workers:
                worker.join(timeout=timeout)
                if worker.is_alive():
                    logger.warning(
                        f"Worker process {worker.pid} did not exit cleanly. "
                        "Terminating."
                    )
                    worker.terminate()
                    worker.join(timeout=1.0)
                # Do not call worker.close() here. It makes the process object
                # unusable for tests that need to check its final state (e.g.,
                # is_alive()), and daemon processes are cleaned up anyway.

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
