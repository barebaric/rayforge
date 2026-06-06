"""
Defines the WorkerPoolManager, a class for managing a pool of long-lived
worker processes to execute tasks efficiently.
"""

from __future__ import annotations

import builtins
import logging
import os
import threading
import time
import traceback
from multiprocessing import Manager, get_context
from multiprocessing.managers import DictProxy
from multiprocessing.process import BaseProcess
from multiprocessing.queues import Queue as MpQueue
from queue import Empty
from typing import Any, Callable, List, Optional, Set, Tuple

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
    adoption_signals: DictProxy[str, bool],
    shared_state: DictProxy[str, Any],
):
    """
    The main function for a worker process.

    It continuously fetches tasks from the task_queue, executes them, and
    reports results, progress, and events back to the main process via the
    result_queue.
    """
    worker_logger = logging.getLogger(__name__)
    worker_logger.debug(
        f"Worker {os.getpid()} shared_state keys: {list(shared_state.keys())}"
    )
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
            initializer(shared_state, *initargs)
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
            worker_logger.debug(
                f"Worker {os.getpid()}: Getting job from task_queue..."
            )
            job = task_queue.get()
            worker_logger.debug(
                f"Worker {os.getpid()}: Got job from task_queue"
            )
        except (EOFError, OSError) as e:
            worker_logger.error(
                f"Worker {os.getpid()}: Task queue connection lost. "
                f"Exception type: {type(e).__name__}, Exception: {e}"
            )
            worker_logger.error(
                f"Worker {os.getpid()}: Last task was: {last_task_key}"
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
                logger.warning(
                    f"Worker {os.getpid()}: "
                    f"Failed to send shutdown info via result queue. "
                    f"Queue may be closed."
                )
            break

        key, task_id, user_func, user_args, user_kwargs = job
        last_task_key = key
        worker_logger.debug(f"Worker {os.getpid()} starting task '{key}'.")
        worker_logger.info(
            f"[DIAGNOSTIC] Worker {os.getpid()} task_id={task_id}, "
            f"key={key}, last_task_key={last_task_key}"
        )

        cancel_key = f"cancel:{task_id}"
        if cancel_key in adoption_signals:
            worker_logger.debug(
                f"Worker {os.getpid()} skipping already-cancelled "
                f"task '{key}' (id: {task_id})."
            )
            try:
                result_queue.put_nowait(
                    (key, task_id, "done", None)
                )
            except (OSError, BrokenPipeError):
                pass
            adoption_signals.pop(cancel_key, None)
            continue

        # Track this task via the DictProxy BEFORE running user_func.
        # This is the sole mechanism for identifying orphaned tasks when
        # a worker crashes — it uses the SyncManager's own connection,
        # which is immune to POSIX semaphore corruption from a crashed
        # peer's _feed thread. Unlike the result queue, a worker crash
        # merely closes the DictProxy socket; the shared dict remains
        # intact and the health check can read the orphaned task info.
        try:
            shared_state[f"_wpool:{os.getpid()}"] = (key, task_id)
        except (OSError, BrokenPipeError):
            pass

        try:
            result_queue.put_nowait(
                (key, task_id, "running", os.getpid())
            )
        except (OSError, BrokenPipeError):
            pass

        # Wrap the result queue to automatically tag all messages from the
        # proxy with this task's unique key.
        tagged_queue = _TaggedQueue(result_queue, key, task_id)
        # The _TaggedQueue implements the necessary 'put_nowait' method
        # (duck typing), but isn't a Queue subclass. We ignore the type
        # checker warning here as the code is functionally correct.
        proxy = ExecutionContextProxy(
            tagged_queue,  # type: ignore
            parent_log_level=log_level,
            adoption_signals=adoption_signals,
            task_id=task_id,
        )

        try:
            result = user_func(proxy, *user_args, **user_kwargs)
            proxy.flush()  # Ensure the final progress is sent before "done"
            result_queue.put_nowait((key, task_id, "done", result))
            # Clean up the DictProxy entry ONLY after the result was
            # successfully sent. If this line is not reached (worker
            # crashes), the entry remains so the health check can detect
            # the orphaned task.
            shared_state.pop(f"_wpool:{os.getpid()}", None)
        except Exception:
            error_info = traceback.format_exc()
            worker_logger.error(
                f"Worker {os.getpid()} task '{key}' failed:\n{error_info}"
            )
            # Also flush on error to send any last-known state
            proxy.flush()
            try:
                result_queue.put_nowait((key, task_id, "error", error_info))
                # Clean up ONLY after error was successfully reported.
                # If this raises (worker crashes), entry stays for
                # health check detection.
                shared_state.pop(f"_wpool:{os.getpid()}", None)
            except Exception:
                # Couldn't send error either. Worker will exit and
                # the DictProxy entry stays for the health check.
                raise
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
        shared_state: Optional[DictProxy[str, Any]] = None,
    ):
        if num_workers is None:
            env_max = os.environ.get("RAYFORGE_MAX_WORKERS")
            if env_max is not None:
                num_workers = min(int(env_max), os.cpu_count() or 1)
            else:
                num_workers = os.cpu_count() or 1
        logger.info(
            f"Initializing WorkerPoolManager with {num_workers} workers."
        )

        self._mp_context = get_context("spawn")
        self._manager = Manager()
        self._task_queue: MpQueue = self._mp_context.Queue()
        self._result_queue: MpQueue = self._mp_context.Queue()
        self._adoption_signals = self._manager.dict()
        if shared_state is not None:
            self._shared_state = shared_state
        else:
            self._shared_state = self._manager.dict()
        self._workers: List[BaseProcess] = []
        self._cancelled_task_ids: Set[int] = set()
        self._lock = threading.Lock()
        self._worker_shutdown_info: dict[int, tuple[int, Any | None]] = {}
        self._worker_task_map: dict[int, Tuple[Any, int]] = {}
        self._worker_start_time: dict[int, float] = {}
        self._pid_to_worker: dict[int, BaseProcess] = {}
        self._health_check_counter = 0
        self._last_result_time = time.monotonic()

        # Signals for the TaskManager to subscribe to
        self.task_event_received = Signal()
        self.task_completed = Signal()
        self.task_failed = Signal()
        self.task_progress_updated = Signal()
        self.task_message_updated = Signal()
        self.worker_died = Signal()

        log_level = logging.getLogger().getEffectiveLevel()

        # Store worker creation params for replacement workers
        self._log_level = log_level
        self._initializer = initializer
        self._initargs = initargs

        for _ in range(num_workers):
            process = self._mp_context.Process(
                target=_worker_main_loop,
                args=(
                    self._task_queue,
                    self._result_queue,
                    log_level,
                    initializer,
                    initargs,
                    self._adoption_signals,
                    self._shared_state,
                ),
                daemon=True,
            )
            self._workers.append(process)
            process.start()
            assert process.pid is not None
            self._pid_to_worker[process.pid] = process

        self._listener_thread = threading.Thread(
            target=self._result_listener_loop, daemon=True
        )
        self._listener_thread.start()

    def get_shared_state(self) -> Any:
        """
        Return the shared state dict for worker initialization.

        This provides a generic mechanism for passing data to worker
        processes. Callers can populate this dict with any data needed
        during worker initialization.
        """
        return self._shared_state

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
        logger.debug(
            f"Putting job on task_queue: key={key}, task_id={task_id}"
        )
        try:
            self._task_queue.put(job)
        except (OSError, BrokenPipeError) as e:
            logger.error(
                f"Failed to put job on task_queue: {e}. "
                f"Task queue may be closed or corrupted."
            )
            raise

    def cancel(self, key: Any, task_id: int):
        """
        Registers a task ID as cancelled. The listener thread will ignore
        any subsequent messages from this task ID. Also sets a flag in
        shared adoption_signals so the worker subprocess can cooperatively
        abort via ExecutionContextProxy.is_cancelled().
        """
        logger.debug(f"Registering task '{key}' (id: {task_id}) as cancelled.")
        with self._lock:
            self._cancelled_task_ids.add(task_id)
        self._adoption_signals[f"cancel:{task_id}"] = True

    def _result_listener_loop(self):
        """

        Runs in a dedicated thread in the main process, listening for results
        from all workers and dispatching them as signals.
        """
        logger.debug("Result listener thread started.")
        while True:
            self._health_check_counter += 1
            if self._health_check_counter % 10 == 0:
                self._check_worker_health()

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
            self._last_result_time = time.monotonic()

            if msg_type == _SHUTDOWN_INFO_MSG:
                pid, last_task_key = value
                with self._lock:
                    self._worker_shutdown_info[pid] = (pid, last_task_key)
                logger.debug(
                    f"Received shutdown info from worker {pid}: "
                    f"last_task={last_task_key}"
                )
                continue

            # Track which worker is processing which task via the result
            # queue. The primary tracking mechanism is the DictProxy
            # (_wpool:pid), but _worker_task_map serves as a secondary
            # source for workers that successfully sent "running" before
            # a potential queue stall or peer crash.
            if msg_type == "running":
                pid = value
                with self._lock:
                    self._worker_task_map[pid] = (key, task_id)
                    self._worker_start_time[pid] = time.monotonic()
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
                    adoption_signals=self._adoption_signals,
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
                        self._adoption_signals.pop(f"cancel:{task_id}", None)
                    else:
                        # It's an intermediate message. Ignore it.
                        logger.debug(
                            f"Ignoring message '{msg_type}' from cancelled "
                            f"task '{key}' (id: {task_id})."
                        )
                        continue

            if msg_type == "done":
                self._adoption_signals.pop(f"cancel:{task_id}", None)
                self.task_completed.send(
                    self, key=key, task_id=task_id, result=value
                )
            elif msg_type == "error":
                self._adoption_signals.pop(f"cancel:{task_id}", None)
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

            # Clean up the worker task mapping when a task finishes.
            if msg_type in ("done", "error"):
                with self._lock:
                    for pid, (k, tid) in list(self._worker_task_map.items()):
                        if k == key and tid == task_id:
                            del self._worker_task_map[pid]
                            self._worker_start_time.pop(pid, None)
                            break

        logger.debug("Result listener thread finished.")

    def _check_worker_health(self):
        """
        Check if any workers that are currently running a task have died.
        If so, emit a worker_died signal for orphaned task cleanup and start
        a replacement worker.

        Also detects workers that are alive but whose results are stuck
        because the result queue's ``_wlock`` semaphore was poisoned by
        a crashed peer. When this is detected, ALL stuck workers are
        terminated and restarted with a fresh result queue.
        """
        dead_info = []
        stuck_info = []
        with self._lock:
            # Check all workers via _worker_task_map (result queue path).
            for pid, (key, task_id) in list(self._worker_task_map.items()):
                worker = self._pid_to_worker.get(pid)
                if worker is not None:
                    try:
                        alive = worker.is_alive()
                    except ValueError:
                        alive = False
                else:
                    alive = False
                if not alive:
                    dead_info.append((pid, key, task_id, worker))
                    del self._worker_task_map[pid]
                    if pid in self._pid_to_worker:
                        del self._pid_to_worker[pid]
                    try:
                        if worker is not None:
                            self._workers.remove(worker)
                    except (ValueError, AttributeError):
                        pass

            # Check all remaining workers via DictProxy.
            for pid, worker in list(self._pid_to_worker.items()):
                if pid in self._worker_task_map:
                    continue
                try:
                    alive = worker.is_alive()
                except ValueError:
                    alive = False

                if not alive:
                    status = self._shared_state.get(f"_wpool:{pid}")
                    if status is not None:
                        key, task_id = status
                        dead_info.append((pid, key, task_id, worker))
                    else:
                        dead_info.append((pid, None, None, worker))
                    if pid in self._pid_to_worker:
                        del self._pid_to_worker[pid]
                    try:
                        self._workers.remove(worker)
                    except ValueError:
                        pass
                else:
                    # Alive worker — check if it has a stuck task.
                    status = self._shared_state.get(f"_wpool:{pid}")
                    if status is not None:
                        stuck_info.append((pid, worker))

        # Log diagnostic state when anomalies are found.
        if dead_info:
            no_result_dur = time.monotonic() - self._last_result_time
            logger.info(
                f"Health check: {len(dead_info)} dead, "
                f"{len(stuck_info)} stuck, "
                f"no_result_for={no_result_dur:.1f}s"
            )

        # Emit signals for dead workers with orphaned tasks.
        for pid, key, task_id, worker in dead_info:
            if key is not None:
                logger.warning(
                    f"Worker PID {pid} died while processing task "
                    f"'{key}' (id: {task_id}). "
                    f"Orphaned task will be marked as failed."
                )
            else:
                logger.warning(
                    f"Worker PID {pid} died while idle."
                )
            try:
                worker.close()
            except ValueError:
                pass
            if key is not None:
                self.worker_died.send(
                    self, key=key, task_id=task_id, pid=pid
                )

        if dead_info:
            for _ in dead_info:
                self._spawn_replacement_worker()

        # Detect poisoned result queue: if any alive workers have
        # DictProxy entries (meaning they're running tasks) but no
        # results have been received recently, the queue is broken.
        # Terminate stuck workers and restart.
        if stuck_info:
            no_result_duration = time.monotonic() - self._last_result_time
            if no_result_duration > 3.0:
                logger.warning(
                    f"Result queue appears poisoned (no results for "
                    f"{no_result_duration:.1f}s, "
                    f"{len(stuck_info)} stuck workers). "
                    f"Restarting all stuck workers."
                )
                self._restart_stuck_workers(stuck_info)
                return

        # Per-worker timeout: even if the global result queue is
        # healthy (other workers producing results), an individual
        # worker may have crashed without being detected by
        # is_alive() (e.g., a segfault in a C extension that
        # leaves the process technically alive but unresponsive).
        # Check how long each worker has been on its current task.
        max_task_duration = 30.0
        timed_out = []
        with self._lock:
            for pid in list(self._worker_start_time.keys()):
                start = self._worker_start_time.get(pid)
                if start is None:
                    continue
                elapsed = time.monotonic() - start
                if elapsed > max_task_duration:
                    worker = self._pid_to_worker.get(pid)
                    if worker is not None:
                        try:
                            alive = worker.is_alive()
                        except ValueError:
                            alive = False
                        if alive:
                            status = self._shared_state.get(
                                f"_wpool:{pid}"
                            )
                            key, task_id = (
                                status if status else (None, None)
                            )
                            timed_out.append(
                                (pid, worker, key, task_id, elapsed)
                            )

        for pid, worker, key, task_id, elapsed in timed_out:
            logger.warning(
                f"Worker PID {pid} has been running task "
                f"'{key}' (id: {task_id}) for {elapsed:.0f}s. "
                f"Terminating as unresponsive."
            )
            self._terminate_stuck_worker(pid, worker, key, task_id)

    def _terminate_stuck_worker(
        self,
        pid: int,
        worker: BaseProcess,
        key: Any,
        task_id: Any,
    ):
        """
        Terminate a single unresponsive worker, finalize its orphaned
        task, and spawn a replacement.
        """
        try:
            worker.terminate()
            worker.join(timeout=2.0)
            worker.close()
        except (ValueError, OSError):
            pass

        with self._lock:
            self._pid_to_worker.pop(pid, None)
            self._worker_task_map.pop(pid, None)
            self._worker_start_time.pop(pid, None)
            try:
                self._workers.remove(worker)
            except ValueError:
                pass

        self._shared_state.pop(f"_wpool:{pid}", None)

        if key is not None:
            self.worker_died.send(
                self, key=key, task_id=task_id, pid=pid
            )

        self._spawn_replacement_worker()

    def _restart_stuck_workers(self, stuck_info):
        """
        Terminate stuck workers (alive but can't deliver results due to
        a poisoned result queue), finalize their orphaned tasks, and
        restart the pool with a fresh result queue.
        """
        # Kill stuck workers and emit worker_died for their tasks.
        for pid, worker in stuck_info:
            status = self._shared_state.get(f"_wpool:{pid}")
            if status is not None:
                key, task_id = status
                logger.warning(
                    f"Terminating stuck worker PID {pid} "
                    f"(task '{key}', id: {task_id})."
                )
                try:
                    worker.terminate()
                    worker.join(timeout=2.0)
                    worker.close()
                except (ValueError, OSError):
                    pass
                self.worker_died.send(
                    self, key=key, task_id=task_id, pid=pid
                )

        with self._lock:
            for pid, worker in stuck_info:
                if pid in self._pid_to_worker:
                    del self._pid_to_worker[pid]
                try:
                    self._workers.remove(worker)
                except ValueError:
                    pass

        for _ in stuck_info:
            self._spawn_replacement_worker()

    def _spawn_replacement_worker(self):
        """Start a single replacement worker process."""
        process = self._mp_context.Process(
            target=_worker_main_loop,
            args=(
                self._task_queue,
                self._result_queue,
                self._log_level,
                self._initializer,
                self._initargs,
                self._adoption_signals,
                self._shared_state,
            ),
            daemon=True,
        )
        process.start()
        assert process.pid is not None
        with self._lock:
            self._workers.append(process)
            self._pid_to_worker[process.pid] = process
        logger.info(
            f"Replacement worker PID {process.pid} started."
        )

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
                except (OSError, BrokenPipeError) as e:
                    logger.warning(
                        f"Failed to send poison pill to worker: {e}. "
                        "Queue may already be closed if workers crashed."
                    )

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
            except (OSError, BrokenPipeError) as e:
                logger.warning(
                    f"Failed to send sentinel to listener: {e}. "
                    "Result queue may already be closed."
                )
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
