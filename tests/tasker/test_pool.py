import logging
import multiprocessing as mp
import threading
import time
import pytest
from pathlib import Path
from rayforge.shared.tasker.pool import WorkerPoolManager
from rayforge.shared.tasker.proxy import ExecutionContextProxy


# --- Test Target Functions ---
# These run in the worker processes.


def simple_add_func(proxy: ExecutionContextProxy, x: int, y: int) -> int:
    """A simple worker function that returns a result."""
    proxy.set_message("Starting addition")
    proxy.set_progress(0.5)
    time.sleep(0.05)
    proxy.set_progress(1.0)
    proxy.set_message("Finished addition")
    return x + y


def failing_func(proxy: ExecutionContextProxy):
    """A worker function that raises an exception."""
    proxy.set_message("About to fail")
    time.sleep(0.01)
    raise ValueError("This is an intentional failure.")


def event_sending_func(proxy: ExecutionContextProxy):
    """A worker function that sends a custom event."""
    proxy.send_event("custom_event", {"payload": 42})
    return "event_sent"


def worker_init(filepath: Path, init_queue: mp.Queue):
    """
    Initializer function that writes its PID to a file and signals completion.
    """
    # This function runs in the worker process.
    # We use a file-based side effect for the test to observe.
    import os

    pid = os.getpid()
    with open(filepath, "a") as f:
        f.write(f"{pid}\n")
    # Signal back to the main process that this worker has initialized.
    init_queue.put(pid)


# --- Fixtures ---


@pytest.fixture
def pool():
    """Provides a WorkerPoolManager instance that is properly shut down."""
    # Use 2 workers for concurrency testing.
    pool_manager = WorkerPoolManager(num_workers=2)
    yield pool_manager
    pool_manager.shutdown()


# --- Test Cases ---


class TestWorkerPoolManager:
    def test_submit_and_complete(self, pool: WorkerPoolManager):
        """Verify the happy path: submit a task and receive its result."""
        completion_event = threading.Event()
        result_holder = {}
        error_holder = {}

        def on_complete(sender, key, task_id, result):
            if key == "task1":
                result_holder["result"] = result
                completion_event.set()

        def on_fail(sender, key, task_id, error):
            if key == "task1":
                error_holder["error"] = error
                completion_event.set()

        pool.task_completed.connect(on_complete)
        pool.task_failed.connect(on_fail)

        pool.submit("task1", 123, simple_add_func, 10, 5)

        assert completion_event.wait(timeout=2), (
            "Task did not complete in time"
        )
        assert result_holder.get("result") == 15
        assert "error" not in error_holder

    def test_submit_and_fail(self, pool: WorkerPoolManager):
        """Verify that a failing task is reported correctly."""
        fail_event = threading.Event()
        result_holder = {}
        error_holder = {}

        def on_complete(sender, key, task_id, result):
            if key == "fail_task":
                result_holder["result"] = result
                fail_event.set()

        def on_fail(sender, key, task_id, error):
            if key == "fail_task":
                error_holder["error"] = error
                fail_event.set()

        pool.task_completed.connect(on_complete)
        pool.task_failed.connect(on_fail)

        pool.submit("fail_task", 456, failing_func)

        assert fail_event.wait(timeout=2), "Failing task was not reported"
        assert "result" not in result_holder
        assert "error" in error_holder
        assert (
            "ValueError: This is an intentional failure."
            in error_holder["error"]
        )

    def test_progress_and_message_updates(self, pool: WorkerPoolManager):
        """Verify that intermediate progress/message updates are received."""
        completion_event = threading.Event()
        progress_updates = []
        messages = []

        def on_progress(sender, key, task_id, progress):
            if key == "progress_task":
                progress_updates.append(progress)

        def on_message(sender, key, task_id, message):
            if key == "progress_task":
                messages.append(message)

        def on_complete(sender, key, task_id, result):
            if key == "progress_task":
                completion_event.set()

        pool.task_progress_updated.connect(on_progress)
        pool.task_message_updated.connect(on_message)
        pool.task_completed.connect(on_complete)

        pool.submit("progress_task", 789, simple_add_func, 1, 2)

        assert completion_event.wait(timeout=2)
        assert progress_updates == [0.5, 1.0]
        assert messages == ["Starting addition", "Finished addition"]

    def test_event_reporting(self, pool: WorkerPoolManager):
        """Verify that custom events are received from the worker."""
        event_received_event = threading.Event()
        received_event_data = {}

        def on_event(sender, key, task_id, event_name, data):
            if key == "event_task":
                received_event_data["name"] = event_name
                received_event_data["data"] = data
                event_received_event.set()

        pool.task_event_received.connect(on_event)
        pool.submit("event_task", 101, event_sending_func)

        assert event_received_event.wait(timeout=2)
        assert received_event_data["name"] == "custom_event"
        assert received_event_data["data"] == {"payload": 42}

    def test_multiple_concurrent_tasks(self, pool: WorkerPoolManager):
        """Verify that the pool can handle multiple tasks concurrently."""
        num_tasks = 10
        completion_events = {
            f"task_{i}": threading.Event() for i in range(num_tasks)
        }
        results = {}

        def on_complete(sender, key, task_id, result):
            results[key] = result
            if key in completion_events:
                completion_events[key].set()

        pool.task_completed.connect(on_complete)

        for i in range(num_tasks):
            pool.submit(f"task_{i}", 1000 + i, simple_add_func, i, i)

        all_completed = all(
            evt.wait(timeout=5) for evt in completion_events.values()
        )
        assert all_completed, "Not all tasks completed in time"

        assert len(results) == num_tasks
        for i in range(num_tasks):
            assert results[f"task_{i}"] == 2 * i

    def test_shutdown_terminates_workers(self, caplog):
        """Verify that shutdown stops workers and the listener thread."""
        caplog.set_level(logging.INFO)
        pool_manager = WorkerPoolManager(num_workers=2)
        # Give workers time to start
        time.sleep(0.2)

        initial_workers = list(pool_manager._workers)
        listener_thread = pool_manager._listener_thread
        assert all(w.is_alive() for w in initial_workers)
        assert listener_thread.is_alive()

        pool_manager.shutdown()

        # After shutdown, process objects are closed, so can't check is_alive()
        # Instead, we verify that the shutdown completed successfully
        # The actual process termination is verified by the shutdown logs
        for w in initial_workers:
            try:
                assert not w.is_alive()
            except ValueError:
                # Process object is closed, which is expected
                pass

        assert not listener_thread.is_alive()
        assert "Shutting down worker pool." in caplog.text
        assert "Worker pool shutdown complete." in caplog.text

    def test_worker_initializer(self, tmp_path):
        """Verify that the initializer is called in each worker process."""
        init_file = tmp_path / "init.log"
        num_workers = 3

        # Use the same multiprocessing context as the pool to create a queue
        ctx = mp.get_context("spawn")
        init_queue = ctx.Queue()

        # Create the pool with the initializer, passing the queue in initargs
        pool = WorkerPoolManager(
            num_workers=num_workers,
            initializer=worker_init,
            initargs=(init_file, init_queue),
        )

        # Wait deterministically for all workers to signal initialization.
        initialized_pids = set()
        for _ in range(num_workers):
            try:
                # Use a generous timeout for slow CI environments
                pid = init_queue.get(timeout=10)
                initialized_pids.add(pid)
            except Exception:
                pytest.fail(
                    "Timed out waiting for a worker to initialize. "
                    f"Expected {num_workers} workers, but only "
                    f"{len(initialized_pids)} signaled completion."
                )

        # Shut down the pool to ensure all processes are finished and files
        # are flushed.
        pool.shutdown()

        # Check the side effect of the initializer
        assert init_file.exists()
        content = init_file.read_text()
        # Filter out empty strings that can result from splitting an empty file
        lines = [line for line in content.strip().split("\n") if line]

        # There should be one line per worker process
        assert len(lines) == num_workers
        # All PIDs should be unique
        assert len(set(lines)) == num_workers
