import asyncio
import threading
import time
from unittest.mock import Mock
import pytest
from rayforge.tasker import (
    TaskManager,
    Task,
    ExecutionContext,
)
from rayforge.tasker.proxy import ExecutionContextProxy


async def simple_coro(
    context: ExecutionContext, duration=0.2, result="coro_done"
):
    """A simple coroutine that reports progress and completes."""
    context.set_total(10)
    for i in range(10):
        if context.is_cancelled():
            raise asyncio.CancelledError()
        context.set_progress(i + 1)
        context.set_message(f"Step {i + 1}")
        await asyncio.sleep(duration / 10)
    return result


async def controllable_coro(
    context: ExecutionContext,
    name: str,
    steps: int,
    proceed_events: list[threading.Event],
    done_events: list[threading.Event],
):
    """
    A coroutine controlled by thread-safe events, allowing deterministic
    testing across different event loops.
    """
    context.set_total(steps)
    for i in range(steps):
        # 1. Asynchronously wait for the thread-safe event without blocking
        # the loop
        await asyncio.to_thread(proceed_events[i].wait)

        # 2. Perform the work for this step
        context.set_progress(i + 1)
        context.set_message(f"{name} step {i+1}")

        # 3. Signal back to the test (this is a thread-safe call)
        done_events[i].set()


async def failing_coro(context: ExecutionContext):
    """A coroutine that intentionally fails."""
    await asyncio.sleep(0.01)
    raise ValueError("Coroutine failed intentionally")


async def cancellable_coro(
    context: ExecutionContext, started_event: threading.Event
):
    """A long-running coroutine that can be cancelled."""
    started_event.set()
    for _ in range(100):  # Run for up to 10 seconds
        if context.is_cancelled():
            raise asyncio.CancelledError()
        await asyncio.sleep(0.1)
    # This part should not be reached in the cancellation test
    pytest.fail("Cancellable coroutine was not cancelled.")


def simple_process_func(
    context: ExecutionContextProxy, duration=0.3, result="process_done"
):
    """A simple function for a subprocess that reports progress."""
    context.set_total(10)
    for i in range(10):
        context.set_progress(i + 1)
        context.set_message(f"Process step {i + 1}")
        time.sleep(duration / 10)
    return result


def failing_process_func(context: ExecutionContextProxy):
    """A process function that intentionally fails."""
    time.sleep(0.01)
    raise ValueError("Process failed intentionally")


def long_running_process_func(
    context: ExecutionContextProxy, started_event: threading.Event
):
    """A long-running process function to test termination."""
    started_event.set()
    # This process doesn't check for cancellation. We rely on
    # the manager terminating it.
    time.sleep(10)
    return "should_not_return"


def exit_code_process_func(context: ExecutionContextProxy):
    """A process function that exits with a non-zero status code."""
    time.sleep(0.1)
    # Note: sys.exit() raises SystemExit, which is caught by the wrapper.
    # To test the exit code, the process must exit more forcefully or the
    # wrapper would need to be different. However, the manager's check
    # for `process.exitcode != 0` is a fallback, and we can test it this
    # way.
    import os

    os._exit(5)  # Force exit without cleanup or exception


@pytest.fixture
def manager():
    """Provides a TaskManager instance that is properly shut down after use."""
    tm = TaskManager()
    yield tm
    # Shutdown ensures the background thread is properly terminated
    tm.shutdown()


@pytest.fixture(autouse=True)
def patch_idle_add(monkeypatch):
    """
    Replaces glib.idle_add with an immediate-execution mock.
    This is necessary because the tests don't run a GLib main loop.
    It allows us to test that callbacks are called with the correct arguments.
    """

    def mock_idle_add(callback, *args, **kwargs):
        return callback(*args, **kwargs)

    # Patch in both modules where it is imported
    monkeypatch.setattr("rayforge.tasker.manager.idle_add", mock_idle_add)
    monkeypatch.setattr("rayforge.tasker.context.idle_add", mock_idle_add)


class ControllableTimer:
    """A mock Timer class that can be manually controlled."""

    def __init__(self, interval, function, args=None, kwargs=None):
        self.interval = interval
        self.function = function
        self.args = args or []
        self.kwargs = kwargs or {}
        self.is_started = False
        self.is_cancelled = False

    def start(self):
        self.is_started = True

    def cancel(self):
        self.is_cancelled = True

    def fire(self):
        """Manually trigger the timer's function."""
        if self.is_started and not self.is_cancelled:
            self.function(*self.args, **self.kwargs)


@pytest.fixture
def mock_timer_factory(mocker):
    """
    Mocks threading.Timer to make the ExecutionContext's debounce
    mechanism deterministic and controllable.
    """
    timers = []

    def factory(*args, **kwargs):
        timer = ControllableTimer(*args, **kwargs)
        timers.append(timer)
        return timer

    mocker.patch("rayforge.tasker.context.threading.Timer", factory)
    return timers


class TestCoroutineTasks:
    """Tests for tasks running as asyncio coroutines."""

    def test_add_and_complete_coroutine(self, manager: TaskManager):
        """Verify the happy path for a coroutine task."""
        completion_event = threading.Event()
        final_task = None

        def on_done(task: Task):
            nonlocal final_task
            final_task = task
            completion_event.set()

        manager.add_coroutine(simple_coro, key="test1", when_done=on_done)
        assert "test1" in manager._tasks

        assert completion_event.wait(timeout=2), (
            "Task did not complete in time"
        )

        assert final_task is not None
        assert final_task.key == "test1"
        assert final_task.get_status() == "completed"
        assert final_task.get_progress() == 1.0
        assert final_task.get_message() == "Step 10"
        assert final_task.result() == "coro_done"
        assert not manager._tasks, "Task was not cleaned up from manager"

    def test_coroutine_failure(self, manager: TaskManager):
        """Verify that a failing coroutine is handled correctly."""
        completion_event = threading.Event()
        final_task = None

        def on_done(task: Task):
            nonlocal final_task
            final_task = task
            completion_event.set()

        manager.add_coroutine(failing_coro, key="fail1", when_done=on_done)
        assert completion_event.wait(timeout=2)

        assert final_task is not None
        assert final_task.get_status() == "failed"
        with pytest.raises(ValueError, match="Coroutine failed intentionally"):
            final_task.result()
        assert not manager._tasks

    def test_coroutine_cancellation(self, manager: TaskManager):
        """Verify that a running coroutine can be cancelled."""
        completion_event = threading.Event()
        started_event = threading.Event()
        final_task = None

        def on_done(task: Task):
            nonlocal final_task
            final_task = task
            completion_event.set()

        manager.add_coroutine(
            cancellable_coro, started_event, key="cancel_me", when_done=on_done
        )
        task = manager._tasks["cancel_me"]

        assert started_event.wait(timeout=1), "Coroutine did not start in time"
        task.cancel()

        assert completion_event.wait(timeout=2), (
            "on_done was not called after cancel"
        )
        assert final_task is not None
        assert final_task.get_status() == "canceled"
        with pytest.raises(asyncio.CancelledError):
            final_task.result()
        assert not manager._tasks

    def test_task_replacement(self, manager: TaskManager):
        """Verify that adding a task with an existing key cancels the old one."""
        first_task_done = threading.Event()
        second_task_done = threading.Event()
        started_event = threading.Event()

        first_task_final_status = None

        def on_done_first(task: Task):
            nonlocal first_task_final_status
            first_task_final_status = task.get_status()
            first_task_done.set()

        second_task_final_status = None

        def on_done_second(task: Task):
            nonlocal second_task_final_status
            second_task_final_status = task.get_status()
            second_task_done.set()

        # Start a long-running task
        manager.add_coroutine(
            cancellable_coro,
            started_event,
            key="shared_key",
            when_done=on_done_first,
        )
        assert started_event.wait(timeout=1)

        # Add a new task with the same key
        manager.add_coroutine(
            simple_coro,
            duration=0.1,
            key="shared_key",
            when_done=on_done_second,
        )

        assert first_task_done.wait(timeout=1), "First task was not cancelled"
        assert first_task_final_status == "canceled"

        assert second_task_done.wait(timeout=1), "Second task did not complete"
        assert second_task_final_status == "completed"

        assert not manager._tasks, (
            "Manager should be empty after all tasks finish"
        )


class TestProcessTasks:
    """Tests for tasks running in a separate process."""

    def test_run_and_complete_process(self, manager: TaskManager):
        """Verify the happy path for a subprocess task."""
        completion_event = threading.Event()
        final_task = None

        def on_done(task: Task):
            nonlocal final_task
            final_task = task
            completion_event.set()

        manager.run_process(
            simple_process_func, key="proc1", when_done=on_done
        )
        assert "proc1" in manager._tasks

        assert completion_event.wait(timeout=3), (
            "Process task did not complete"
        )

        assert final_task is not None
        assert final_task.get_status() == "completed"
        assert final_task.get_progress() == 1.0
        assert final_task.get_message() == "Process step 10"
        assert final_task.result() == "process_done"
        assert not manager._tasks

    def test_process_failure(self, manager: TaskManager):
        """Verify that an exception in a subprocess is handled."""
        completion_event = threading.Event()
        final_task = None

        def on_done(task: Task):
            nonlocal final_task
            final_task = task
            completion_event.set()

        manager.run_process(
            failing_process_func, key="proc_fail", when_done=on_done
        )
        assert completion_event.wait(timeout=2)

        assert final_task is not None
        assert final_task.get_status() == "failed"
        with pytest.raises(Exception) as excinfo:
            final_task.result()

        # Check that the error message contains the subprocess exception
        # details
        assert "Subprocess Traceback" in str(excinfo.value)
        assert "Process failed intentionally" in str(excinfo.value)
        assert not manager._tasks

    def test_process_cancellation(self, manager: TaskManager):
        """
        Verify that a running subprocess can be terminated via cancellation.
        """
        completion_event = threading.Event()
        started_event = manager._mp_context.Event()
        final_task = None

        def on_done(task: Task):
            nonlocal final_task
            final_task = task
            completion_event.set()

        manager.run_process(
            long_running_process_func,
            started_event,
            key="proc_cancel",
            when_done=on_done,
        )
        task = manager._tasks["proc_cancel"]

        assert started_event.wait(timeout=2), "Process task did not start"
        task.cancel()

        assert completion_event.wait(timeout=3), (
            "Process did not clean up after cancel"
        )
        assert final_task is not None
        assert final_task.get_status() == "canceled"
        with pytest.raises(asyncio.CancelledError):
            final_task.result()
        assert not manager._tasks

    def test_process_unexpected_exit(self, manager: TaskManager):
        """Verify failure when a process exits with a non-zero exit code."""
        completion_event = threading.Event()
        final_task = None

        def on_done(task: Task):
            nonlocal final_task
            final_task = task
            completion_event.set()

        manager.run_process(
            exit_code_process_func, key="proc_exit", when_done=on_done
        )
        assert completion_event.wait(timeout=2)

        assert final_task is not None
        assert final_task.get_status() == "failed"
        with pytest.raises(Exception) as excinfo:
            final_task.result()

        assert "terminated unexpectedly with exit code 5" in str(excinfo.value)
        assert not manager._tasks


class TestTaskManagerGlobals:
    """Tests for the manager's overall state and signals."""

    @pytest.mark.asyncio
    async def test_overall_progress(
        self, manager: TaskManager, mock_timer_factory
    ):
        """Verify the calculation of overall progress."""
        assert manager.get_overall_progress() == 1.0

        steps = 4
        p_events1, p_events2 = (
            [threading.Event() for _ in range(steps)] for _ in range(2)
        )
        d_events1, d_events2 = (
            [threading.Event() for _ in range(steps)] for _ in range(2)
        )
        done_event1, done_event2 = threading.Event(), threading.Event()

        manager.add_coroutine(
            controllable_coro,
            "Task1",
            steps,
            p_events1,
            d_events1,
            key="c1",
            when_done=lambda t: done_event1.set(),
        )
        manager.add_coroutine(
            controllable_coro,
            "Task2",
            steps,
            p_events2,
            d_events2,
            key="c2",
            when_done=lambda t: done_event2.set(),
        )

        await asyncio.sleep(0.02)
        assert manager.get_overall_progress() == 0.0

        # --- Step 1: Progress task 1 by one step (25%) ---
        p_events1[0].set()
        await asyncio.to_thread(d_events1[0].wait)
        # Manually fire the debounce timer to force the progress update
        mock_timer_factory[-1].fire()
        assert manager.get_overall_progress() == pytest.approx(0.125)

        # --- Step 2: Progress task 2 by two steps (50%) ---
        p_events2[0].set()
        await asyncio.to_thread(d_events2[0].wait)
        mock_timer_factory[-1].fire()  # Fire after first step
        p_events2[1].set()
        await asyncio.to_thread(d_events2[1].wait)
        mock_timer_factory[-1].fire()  # Fire after second step
        assert manager.get_overall_progress() == pytest.approx(0.375)

        # --- Step 3: Let both tasks complete ---
        for i in range(1, steps):
            p_events1[i].set()
        for i in range(2, steps):
            p_events2[i].set()

        await asyncio.gather(
            asyncio.to_thread(done_event1.wait),
            asyncio.to_thread(done_event2.wait),
        )

        await asyncio.sleep(0.02)
        assert manager.get_overall_progress() == 1.0
        assert not manager._tasks

    def test_tasks_updated_signal(self, manager: TaskManager):
        """Verify the tasks_updated signal is emitted correctly."""
        signal_receiver = Mock()
        manager.tasks_updated.connect(signal_receiver)

        completion_event = threading.Event()
        manager.add_coroutine(
            simple_coro,
            duration=0.2,
            key="sig_test",
            when_done=lambda t: completion_event.set(),
        )

        # Give time for the "add" signal to fire.
        time.sleep(0.02)

        # Check first call (task added, status pending)
        signal_receiver.assert_called()
        args, kwargs = signal_receiver.call_args
        assert kwargs["tasks"][0].key == "sig_test"
        assert kwargs["tasks"][0].get_status() in ("pending", "running")
        assert kwargs["progress"] == 0.0

        # Wait for some progress
        time.sleep(0.1)

        # Check intermediate calls (running with progress)
        last_call_args, last_call_kwargs = signal_receiver.call_args
        assert len(last_call_kwargs["tasks"]) == 1
        assert last_call_kwargs["tasks"][0].get_status() == "running"
        assert last_call_kwargs["progress"] > 0.1
        assert (
            signal_receiver.call_count > 2
        )  # Add, Running, and progress updates

        # Wait for completion and final signal
        assert completion_event.wait(timeout=1)
        time.sleep(0.02)

        # Check final call (task removed)
        last_call_args, last_call_kwargs = signal_receiver.call_args
        assert not last_call_kwargs["tasks"]
        assert last_call_kwargs["progress"] == 1.0
