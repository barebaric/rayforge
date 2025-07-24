from __future__ import annotations
import pytest
import asyncio
from unittest.mock import Mock, AsyncMock
from rayforge.tasker.task import Task
from rayforge.tasker.context import ExecutionContext


class MockExecutionContext(ExecutionContext):
    """
    A mock ExecutionContext that allows test coroutines to update the Task.
    The real context would likely schedule this call on a main GUI thread,
    but for testing, we can call it directly.
    """

    def __init__(self, task: Task):
        self._task = task

    def update_task_progress(
        self, progress: float, message: str | None = None
    ) -> None:
        """Simulates the context updating the task's progress and message."""
        self._task.update(progress=progress, message=message)


@pytest.fixture
def signal_tracker():
    """A pytest fixture that tracks calls to a blinker Signal."""

    class Tracker:
        def __init__(self):
            self.received = []

        def __call__(self, sender, **kwargs):
            # Store the sender and its state at the time of the signal
            self.received.append(
                {
                    "sender": sender,
                    "status": sender.get_status(),
                    "progress": sender.get_progress(),
                    "message": sender.get_message(),
                }
            )

    return Tracker()


async def successful_coro(context: MockExecutionContext, *args, **kwargs):
    """A test coroutine that completes successfully and reports progress."""
    context.update_task_progress(0.5, "Halfway there")
    await asyncio.sleep(0.01)
    return "Success"


async def failing_coro(context: MockExecutionContext, *args, **kwargs):
    """A test coroutine that raises an exception."""
    await asyncio.sleep(0.01)
    raise ValueError("Something went wrong")


async def long_running_coro(
    context: MockExecutionContext,
    started_event: asyncio.Event,
    *args,
    **kwargs,
):
    """
    A test coroutine that runs long enough to be cancelled.
    It signals via an event when it has started.
    """
    try:
        context.update_task_progress(0.1, "Starting...")
        # Signal that the coroutine has started and sent its first update
        started_event.set()
        await asyncio.sleep(2)  # Long sleep to allow for cancellation
    except asyncio.CancelledError:
        context.update_task_progress(0.0, "Cancellation caught in coro")
        raise
    return "Should not be reached"


class TestTaskInitialization:
    """Tests for the Task.__init__ method."""

    def test_init_defaults(self):
        """Test task initialization with default values."""
        task = Task(successful_coro)
        assert task.coro == successful_coro
        assert task.key == id(task)
        assert task.get_status() == "pending"
        assert task.get_progress() == 0.0
        assert task.get_message() is None
        assert not task.is_cancelled()
        assert task._task is None

    def test_init_with_key_and_args(self):
        """Test task initialization with a custom key, args, and kwargs."""
        task = Task(successful_coro, 1, 2, key="my-task", kwarg1="test")
        assert task.key == "my-task"
        assert task.args == (1, 2)
        assert task.kwargs == {"kwarg1": "test"}


class TestTaskUpdate:
    """Tests for the Task.update method and signal emission."""

    def test_update_progress_and_message(self, signal_tracker):
        """Test updating both progress and message emits one signal."""
        task = Task(successful_coro)
        task.status_changed.connect(signal_tracker)

        task.update(progress=0.5, message="Updating")

        assert task.get_progress() == 0.5
        assert task.get_message() == "Updating"
        assert len(signal_tracker.received) == 1
        assert signal_tracker.received[0]["sender"] is task

    def test_update_no_change(self, signal_tracker):
        """Test that updating with the same values does not emit a signal."""
        task = Task(successful_coro)
        task.update(progress=0.25)  # Initial state
        task.status_changed.connect(signal_tracker)

        task.update(progress=0.25, message=None)  # No change

        assert len(signal_tracker.received) == 0


@pytest.mark.asyncio
class TestTaskExecution:
    """Tests for the complete lifecycle of a Task via the run() method."""

    async def test_run_successful(self, signal_tracker):
        """Test a task that runs to completion successfully."""
        task = Task(successful_coro)
        context = MockExecutionContext(task)
        task.status_changed.connect(signal_tracker)

        # Pre-run state
        assert task.get_status() == "pending"

        # Run the task
        await task.run(context)

        # Post-run state
        assert task.get_status() == "completed"
        assert task.get_progress() == 1.0
        # FIX: Removed 'await' from synchronous result() call
        assert task.result() == "Success"

        # Check signals
        assert (
            len(signal_tracker.received) == 3
        )  # running, progress update, completed
        assert signal_tracker.received[0]["status"] == "running"
        assert signal_tracker.received[1]["progress"] == 0.5
        assert signal_tracker.received[1]["message"] == "Halfway there"
        assert signal_tracker.received[2]["status"] == "completed"

    async def test_run_failure(self, signal_tracker):
        """Test a task that fails with an exception."""
        task = Task(failing_coro)
        context = MockExecutionContext(task)
        task.status_changed.connect(signal_tracker)

        with pytest.raises(ValueError, match="Something went wrong"):
            await task.run(context)

        assert task.get_status() == "failed"
        with pytest.raises(ValueError):
            # FIX: Removed 'await' from synchronous result() call
            task.result()

        # Check signals
        assert len(signal_tracker.received) == 2  # running, failed
        assert signal_tracker.received[0]["status"] == "running"
        assert signal_tracker.received[1]["status"] == "failed"

    async def test_run_and_cancel(self, signal_tracker):
        """Test cancelling a task while it is running."""
        # FIX: Use an asyncio.Event to synchronize the test
        started_event = asyncio.Event()
        task = Task(long_running_coro, started_event)  # Pass event to coro
        context = MockExecutionContext(task)
        task.status_changed.connect(signal_tracker)

        async def canceller(running_task, event):
            # Wait for the coro to signal it has started before cancelling
            await event.wait()
            running_task.cancel()

        # The task's run() method re-raises the CancelledError
        with pytest.raises(asyncio.CancelledError):
            await asyncio.gather(
                task.run(context), canceller(task, started_event)
            )

        assert task.is_cancelled()
        assert task.get_status() == "canceled"
        with pytest.raises(asyncio.CancelledError):
            # FIX: Removed 'await' from synchronous result() call
            task.result()

        # FIX: With synchronization, we can be precise about the number of signals
        # 1. running, 2. progress 0.1, 3. progress 0.0 (in coro except), 4. canceled (in finally)
        assert len(signal_tracker.received) == 4
        assert signal_tracker.received[0]["status"] == "running"
        assert (
            signal_tracker.received[1]["status"] == "running"
            and signal_tracker.received[1]["progress"] == 0.1
        )
        assert (
            signal_tracker.received[2]["status"] == "running"
            and signal_tracker.received[2]["progress"] == 0.0
        )
        assert signal_tracker.received[3]["status"] == "canceled"

    async def test_run_after_early_cancel(self, signal_tracker):
        """Test running a task that was cancelled before it started."""
        # Use an AsyncMock to verify the coroutine is never awaited
        mock_coro = AsyncMock()
        task = Task(mock_coro)
        context = MockExecutionContext(task)
        task.status_changed.connect(signal_tracker)

        task.cancel()  # Cancel before run() is called

        assert task.is_cancelled()
        assert task.get_status() == "pending"  # Status only changes in run()

        with pytest.raises(asyncio.CancelledError):
            await task.run(context)

        # Coroutine should never have been started
        mock_coro.assert_not_called()
        assert task.get_status() == "canceled"

        # Check signals: only one for the 'canceled' state
        assert len(signal_tracker.received) == 1
        assert signal_tracker.received[0]["status"] == "canceled"


class TestTaskCancellationMethod:
    """Tests specifically for the Task.cancel() method."""

    def test_cancel_before_run(self):
        """Test calling cancel() on a pending task."""
        task = Task(successful_coro)
        mock_internal_task = Mock()
        task._task = mock_internal_task

        task.cancel()

        assert task.is_cancelled()
        # Should not try to cancel a non-existent or non-running task
        mock_internal_task.cancel.assert_not_called()

    @pytest.mark.asyncio
    async def test_cancel_after_run(self):
        """Test calling cancel() on a completed task."""
        task = Task(successful_coro)
        await task.run(MockExecutionContext(task))

        assert task.get_status() == "completed"
        assert task._task.done()

        # Spy on the internal task's cancel method
        task._task.cancel = Mock()
        task.cancel()

        assert task.is_cancelled()  # The request flag is still set
        # But the already-done internal task is not cancelled again
        task._task.cancel.assert_not_called()


class TestTaskGettersAndResult:
    """Tests for getter methods and the result() method."""

    def test_getters_initial_state(self):
        """Test getters on a newly initialized task."""
        task = Task(successful_coro)
        assert task.get_status() == "pending"
        assert task.get_progress() == 0.0
        assert task.get_message() is None

    @pytest.mark.asyncio
    async def test_result_before_done(self):
        """Test that result() raises InvalidStateError if task is not done."""
        task = Task(long_running_coro, asyncio.Event())
        with pytest.raises(asyncio.InvalidStateError):
            task.result()

    @pytest.mark.asyncio
    async def test_result_after_success(self):
        """Test result() after successful completion."""
        task = Task(successful_coro)
        await task.run(MockExecutionContext(task))
        # FIX: Removed 'await' from synchronous result() call
        assert task.result() == "Success"
