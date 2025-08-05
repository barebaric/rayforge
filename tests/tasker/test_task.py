from __future__ import annotations
import pytest
import asyncio
from unittest.mock import Mock, AsyncMock
from rayforge.shared.tasker.task import Task
from rayforge.shared.tasker.context import ExecutionContext


@pytest.fixture
def mock_idle_add(mocker):
    """Mocks glib.idle_add to execute callbacks immediately."""

    def idle_add_immediate(callback, *args, **kwargs):
        callback(*args, **kwargs)
        return 0

    return mocker.patch(
        "rayforge.shared.tasker.context.idle_add", idle_add_immediate
    )


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
    Mocks threading.Timer with a controllable version and returns a list
    of all created timer instances.
    """
    timers = []

    def factory(*args, **kwargs):
        timer = ControllableTimer(*args, **kwargs)
        timers.append(timer)
        return timer

    mocker.patch("rayforge.shared.tasker.context.threading.Timer", factory)
    return timers


class MockExecutionContext(ExecutionContext):
    """
    A mock ExecutionContext that allows test coroutines to update the Task.
    The real context would likely schedule this call on a main GUI thread,
    but for testing, we can call it directly.
    """

    def __init__(self, task: Task):
        super().__init__(
            update_callback=task.update, check_cancelled=task.is_cancelled
        )
        self._task = task


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


async def successful_coro(context: ExecutionContext, *args, **kwargs):
    """A test coroutine that completes successfully and reports progress."""
    context.set_progress(0.5)
    context.set_message("Halfway there")
    await asyncio.sleep(0.01)
    return "Success"


async def failing_coro(context: ExecutionContext, *args, **kwargs):
    """A test coroutine that raises an exception."""
    await asyncio.sleep(0.01)
    raise ValueError("Something went wrong")


async def long_running_coro(
    context: ExecutionContext,
    started_event: asyncio.Event,
    *args,
    **kwargs,
):
    """
    A test coroutine that runs long enough to be cancelled.
    It signals via an event when it has started.
    """
    try:
        context.set_message("Starting...")
        # Signal that the coroutine has started and sent its first update
        started_event.set()
        await asyncio.sleep(2)  # Long sleep to allow for cancellation
    except asyncio.CancelledError:
        context.set_message("Cancellation caught in coro")
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

        task.update(progress=0.25, message=None)

        assert len(signal_tracker.received) == 0


@pytest.mark.asyncio
class TestTaskExecution:
    """Tests for the complete lifecycle of a Task via the run() method."""

    async def test_run_successful(
        self, signal_tracker, mock_idle_add, mock_timer_factory
    ):
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
        assert task.result() == "Success"

        # We expect 3 signals:
        # 1. 'running'
        # 2. The flushed intermediate update (progress=0.5, message='...')
        # 3. The final 'completed' state (progress=1.0)
        assert len(signal_tracker.received) == 3
        assert signal_tracker.received[0]["status"] == "running"
        assert signal_tracker.received[1]["progress"] == 0.5
        assert signal_tracker.received[1]["message"] == "Halfway there"
        assert signal_tracker.received[2]["status"] == "completed"
        assert signal_tracker.received[2]["progress"] == 1.0

    async def test_run_failure(
        self, signal_tracker, mock_idle_add, mock_timer_factory
    ):
        """Test a task that fails with an exception."""
        task = Task(failing_coro)
        context = MockExecutionContext(task)
        task.status_changed.connect(signal_tracker)

        with pytest.raises(ValueError, match="Something went wrong"):
            await task.run(context)

        assert task.get_status() == "failed"
        with pytest.raises(ValueError):
            task.result()

        # We expect 2 signals: 'running' and 'failed'.
        assert len(signal_tracker.received) == 2
        assert signal_tracker.received[0]["status"] == "running"
        assert signal_tracker.received[1]["status"] == "failed"

    async def test_run_and_cancel(
        self, signal_tracker, mock_idle_add, mock_timer_factory
    ):
        """Test cancelling a task while it is running."""
        started_event = asyncio.Event()
        task = Task(long_running_coro, started_event)
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
            task.result()

        # We expect 3 signals, because flush() works correctly.
        # 1. 'running': Sent at the start of run().
        # 2. Flushed update: Sent from flush(), triggered by the finally block.
        #    At this point, status is already 'canceled' from the except block.
        # 3. Final update: Sent by the unconditional _emit_status_changed() at
        #    the end of the finally block.
        assert len(signal_tracker.received) == 3

        # Signal 1: The task starts running.
        assert signal_tracker.received[0]["status"] == "running"

        # Signal 2: The flush() call sends the pending message. The status
        # has already been set to 'canceled' in the except block.
        assert signal_tracker.received[1]["status"] == "canceled"
        assert (
            signal_tracker.received[1]["message"]
            == "Cancellation caught in coro"
        )

        # Signal 3: The final signal confirms the 'canceled' state. The message
        # from the previous signal persists.
        assert signal_tracker.received[2]["status"] == "canceled"
        assert (
            signal_tracker.received[2]["message"]
            == "Cancellation caught in coro"
        )

    async def test_run_after_early_cancel(
        self, signal_tracker, mock_idle_add, mock_timer_factory
    ):
        """Test running a task that was cancelled before it started."""
        mock_coro = AsyncMock()
        task = Task(mock_coro)
        context = MockExecutionContext(task)
        task.status_changed.connect(signal_tracker)

        task.cancel()

        assert task.is_cancelled()
        assert task.get_status() == "pending"

        with pytest.raises(asyncio.CancelledError):
            await task.run(context)

        mock_coro.assert_not_called()
        assert task.get_status() == "canceled"

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
    async def test_cancel_after_run(self, mock_idle_add, mock_timer_factory):
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
    async def test_result_after_success(
        self, mock_idle_add, mock_timer_factory
    ):
        """Test result() after successful completion."""
        task = Task(successful_coro)
        await task.run(MockExecutionContext(task))
        assert task.result() == "Success"
