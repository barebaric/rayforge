import pytest
from unittest.mock import Mock
from rayforge.tasker import ExecutionContext


@pytest.fixture
def mock_idle_add(mocker):
    """Mocks the glib.idle_add function."""
    return mocker.patch("rayforge.tasker.context.idle_add")


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

    mocker.patch("threading.Timer", factory)
    return timers


class TestExecutionContext:
    def test_root_context_initialization(self, mock_idle_add):
        update_cb = Mock()
        cancel_cb = Mock(return_value=False)
        ctx = ExecutionContext(
            update_callback=update_cb,
            check_cancelled=cancel_cb,
            debounce_interval_ms=50,
        )
        assert ctx._parent_context is None
        assert ctx._update_callback is update_cb
        assert ctx._check_cancelled is cancel_cb
        assert ctx._debounce_interval_sec == 0.05
        assert ctx._lock is not None
        assert ctx._base == 0.0
        assert ctx._range == 1.0
        assert ctx._total == 1.0
        assert ctx.task is None

    def test_update_scheduling_and_throttling(
        self, mock_idle_add, mock_timer_factory
    ):
        ctx = ExecutionContext(
            update_callback=Mock(), debounce_interval_ms=100
        )

        # First update starts a timer
        ctx.set_progress(0.1)
        assert len(mock_timer_factory) == 1
        timer1 = mock_timer_factory[0]
        assert timer1.is_started
        assert not timer1.is_cancelled
        assert ctx._pending_progress == 0.1

        # Second update before timer fires should NOT create a new timer
        ctx.set_message("Throttling")
        assert len(mock_timer_factory) == 1
        # The original timer should still be active
        assert not timer1.is_cancelled
        assert ctx._pending_progress == 0.1  # Progress is preserved
        assert ctx._pending_message == "Throttling"  # Message is updated

    def test_fire_update(self, mock_idle_add, mock_timer_factory):
        update_cb = Mock()
        ctx = ExecutionContext(update_callback=update_cb)
        ctx.set_progress(0.5)
        ctx.set_message("Update fired")

        # Both updates should only result in one timer being created
        assert len(mock_timer_factory) == 1
        timer = mock_timer_factory[-1]

        # Manually fire the timer
        timer.fire()

        # The callback should receive the latest values for both progress and
        # message
        mock_idle_add.assert_called_once_with(update_cb, 0.5, "Update fired")
        assert ctx._pending_progress is None
        assert ctx._pending_message is None
        assert ctx._update_timer is None

    def test_fire_update_when_cancelled(
        self, mock_idle_add, mock_timer_factory
    ):
        update_cb = Mock()
        ctx = ExecutionContext(
            update_callback=update_cb, check_cancelled=lambda: True
        )
        ctx.set_progress(0.5)
        timer = mock_timer_factory[-1]

        timer.fire()

        mock_idle_add.assert_not_called()

    def test_flush(self, mock_idle_add, mock_timer_factory):
        update_cb = Mock()
        ctx = ExecutionContext(update_callback=update_cb)
        ctx.set_progress(0.75)
        ctx.set_message("Flushing")

        # Both updates should only result in one timer
        assert len(mock_timer_factory) == 1
        timer = mock_timer_factory[-1]
        assert timer.is_started

        ctx.flush()

        # Flush should cancel the pending timer and fire the update immediately
        assert timer.is_cancelled
        mock_idle_add.assert_called_once_with(update_cb, 0.75, "Flushing")
        assert ctx._pending_progress is None
        assert ctx._pending_message is None

    def test_flush_when_nothing_pending(self, mock_idle_add):
        update_cb = Mock()
        ctx = ExecutionContext(update_callback=update_cb)
        ctx.flush()
        mock_idle_add.assert_not_called()

    def test_sub_context_initialization(self):
        root = ExecutionContext()
        sub = root.sub_context(0.2, 0.5, total=10)

        assert isinstance(sub, ExecutionContext)
        assert sub._parent_context is root
        assert sub._get_root() is root
        assert sub._update_callback is None  # Delegates to parent
        assert sub._lock is None  # Not needed
        assert sub._base == 0.2
        assert sub._range == 0.5
        assert sub._total == 10.0
        # Inherits cancellation check
        assert sub.is_cancelled() is root.is_cancelled()

    def test_sub_context_progress_reporting(self, mock_timer_factory):
        root = ExecutionContext()
        # Sub-context represents the 20%-70% block (base=0.2, range=0.5)
        sub = root.sub_context(0.2, 0.5, total=100)

        # Report 50% progress (50/100) in the sub-context
        sub.set_progress(50)

        # Expected scaled progress in root: 0.2 + (0.5 * 0.5) = 0.45
        assert root._pending_progress == pytest.approx(0.45)
        assert root._pending_message is None
        # The root context should have scheduled an update
        assert len(mock_timer_factory) == 1
        assert mock_timer_factory[0].is_started

    def test_deeply_nested_sub_context_progress(self, mock_timer_factory):
        root = ExecutionContext()
        # sub1: 10% - 90% (base=0.1, range=0.8)
        sub1 = root.sub_context(0.1, 0.8)
        # sub2: 50% - 75% of sub1's range (base=0.5, range=0.25)
        sub2 = sub1.sub_context(0.5, 0.25, total=10)

        # Report 80% progress (8/10) in sub2
        sub2.set_progress(8)

        # Calculation:
        # sub2's contribution to the global progress
        # sub2's base is 0.1 (from sub1) + 0.5 * 0.8 (from sub1) = 0.5
        # sub2's range is 0.8 (from sub1) * 0.25 = 0.2
        # sub2's normalized progress is 0.8
        # Final global progress = sub2_base + (sub2_norm * sub2_range)
        # = 0.5 + (0.8 * 0.2) = 0.5 + 0.16 = 0.66
        assert root._pending_progress == pytest.approx(0.66)
        assert len(mock_timer_factory) == 1
        assert mock_timer_factory[0].is_started

    def test_sub_context_message_reporting(self, mock_timer_factory):
        root = ExecutionContext()
        sub = root.sub_context(0, 1)
        sub.set_message("From sub-context")

        assert root._pending_message == "From sub-context"
        assert root._pending_progress is None
        assert len(mock_timer_factory) == 1
        assert mock_timer_factory[0].is_started

    def test_sub_context_cancellation_check(self):
        root_cancelled = False

        def root_is_cancelled():
            return root_cancelled

        sub_cancelled = False

        def sub_is_cancelled():
            return sub_cancelled

        root = ExecutionContext(check_cancelled=root_is_cancelled)
        sub = root.sub_context(0, 1)
        sub_override = root.sub_context(0, 1, check_cancelled=sub_is_cancelled)

        # Test inheritance
        assert not sub.is_cancelled()
        root_cancelled = True
        assert sub.is_cancelled()

        # Test override
        assert not sub_override.is_cancelled()
        sub_cancelled = True
        assert sub_override.is_cancelled()
        # Root's state doesn't affect the override
        root_cancelled = False
        assert sub_override.is_cancelled()

    def test_sub_context_flush(self, mock_idle_add, mock_timer_factory):
        update_cb = Mock()
        root = ExecutionContext(update_callback=update_cb)
        sub = root.sub_context(0, 1)

        sub.set_progress(0.5)
        assert len(mock_timer_factory) == 1
        timer = mock_timer_factory[-1]

        sub.flush()  # Should delegate to root

        assert timer.is_cancelled
        # sub has base=0, range=1. progress=0.5 is normalized.
        # global progress = 0 + (0.5*1) = 0.5
        mock_idle_add.assert_called_once_with(update_cb, 0.5, None)
        assert root._pending_progress is None
