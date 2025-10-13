import pytest
from queue import Queue, Full
from unittest.mock import Mock, call
from rayforge.shared.tasker.proxy import ExecutionContextProxy


@pytest.fixture
def mock_queue():
    """Provides a standard queue for proxy tests."""
    return Queue()


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


class TestExecutionContextProxy:
    def test_initialization(self, mock_queue):
        proxy = ExecutionContextProxy(
            mock_queue, base_progress=0.1, progress_range=0.8
        )
        assert proxy._queue is mock_queue
        assert proxy._base == 0.1
        assert proxy._range == 0.8
        assert proxy._total == 1.0
        assert proxy.task is None

    def test_set_total(self, mock_queue):
        proxy = ExecutionContextProxy(mock_queue)
        proxy.set_total(200)
        assert proxy._total == 200.0
        proxy.set_total(0)
        assert proxy._total == 1.0
        proxy.set_total(-10)
        assert proxy._total == 1.0

    def test_set_progress_simple(self, mock_queue):
        proxy = ExecutionContextProxy(mock_queue)
        proxy.set_total(100)
        proxy.set_progress(25)
        assert mock_queue.get_nowait() == ("progress", 0.25)

    def test_set_progress_clamping(self, mock_queue):
        proxy = ExecutionContextProxy(mock_queue)
        proxy.set_total(50)
        proxy.set_progress(-10)
        assert mock_queue.get_nowait() == ("progress", 0.0)
        proxy.set_progress(100)
        assert mock_queue.get_nowait() == ("progress", 1.0)

    def test_set_progress_with_scaling(self, mock_queue):
        # This proxy represents the 20% to 70% range of a larger task.
        proxy = ExecutionContextProxy(
            mock_queue, base_progress=0.2, progress_range=0.5
        )
        proxy.set_total(10)

        # Report 50% progress within its own context (5/10)
        proxy.set_progress(5)

        # Expected final progress:
        #   base + (normalized * range) = 0.2 + (0.5 * 0.5) = 0.45
        assert mock_queue.get_nowait() == ("progress", 0.45)

    def test_set_message(self, mock_queue):
        proxy = ExecutionContextProxy(mock_queue)
        proxy.set_message("Working...")
        assert mock_queue.get_nowait() == ("message", "Working...")

    def test_queue_full(self, mocker):
        mock_q = Mock(spec=Queue)
        mock_q.put_nowait.side_effect = Full
        proxy = ExecutionContextProxy(mock_q)

        # These calls should not raise an exception
        proxy.set_progress(0.5)
        proxy.set_message("test")

        assert mock_q.put_nowait.call_count == 2
        mock_q.put_nowait.assert_has_calls(
            [call(("progress", 0.5)), call(("message", "test"))]
        )

    def test_sub_context_creation(self, mock_queue):
        parent = ExecutionContextProxy(
            mock_queue, base_progress=0.1, progress_range=0.8
        )
        child = parent.sub_context(
            base_progress=0.5, progress_range=0.25, total=50
        )

        assert isinstance(child, ExecutionContextProxy)
        assert child._queue is mock_queue
        # Expected base:
        #  parent_base + (sub_base * parent_range) = 0.1 + (0.5 * 0.8) = 0.5
        assert child._base == pytest.approx(0.5)
        # Expected range: parent_range * sub_range = 0.8 * 0.25 = 0.2
        assert child._range == pytest.approx(0.2)
        assert child._total == 50.0

    def test_sub_context_progress_reporting(self, mock_queue):
        # Parent represents 10%-90% (range 0.8) of the total task.
        parent = ExecutionContextProxy(
            mock_queue, base_progress=0.1, progress_range=0.8
        )
        # Child represents the second half (50%-100%, range 0.5)
        # of the parent's task.
        child = parent.sub_context(
            base_progress=0.5, progress_range=0.5, total=200
        )

        # Report 50% progress in child's context (100/200)
        child.set_progress(100)

        # Child normalized progress = 0.5
        # Child's contribution to overall progress:
        # child_base = parent_base + (sub_base * parent_range)
        #            = 0.1 + (0.5 * 0.8) = 0.5
        # child_range = parent_range * sub_range = 0.8 * 0.5 = 0.4
        # final_progress = child_base + (child_normalized * child_range)
        #                = 0.5 + (0.5 * 0.4) = 0.5 + 0.2 = 0.7

        assert mock_queue.get_nowait() == ("progress", pytest.approx(0.7))

    def test_sub_context_message_reporting(self, mock_queue):
        parent = ExecutionContextProxy(mock_queue)
        child = parent.sub_context(0, 1)
        child.set_message("Sub-task update")
        assert mock_queue.get_nowait() == ("message", "Sub-task update")

    def test_is_cancelled_and_flush(self, mock_queue):
        proxy = ExecutionContextProxy(mock_queue)
        assert not proxy.is_cancelled()
        proxy.flush()  # Should be a no-op
        assert mock_queue.empty()
