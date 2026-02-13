import pytest
import time
import logging
from queue import Queue, Full
from unittest.mock import Mock, call
from multiprocessing import Manager
from rayforge.shared.tasker.proxy import ExecutionContextProxy


@pytest.fixture
def mock_queue():
    """Provides a standard queue for proxy tests."""
    return Queue()


@pytest.fixture
def manager():
    """Provides a multiprocessing Manager for shared state tests."""
    mgr = Manager()
    yield mgr
    mgr.shutdown()


@pytest.fixture
def logger():
    """Provides a logger for tests that need it."""
    return logging.getLogger("test_proxy")


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
        # The next update might be throttled because it's called immediately.
        # We call flush() to ensure the final value is sent for this test.
        proxy.set_progress(100)
        proxy.flush()
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

    def test_is_cancelled_always_false(self, mock_queue):
        """The proxy cannot know about cancellation; it is a one-way street."""
        proxy = ExecutionContextProxy(mock_queue)
        assert not proxy.is_cancelled()

    def test_flush_behavior(self, mock_queue):
        """
        Verify flush() sends throttled progress and is a no-op otherwise.
        """
        proxy = ExecutionContextProxy(mock_queue)
        # Set a long interval to guarantee throttling for the test
        proxy.PROGRESS_REPORT_INTERVAL_S = 10.0

        # 1. An initial flush should do nothing.
        proxy.flush()
        assert mock_queue.empty()

        # 2. First update goes through.
        proxy.set_progress(0.2)
        assert mock_queue.get_nowait() == ("progress", 0.2)

        # 3. Second update should be stored but throttled (not sent).
        proxy.set_progress(0.8)
        assert mock_queue.empty()

        # 4. Flush sends the stored value and resets the pending state.
        proxy.flush()
        assert mock_queue.get_nowait() == ("progress", 0.8)

        # 5. A second flush now does nothing.
        proxy.flush()
        assert mock_queue.empty()

    def test_send_event_and_wait_no_adoption_signals(self, mock_queue):
        """
        When no adoption_signals dict is provided, send_event_and_wait
        should return True immediately (fire-and-forget mode).
        """
        proxy = ExecutionContextProxy(mock_queue)
        result = proxy.send_event_and_wait("test_event", {"data": 1})
        assert result is True
        # Event should still be sent
        assert mock_queue.get_nowait() == (
            "event",
            ("test_event", {"data": 1}),
        )

    def test_send_event_and_wait_with_adoption_signal(
        self, mock_queue, manager, logger
    ):
        """
        When adoption_signals is provided, send_event_and_wait should
        wait for the signal to be set before returning True.
        """
        adoption_signals = manager.dict()
        task_id = 12345
        proxy = ExecutionContextProxy(
            mock_queue,
            adoption_signals=adoption_signals,
            task_id=task_id,
        )

        # Start the wait in a separate thread
        import threading

        result_holder = {"result": None}

        def wait_for_event():
            result_holder["result"] = proxy.send_event_and_wait(  # type: ignore
                "my_event", {}, timeout=2
            )

        wait_thread = threading.Thread(target=wait_for_event)
        wait_thread.start()

        # Give the thread time to start waiting
        time.sleep(0.1)

        # Signal should not be set yet
        signal_key = f"{task_id}:my_event"
        assert signal_key not in adoption_signals

        # Set the signal
        adoption_signals[signal_key] = True

        # Wait for the thread to complete
        wait_thread.join(timeout=3)
        assert not wait_thread.is_alive()
        assert result_holder["result"] is True

    def test_send_event_and_wait_timeout(self, mock_queue, manager, logger):
        """
        When adoption_signals is provided but the signal is never set,
        send_event_and_wait should return False after timeout.
        """
        adoption_signals = manager.dict()
        proxy = ExecutionContextProxy(
            mock_queue,
            adoption_signals=adoption_signals,
            task_id=99999,
        )

        # Use a short timeout for the test
        start_time = time.monotonic()
        result = proxy.send_event_and_wait(
            "timeout_event",
            {"test": "data"},
            timeout=0.3,
            logger=logger,
        )
        elapsed = time.monotonic() - start_time

        assert result is False
        # Should have waited at least the timeout duration
        assert elapsed >= 0.25  # Allow some margin
        # Event should still have been sent
        msg = mock_queue.get_nowait()
        assert msg[0] == "event"
        assert msg[1][0] == "timeout_event"

    def test_send_event_and_wait_signal_cleanup(
        self, mock_queue, manager, logger
    ):
        """
        When the signal is received, the signal key should be cleaned up
        from the adoption_signals dict.
        """
        adoption_signals = manager.dict()
        task_id = 55555
        proxy = ExecutionContextProxy(
            mock_queue,
            adoption_signals=adoption_signals,
            task_id=task_id,
        )

        signal_key = f"{task_id}:cleanup_event"
        adoption_signals[signal_key] = True

        result = proxy.send_event_and_wait("cleanup_event", {}, timeout=1)

        assert result is True
        # The signal key should have been cleaned up
        assert signal_key not in adoption_signals

    def test_sub_context_inherits_adoption_signals(self, mock_queue, manager):
        """
        Sub-contexts should inherit the adoption_signals and task_id
        from their parent.
        """
        adoption_signals = manager.dict()
        parent = ExecutionContextProxy(
            mock_queue,
            adoption_signals=adoption_signals,
            task_id=11111,
        )

        child = parent.sub_context(0.0, 0.5)

        assert child._adoption_signals is adoption_signals
        assert child._task_id == 11111
