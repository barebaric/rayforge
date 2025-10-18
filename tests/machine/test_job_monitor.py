import pytest
from unittest.mock import MagicMock
from rayforge.core.ops import Ops, MoveToCommand, LineToCommand
from rayforge.machine.job_monitor import JobMonitor


class TestJobMonitor:
    """Test suite for the JobMonitor class."""

    @pytest.fixture
    def simple_ops(self):
        """Creates a simple Ops object with a few commands."""
        ops = Ops()
        ops.add(MoveToCommand((10, 10, 0)))  # op 0, distance 0
        ops.add(LineToCommand((20, 10, 0)))  # op 1, distance 10
        ops.add(LineToCommand((20, 20, 0)))  # op 2, distance 10
        return ops

    @pytest.fixture
    def monitor(self, simple_ops):
        """Provides a JobMonitor instance initialized with simple_ops."""
        return JobMonitor(simple_ops)

    def test_initialization(self, monitor, simple_ops):
        """Test correct initialization of distances."""
        assert monitor.total_distance == 20.0
        assert monitor.traveled_distance == 0.0
        assert monitor.ops is simple_ops
        assert monitor._distance_map == {0: 0.0, 1: 10.0, 2: 10.0}

    def test_update_progress(self, monitor):
        """Test that update_progress correctly increments traveled_distance."""
        signal_spy = MagicMock()
        monitor.progress_updated.connect(signal_spy)

        # Update for op 0 (MoveTo, distance 0)
        monitor.update_progress(0)
        assert monitor.traveled_distance == 0.0
        signal_spy.assert_called_once()
        metrics = signal_spy.call_args.kwargs["metrics"]
        assert metrics["traveled_distance"] == 0.0
        assert metrics["progress_fraction"] == 0.0

        # Update for op 1 (LineTo, distance 10)
        monitor.update_progress(1)
        assert monitor.traveled_distance == 10.0
        assert signal_spy.call_count == 2
        metrics = signal_spy.call_args.kwargs["metrics"]
        assert metrics["traveled_distance"] == 10.0
        assert metrics["progress_fraction"] == 0.5

        # Update for op 2 (LineTo, distance 10)
        monitor.update_progress(2)
        assert monitor.traveled_distance == 20.0
        assert signal_spy.call_count == 3
        metrics = signal_spy.call_args.kwargs["metrics"]
        assert metrics["traveled_distance"] == 20.0
        assert metrics["progress_fraction"] == 1.0

    def test_update_progress_clamps_at_total_on_float_error(self):
        """Test that progress is clamped if float errors cause an overflow."""
        monitor = JobMonitor(Ops())  # Start with empty ops
        monitor.total_distance = 10.0
        monitor._distance_map = {0: 5.0, 1: 5.1}  # Sum > total_distance

        monitor.update_progress(0)
        assert monitor.traveled_distance == 5.0

        monitor.update_progress(1)
        # Should be clamped to the total, not 10.1
        assert monitor.traveled_distance == 10.0

    def test_mark_as_complete(self, monitor):
        """Test that mark_as_complete sets progress to 100%."""
        signal_spy = MagicMock()
        monitor.progress_updated.connect(signal_spy)

        # Set some partial progress first
        monitor.update_progress(1)
        assert monitor.traveled_distance == 10.0
        signal_spy.reset_mock()

        monitor.mark_as_complete()

        # Verify final state
        assert monitor.traveled_distance == 20.0
        signal_spy.assert_called_once()
        metrics = signal_spy.call_args.kwargs["metrics"]
        assert metrics["traveled_distance"] == 20.0
        assert metrics["progress_fraction"] == 1.0

    def test_with_empty_ops(self):
        """Test that the monitor handles empty Ops gracefully."""
        ops = Ops()
        monitor = JobMonitor(ops)
        assert monitor.total_distance == 0.0
        assert monitor.traveled_distance == 0.0

        signal_spy = MagicMock()
        monitor.progress_updated.connect(signal_spy)

        # Should not raise an error
        monitor.update_progress(0)
        assert monitor.traveled_distance == 0.0
        signal_spy.assert_called_once()  # Still signals

        monitor.mark_as_complete()
        assert monitor.traveled_distance == 0.0
        assert signal_spy.call_count == 2
        metrics = signal_spy.call_args.kwargs["metrics"]
        # 0/0 should result in 100% complete
        assert metrics["progress_fraction"] == 1.0

    def test_metrics_property(self, monitor):
        """Test that the metrics property is always up-to-date."""
        metrics = monitor.metrics
        assert metrics["total_distance"] == 20.0
        assert metrics["traveled_distance"] == 0.0
        assert metrics["progress_fraction"] == 0.0

        monitor.update_progress(1)
        metrics = monitor.metrics
        assert metrics["total_distance"] == 20.0
        assert metrics["traveled_distance"] == 10.0
        assert metrics["progress_fraction"] == 0.5
