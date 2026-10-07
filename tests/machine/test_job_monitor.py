import itertools
from unittest.mock import MagicMock

import pytest
from raygeo.ops import Ops

from rayforge.machine.job_monitor import JobMonitor

FEED = 600.0  # mm/min => 10 mm/s
RAPID = 3000.0
ACCELERATION = 0.0  # Makes move durations pure distance/speed.


class FakeClock:
    """A monotonic clock the tests advance by hand."""

    def __init__(self, start: float = 0.0):
        self.now = start

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float):
        self.now += seconds


def straight_ops(lines: int) -> Ops:
    """Lines of 10 mm at FEED: 1 s of modeled time each."""
    ops = Ops()
    ops.move_to(0, 0, 0)
    ops.set_feed_rate(FEED)
    for i in range(lines):
        ops.line_to(10 * (i + 1), 0, 0)
    return ops


def mixed_ops(pairs: int) -> Ops:
    """Alternating fast and slow phases: per pair one 10 mm line at
    FEED (1 s) and one 1 mm line at FEED / 100 (10 s), so the local
    speed alternates by a factor of 100."""
    ops = Ops()
    ops.move_to(0, 0, 0)
    x = 0.0
    for _ in range(pairs):
        ops.set_feed_rate(FEED)
        x += 10
        ops.line_to(x, 0, 0)
        ops.set_feed_rate(FEED / 100)
        x += 1
        ops.line_to(x, 0, 0)
    return ops


def make_monitor(ops, clock, **kwargs):
    estimated = ops.estimate_time(FEED, RAPID, ACCELERATION)
    return JobMonitor(
        ops,
        estimated_seconds=estimated,
        default_feed_rate=FEED,
        default_rapid_rate=RAPID,
        acceleration=ACCELERATION,
        clock=clock,
        **kwargs,
    )


class TestJobMonitor:
    """Test suite for the JobMonitor class."""

    @pytest.fixture
    def simple_ops(self):
        """Creates a simple Ops object with a few commands."""
        ops = Ops()
        ops.move_to(10, 10, 0)  # op 0, distance 0
        ops.line_to(20, 10, 0)  # op 1, distance 10
        ops.line_to(20, 20, 0)  # op 2, distance 10
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

    def test_mark_as_complete(self, simple_ops):
        """Test that mark_as_complete sets progress to 100%."""
        monitor = JobMonitor(simple_ops, estimated_seconds=100.0)
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
        assert metrics["eta_seconds"] == 0.0

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

    def test_eta_none_without_estimate(self, monitor):
        """Without an estimate, eta stays None."""
        assert monitor.metrics["eta_seconds"] is None
        monitor.update_progress(1)
        assert monitor.metrics["eta_seconds"] is None

    def test_eta_counts_down_estimate_before_progress(self):
        """Without acknowledged progress the ETA counts down the
        estimate. This also covers drivers that never report granular
        progress."""
        ops = straight_ops(2)
        clock = FakeClock()
        monitor = make_monitor(ops, clock)

        assert monitor.metrics["eta_seconds"] == pytest.approx(2.0)

        clock.advance(0.5)
        assert monitor.metrics["eta_seconds"] == pytest.approx(1.5)

        clock.advance(10.0)
        assert monitor.metrics["eta_seconds"] == 0.0

    def test_eta_follows_model_progress(self):
        """Once commands are acknowledged, the ETA counts down the
        modeled time remaining at the acknowledged frontier."""
        ops = straight_ops(2)  # 2 lines, 1 s each
        clock = FakeClock()
        monitor = make_monitor(ops, clock)

        # Op indices: 0 = move_to, 1 = set_feed_rate, 2..3 = lines.
        monitor.update_progress(2)
        assert monitor._frontier_seconds == pytest.approx(1.0)
        assert monitor.metrics["eta_seconds"] == pytest.approx(1.0)

        monitor.update_progress(3)
        assert monitor.metrics["eta_seconds"] == 0.0

    def test_eta_pace_correction_converges(self):
        """A machine running uniformly slower than the model must
        inflate the ETA by its true pace once the ramp completes."""
        ops = straight_ops(10)  # 10 s modeled
        clock = FakeClock()
        monitor = make_monitor(ops, clock)
        monitor.PACE_WARMUP_SECONDS = 2.0
        monitor.PACE_RAMP_SECONDS = 8.0

        # Acknowledge the first five line ops (2..6), taking 2 s of
        # wall time per modeled second (the machine runs at half the
        # modeled speed).
        for i in range(2, 7):
            monitor.update_progress(i)
            clock.advance(2.0)

        # 5 lines done: 5 s modeled, 10 s elapsed. Pace 2.0 at full
        # weight: (10 - 5) * 2.0.
        assert monitor.metrics["eta_seconds"] == pytest.approx(10.0)

    def test_pace_is_clamped(self):
        """Pathological elapsed/frontier ratios must not explode the
        ETA beyond the pace clamp."""
        ops = straight_ops(10)
        clock = FakeClock()
        monitor = make_monitor(ops, clock)
        monitor.PACE_WARMUP_SECONDS = 0.0
        monitor.PACE_RAMP_SECONDS = 1.0

        monitor.update_progress(2)  # 1 s modeled
        clock.advance(10_000.0)  # pace 10000: clamped to 10

        assert monitor.metrics["eta_seconds"] == pytest.approx(9.0 * 10.0)

    def test_eta_survives_ack_bursts(self):
        """Regression test: GRBL acks arrive when the firmware buffers
        commands, so a burst can acknowledge many operations within
        milliseconds. The ETA must jump to the modeled remaining time
        at that frontier, not collapse towards zero."""
        ops = mixed_ops(30)  # 330 s modeled
        clock = FakeClock()
        monitor = make_monitor(ops, clock)

        clock.advance(0.1)
        for i in range(1, 13):  # Three pairs: 33 s of modeled time
            monitor.update_progress(i)

        assert monitor._frontier_seconds == pytest.approx(33.0)
        assert monitor.metrics["eta_seconds"] == pytest.approx(297.0)

    def test_eta_stable_on_mixed_job(self):
        """Regression test: a job alternating fast and slow phases by a
        factor of 100 in local speed must not swing the ETA. With the
        machine matching the model, the ETA equals the modeled
        remaining time at every step and decreases monotonically."""
        ops = mixed_ops(30)  # 330 s modeled
        clock = FakeClock()
        monitor = make_monitor(ops, clock)

        cumulative = ops.build_cumulative_time_index(FEED, RAPID, ACCELERATION)
        total = cumulative[-1]

        etas = []
        previous = 0.0
        for i in range(1, ops.len()):
            clock.advance(cumulative[i] - previous)
            monitor.update_progress(i)
            previous = cumulative[i]

            eta = monitor.metrics["eta_seconds"]
            remaining = total - cumulative[i]
            assert eta == pytest.approx(remaining, abs=1e-6)
            etas.append(eta)

        assert all(a >= b for a, b in itertools.pairwise(etas))

    def test_eta_stable_on_uniformly_slow_mixed_job(self):
        """The same mixed job on a machine 1.5x slower than the model:
        the pace correction converges to 1.5x the modeled remaining
        time (the true remaining time) instead of oscillating with the
        local speed of the last few commands."""
        ops = mixed_ops(30)  # 330 s modeled
        clock = FakeClock()
        monitor = make_monitor(ops, clock)
        monitor.PACE_WARMUP_SECONDS = 5.0
        monitor.PACE_RAMP_SECONDS = 100.0

        cumulative = ops.build_cumulative_time_index(FEED, RAPID, ACCELERATION)
        total = cumulative[-1]

        previous = 0.0
        for i in range(1, ops.len()):
            clock.advance(1.5 * (cumulative[i] - previous))
            monitor.update_progress(i)
            previous = cumulative[i]

            eta = monitor.metrics["eta_seconds"]
            remaining = total - cumulative[i]
            if remaining <= 0.0:
                assert eta == 0.0
                continue
            ratio = eta / remaining
            assert 1.0 - 1e-6 <= ratio <= 1.5 + 1e-6
