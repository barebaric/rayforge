from __future__ import annotations

import logging
import time
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from blinker import Signal
from raygeo.geo.types import Point3D
from raygeo.ops.types import CommandCategory

if TYPE_CHECKING:
    from raygeo.ops import Ops


logger = logging.getLogger(__name__)


class JobMonitor:
    """
    Tracks and reports the progress of a machine job based on Ops data.

    This class calculates the total distance of a job from an Ops object and
    updates the progress as individual operations complete. It emits a signal
    with detailed metrics whenever the progress changes.

    The ETA counts down the modeled time remaining at the acknowledged
    command frontier. A pace correction is blended in after a warmup
    period: wall-clock time is compared against the time the model
    predicted for the acknowledged commands, and the remaining
    estimate is scaled by that ratio. This lets the ETA converge to
    the true remaining time when the model is uniformly off (e.g. the
    configured acceleration does not match the firmware) without ever
    reacting to the local speed of the last few commands: a windowed
    speed measurement multiplied by the global remaining distance
    used to swing the displayed ETA between a fraction and a multiple
    of the real remaining time whenever the job mixed fast and slow
    phases.
    """

    # No pace correction before this much wall-clock time has passed.
    # GRBL-style drivers acknowledge commands when the firmware buffers
    # them, so the frontier runs ahead of the tool by the planner depth
    # (several moves' worth of time); the pace ratio only becomes
    # meaningful once the elapsed time dwarfs that lead.
    PACE_WARMUP_SECONDS: float = 120.0

    # Duration over which the pace correction is blended in, so the
    # ETA glides towards the corrected value instead of jumping.
    PACE_RAMP_SECONDS: float = 480.0

    # Bounds for the pace ratio, guarding the display against
    # pathological inputs such as a driver resending its command
    # stream or a machine sitting in a feed hold for a long time.
    MIN_PACE: float = 0.1
    MAX_PACE: float = 10.0

    def __init__(
        self,
        ops: Ops,
        estimated_seconds: float | None = None,
        default_feed_rate: float = 1000.0,
        default_rapid_rate: float = 3000.0,
        acceleration: float = 1000.0,
        clock: Callable[[], float] | None = None,
    ):
        """
        Initializes the JobMonitor.

        Args:
            ops: The Ops object representing the job to be monitored.
            estimated_seconds: Total estimated job duration in seconds,
                as produced by ops.estimate_time() with the parameters
                below. Counts down as the ETA while no progress has
                been acknowledged and anchors the pace correction once
                progress arrives. GRBL-style drivers acknowledge
                commands when the firmware buffers them, so short jobs
                complete their streaming long before the machine stops
                moving.
            default_feed_rate: Feed rate assumption for the time model,
                as passed to ops.estimate_time().
            default_rapid_rate: Rapid rate assumption for the time
                model.
            acceleration: Acceleration assumption for the time model.
            clock: Monotonic time source; defaults to time.monotonic.
        """
        self.ops = ops
        self.total_distance = ops.distance()
        self.estimated_seconds = estimated_seconds
        self.traveled_distance = 0.0
        self._clock = clock or time.monotonic
        self.start_time = self._clock()
        self._time_params = (
            default_feed_rate,
            default_rapid_rate,
            acceleration,
        )

        # Modeled execution time (seconds) of the commands up to and
        # including the last acknowledged one.
        self._frontier_seconds = 0.0

        # Create a map from op_index to the distance of that op
        self._distance_map: dict[int, float] = {}
        last_point: Point3D | None = None
        for i in range(ops.len()):
            dist = ops.distance_at(i, last_point)
            self._distance_map[i] = dist
            if ops.category(i) == CommandCategory.MOVING:
                last_point = ops.endpoint(i)

        self.progress_updated = Signal()

    @property
    def metrics(self) -> dict[str, Any]:
        """Returns the current progress metrics as a dictionary."""
        progress_fraction = (
            self.traveled_distance / self.total_distance
            if self.total_distance > 0
            else 1.0
        )
        return {
            "total_distance": self.total_distance,
            "traveled_distance": self.traveled_distance,
            "progress_fraction": progress_fraction,
            "eta_seconds": self._eta_seconds(),
        }

    def _eta_seconds(self) -> float | None:
        """Estimates the remaining job duration in seconds."""
        total = self.estimated_seconds
        if total is None or total <= 0.0:
            return None
        remaining = max(total - self._frontier_seconds, 0.0)
        elapsed = self._clock() - self.start_time
        if remaining <= 0.0:
            return 0.0
        if self._frontier_seconds <= 0.0:
            # Nothing acknowledged yet: count down the estimate. This
            # also covers drivers that never report granular progress.
            return max(total - elapsed, 0.0)

        pace = elapsed / self._frontier_seconds
        pace = min(max(pace, self.MIN_PACE), self.MAX_PACE)
        ramp = (elapsed - self.PACE_WARMUP_SECONDS) / self.PACE_RAMP_SECONDS
        weight = min(max(ramp, 0.0), 1.0)
        return remaining * (1.0 + weight * (pace - 1.0))

    def _model_time_at(self, op_index: int) -> float:
        """The modeled execution time (s) up to and including op_index."""
        if self.estimated_seconds is None:
            return 0.0
        modeled = self.ops.get_cumulative_time_at(op_index, *self._time_params)
        return min(modeled, self.estimated_seconds)

    def update_progress(self, op_index: int) -> None:
        """
        Updates the progress based on a completed operation.

        Args:
            op_index: The index of the Ops command that has finished.
        """
        logger.debug(f"JobMonitor: progress updated for op_index {op_index}.")

        distance_for_op = self._distance_map.get(op_index, 0.0)

        # Always update traveled_distance (even if distance is 0)
        self.traveled_distance += distance_for_op

        # Clamp to ensure we don't exceed total_distance due to float errors
        self.traveled_distance = min(
            self.traveled_distance, self.total_distance
        )

        # The frontier only moves forward: a repeated or reordered
        # index (e.g. after a connection hiccup) must not rewind it.
        self._frontier_seconds = max(
            self._frontier_seconds, self._model_time_at(op_index)
        )

        logger.debug(
            f"  -> New progress: {self.metrics['progress_fraction']:.2f}"
        )
        # Always emit the signal, even for operations with 0 distance
        self.progress_updated.send(self, metrics=self.metrics)

    def mark_as_complete(self) -> None:
        """
        Marks the job as fully complete, setting progress to 100%.
        """
        self.traveled_distance = self.total_distance
        if self.estimated_seconds:
            self._frontier_seconds = self.estimated_seconds
        logger.debug("JobMonitor: marked as complete.")
        self.progress_updated.send(self, metrics=self.metrics)
