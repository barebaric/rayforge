from __future__ import annotations
import logging
import time
from typing import TYPE_CHECKING, Dict, Any
from collections import deque
from blinker import Signal
from ..core.ops.commands import MovingCommand

if TYPE_CHECKING:
    from ..core.ops import Ops


logger = logging.getLogger(__name__)


class JobMonitor:
    """
    Tracks and reports the progress of a machine job based on Ops data.

    This class calculates the total distance of a job from an Ops object and
    updates the progress as individual operations complete. It emits a signal
    with detailed metrics whenever the progress changes.
    """

    def __init__(self, ops: "Ops"):
        """
        Initializes the JobMonitor.

        Args:
            ops: The Ops object representing the job to be monitored.
        """
        self.ops = ops
        self.total_distance = ops.distance()
        self.traveled_distance = 0.0
        self.start_time = time.monotonic()

        # Create a map from op_index to the distance of that op
        self._distance_map: Dict[int, float] = {}
        last_point = None
        for i, cmd in enumerate(self.ops):
            # Calculate distance for ANY command. It will be 0 for non-moving
            # ones.
            dist = cmd.distance(last_point)
            self._distance_map[i] = dist
            if isinstance(cmd, MovingCommand):
                last_point = cmd.end

        # Deque for calculating recent average speed.
        # Stores (timestamp, distance).
        # A larger maxlen provides more smoothing but is slower to react to
        # speed changes. 20 is a reasonable starting point.
        self._samples = deque(maxlen=200)

        self.progress_updated = Signal()

    @property
    def metrics(self) -> Dict[str, Any]:
        """Returns the current progress metrics as a dictionary."""
        progress_fraction = (
            self.traveled_distance / self.total_distance
            if self.total_distance > 0
            else 1.0
        )

        eta_seconds = None
        # Calculate ETA based on recent average speed to avoid fluctuations
        # caused by pauses or non-moving commands.
        if len(self._samples) > 1:
            start_time, start_dist = self._samples[0]
            end_time, end_dist = self._samples[-1]

            delta_time = end_time - start_time
            delta_dist = end_dist - start_dist

            if delta_time > 0.01 and delta_dist > 0:
                recent_average_speed = delta_dist / delta_time
                distance_remaining = (
                    self.total_distance - self.traveled_distance
                )
                if recent_average_speed > 0:
                    eta_seconds = distance_remaining / recent_average_speed

        return {
            "total_distance": self.total_distance,
            "traveled_distance": self.traveled_distance,
            "progress_fraction": progress_fraction,
            "eta_seconds": eta_seconds,
        }

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
        if self.traveled_distance > self.total_distance:
            self.traveled_distance = self.total_distance

        # Only add a sample for ETA calculation if there's actual distance
        if distance_for_op > 0.0:
            # Add a new sample for the ETA calculation
            self._samples.append((time.monotonic(), self.traveled_distance))

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
        logger.debug("JobMonitor: marked as complete.")
        self.progress_updated.send(self, metrics=self.metrics)
