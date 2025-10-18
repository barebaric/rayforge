from typing import TYPE_CHECKING, Dict, Any
from blinker import Signal

if TYPE_CHECKING:
    from ..core.ops import Ops


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

        # Create a map from op_index to the distance of that op
        self._distance_map: Dict[int, float] = {}
        last_point = None
        for i, cmd in enumerate(self.ops):
            dist = cmd.distance(last_point)
            self._distance_map[i] = dist
            if hasattr(cmd, "end") and cmd.end is not None:
                last_point = cmd.end

        self.progress_updated = Signal()

    @property
    def metrics(self) -> Dict[str, Any]:
        """Returns the current progress metrics as a dictionary."""
        return {
            "total_distance": self.total_distance,
            "traveled_distance": self.traveled_distance,
            "progress_fraction": (
                self.traveled_distance / self.total_distance
                if self.total_distance > 0
                else 1.0
            ),
        }

    def update_progress(self, op_index: int) -> None:
        """
        Updates the progress based on a completed operation.

        Args:
            op_index: The index of the Ops command that has finished.
        """

        distance_for_op = self._distance_map.get(op_index, 0.0)
        self.traveled_distance += distance_for_op

        # Clamp to ensure we don't exceed total_distance due to float errors
        if self.traveled_distance > self.total_distance:
            self.traveled_distance = self.total_distance

        self.progress_updated.send(self, metrics=self.metrics)

    def mark_as_complete(self) -> None:
        """
        Marks the job as fully complete, setting progress to 100%.
        """
        self.traveled_distance = self.total_distance
        self.progress_updated.send(self, metrics=self.metrics)
