from __future__ import annotations
import math
from abc import ABC, abstractmethod
from typing import List, Dict, Optional, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    from ...core.matrix import Matrix
    from ...core.workpiece import WorkPiece
    from ...shared.tasker.context import ExecutionContext


class LayoutStrategy(ABC):
    """
    Abstract base class for alignment and distribution strategies.

    Each strategy calculates the necessary transformation deltas to apply
    to a list of workpieces to achieve a specific layout.
    """

    def __init__(self, workpieces: List[WorkPiece]):
        if not workpieces:
            raise ValueError("LayoutStrategy requires at least one workpiece.")
        self.workpieces = workpieces

    @staticmethod
    def _get_workpiece_world_bbox(
        wp: WorkPiece,
    ) -> Optional[Tuple[float, float, float, float]]:
        """
        Calculates the axis-aligned bounding box (min_x, min_y, max_x, max_y)
        of a single workpiece in world (mm) coordinates.
        """
        from ...core.matrix import Matrix  # noqa
        transform = wp.get_world_transform()
        # The workpiece's local geometry is a 1x1 unit square
        local_corners = [(0, 0), (1, 0), (1, 1), (0, 1)]
        world_corners = [transform.transform_point(p) for p in local_corners]

        min_x = min(p[0] for p in world_corners)
        min_y = min(p[1] for p in world_corners)
        max_x = max(p[0] for p in world_corners)
        max_y = max(p[1] for p in world_corners)
        return (min_x, min_y, max_x, max_y)

    def _get_selection_world_bbox(
        self,
    ) -> Optional[Tuple[float, float, float, float]]:
        """
        Calculates the collective world-space bounding box for all
        workpieces. Returns (min_x, min_y, max_x, max_y).
        """
        overall_min_x, overall_max_x = float("inf"), float("-inf")
        overall_min_y, overall_max_y = float("inf"), float("-inf")

        for wp in self.workpieces:
            bbox = self._get_workpiece_world_bbox(wp)
            if not bbox:
                continue
            min_x, min_y, max_x, max_y = bbox
            overall_min_x = min(overall_min_x, min_x)
            overall_max_x = max(overall_max_x, max_x)
            overall_min_y = min(overall_min_y, min_y)
            overall_max_y = max(overall_max_y, max_y)

        if math.isinf(overall_min_x):
            return None
        return (overall_min_x, overall_min_y, overall_max_x, overall_max_y)

    @abstractmethod
    def calculate_deltas(
        self, context: Optional[ExecutionContext] = None
    ) -> Dict[WorkPiece, Matrix]:
        """
        Calculates the required delta transformation matrix for each
        workpiece.

        Returns:
            A dictionary mapping each WorkPiece to a delta Matrix that,
            when pre-multiplied with the workpiece's current matrix, will
            move it to the target position.
        """
        pass
