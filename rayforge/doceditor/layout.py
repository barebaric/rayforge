from __future__ import annotations
import math
from abc import ABC, abstractmethod
from typing import List, Dict, Optional, Tuple, TYPE_CHECKING
from ..core.matrix import Matrix
from ..core.workpiece import WorkPiece

if TYPE_CHECKING:
    from ..workbench.surface import WorkSurface


class LayoutStrategy(ABC):
    """
    Abstract base class for alignment and distribution strategies.

    Each strategy calculates the necessary transformation deltas to apply
    to a list of workpieces to achieve a specific layout.
    """

    def __init__(self, workpieces: List[WorkPiece], surface: "WorkSurface"):
        if not workpieces:
            raise ValueError("LayoutStrategy requires at least one workpiece.")
        self.workpieces = workpieces
        self.surface = surface
        self.doc = surface.doc

    @staticmethod
    def _get_workpiece_world_bbox(
        wp: WorkPiece,
    ) -> Optional[Tuple[float, float, float, float]]:
        """
        Calculates the axis-aligned bounding box (min_x, min_y, max_x, max_y)
        of a single workpiece in world (mm) coordinates.
        """
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
    def calculate_deltas(self) -> Dict[WorkPiece, Matrix]:
        """
        Calculates the required delta transformation matrix for each
        workpiece.

        Returns:
            A dictionary mapping each WorkPiece to a delta Matrix that,
            when pre-multiplied with the workpiece's current matrix, will
            move it to the target position.
        """
        pass


class BboxAlignLeftStrategy(LayoutStrategy):
    """Aligns the left edges of the selection's bounding boxes."""

    def calculate_deltas(self) -> Dict[WorkPiece, Matrix]:
        target_x: float
        if len(self.workpieces) == 1:
            target_x = 0.0  # Align to canvas edge
        else:
            bbox = self._get_selection_world_bbox()
            if not bbox:
                return {}
            target_x = bbox[0]  # Align to selection's left edge

        deltas = {}
        for wp in self.workpieces:
            wp_bbox = self._get_workpiece_world_bbox(wp)
            if not wp_bbox:
                continue
            delta_x = target_x - wp_bbox[0]
            if abs(delta_x) > 1e-6:
                deltas[wp] = Matrix.translation(delta_x, 0)
        return deltas


class BboxAlignCenterStrategy(LayoutStrategy):
    """Horizontally centers the selection's bounding boxes."""

    def calculate_deltas(self) -> Dict[WorkPiece, Matrix]:
        target_center_x: float
        if len(self.workpieces) == 1:
            target_center_x = self.surface.width_mm / 2
        else:
            bbox = self._get_selection_world_bbox()
            if not bbox:
                return {}
            target_center_x = bbox[0] + (bbox[2] - bbox[0]) / 2

        deltas = {}
        for wp in self.workpieces:
            wp_bbox = self._get_workpiece_world_bbox(wp)
            if not wp_bbox:
                continue
            wp_center_x = wp_bbox[0] + (wp_bbox[2] - wp_bbox[0]) / 2
            delta_x = target_center_x - wp_center_x
            if abs(delta_x) > 1e-6:
                deltas[wp] = Matrix.translation(delta_x, 0)
        return deltas


class BboxAlignRightStrategy(LayoutStrategy):
    """Aligns the right edges of the selection's bounding boxes."""

    def calculate_deltas(self) -> Dict[WorkPiece, Matrix]:
        target_x: float
        if len(self.workpieces) == 1:
            target_x = self.surface.width_mm
        else:
            bbox = self._get_selection_world_bbox()
            if not bbox:
                return {}
            target_x = bbox[2]  # Right edge of collective box

        deltas = {}
        for wp in self.workpieces:
            wp_bbox = self._get_workpiece_world_bbox(wp)
            if not wp_bbox:
                continue
            delta_x = target_x - wp_bbox[2]
            if abs(delta_x) > 1e-6:
                deltas[wp] = Matrix.translation(delta_x, 0)
        return deltas


class BboxAlignTopStrategy(LayoutStrategy):
    """Aligns the top edges of the selection's bounding boxes."""

    def calculate_deltas(self) -> Dict[WorkPiece, Matrix]:
        target_y: float
        if len(self.workpieces) == 1:
            target_y = self.surface.height_mm
        else:
            bbox = self._get_selection_world_bbox()
            if not bbox:
                return {}
            target_y = bbox[3]  # Top edge of collective box

        deltas = {}
        for wp in self.workpieces:
            wp_bbox = self._get_workpiece_world_bbox(wp)
            if not wp_bbox:
                continue
            delta_y = target_y - wp_bbox[3]
            if abs(delta_y) > 1e-6:
                deltas[wp] = Matrix.translation(0, delta_y)
        return deltas


class BboxAlignMiddleStrategy(LayoutStrategy):
    """Vertically centers the selection's bounding boxes."""

    def calculate_deltas(self) -> Dict[WorkPiece, Matrix]:
        target_center_y: float
        if len(self.workpieces) == 1:
            target_center_y = self.surface.height_mm / 2
        else:
            bbox = self._get_selection_world_bbox()
            if not bbox:
                return {}
            target_center_y = bbox[1] + (bbox[3] - bbox[1]) / 2

        deltas = {}
        for wp in self.workpieces:
            wp_bbox = self._get_workpiece_world_bbox(wp)
            if not wp_bbox:
                continue
            wp_center_y = wp_bbox[1] + (wp_bbox[3] - wp_bbox[1]) / 2
            delta_y = target_center_y - wp_center_y
            if abs(delta_y) > 1e-6:
                deltas[wp] = Matrix.translation(0, delta_y)
        return deltas


class BboxAlignBottomStrategy(LayoutStrategy):
    """Aligns the bottom edges of the selection's bounding boxes."""

    def calculate_deltas(self) -> Dict[WorkPiece, Matrix]:
        target_y: float
        if len(self.workpieces) == 1:
            target_y = 0.0
        else:
            bbox = self._get_selection_world_bbox()
            if not bbox:
                return {}
            target_y = bbox[1]  # Bottom edge of collective box

        deltas = {}
        for wp in self.workpieces:
            wp_bbox = self._get_workpiece_world_bbox(wp)
            if not wp_bbox:
                continue
            delta_y = target_y - wp_bbox[1]
            if abs(delta_y) > 1e-6:
                deltas[wp] = Matrix.translation(0, delta_y)
        return deltas


class SpreadHorizontallyStrategy(LayoutStrategy):
    """Distributes workpieces evenly in the horizontal direction."""

    def calculate_deltas(self) -> Dict[WorkPiece, Matrix]:
        if len(self.workpieces) < 3:
            return {}

        wps_with_bboxes = []
        for wp in self.workpieces:
            bbox = self._get_workpiece_world_bbox(wp)
            if bbox:
                wps_with_bboxes.append((wp, bbox))

        if len(wps_with_bboxes) < 3:
            return {}

        # Sort by the center x of the bounding box
        wps_with_bboxes.sort(key=lambda item: (item[1][0] + item[1][2]) / 2)

        leftmost_bbox = wps_with_bboxes[0][1]
        rightmost_bbox = wps_with_bboxes[-1][1]

        total_span = rightmost_bbox[2] - leftmost_bbox[0]
        total_items_width = sum(
            bbox[2] - bbox[0] for _, bbox in wps_with_bboxes
        )
        total_gap_space = total_span - total_items_width
        gap_size = total_gap_space / (len(wps_with_bboxes) - 1)

        deltas = {}
        current_x = leftmost_bbox[2]
        for wp, bbox in wps_with_bboxes[1:-1]:
            target_min_x = current_x + gap_size
            delta_x = target_min_x - bbox[0]
            if abs(delta_x) > 1e-6:
                deltas[wp] = Matrix.translation(delta_x, 0)

            item_width = bbox[2] - bbox[0]
            current_x = target_min_x + item_width

        return deltas


class SpreadVerticallyStrategy(LayoutStrategy):
    """Distributes workpieces evenly in the vertical direction."""

    def calculate_deltas(self) -> Dict[WorkPiece, Matrix]:
        if len(self.workpieces) < 3:
            return {}

        wps_with_bboxes = []
        for wp in self.workpieces:
            bbox = self._get_workpiece_world_bbox(wp)
            if bbox:
                wps_with_bboxes.append((wp, bbox))

        if len(wps_with_bboxes) < 3:
            return {}

        # Sort by the center y of the bounding box
        wps_with_bboxes.sort(key=lambda item: (item[1][1] + item[1][3]) / 2)

        bottommost_bbox = wps_with_bboxes[0][1]
        topmost_bbox = wps_with_bboxes[-1][1]

        total_span = topmost_bbox[3] - bottommost_bbox[1]
        total_items_height = sum(
            bbox[3] - bbox[1] for _, bbox in wps_with_bboxes
        )
        total_gap_space = total_span - total_items_height
        gap_size = total_gap_space / (len(wps_with_bboxes) - 1)

        deltas = {}
        current_y = bottommost_bbox[3]
        for wp, bbox in wps_with_bboxes[1:-1]:
            target_min_y = current_y + gap_size
            delta_y = target_min_y - bbox[1]
            if abs(delta_y) > 1e-6:
                deltas[wp] = Matrix.translation(0, delta_y)

            item_height = bbox[3] - bbox[1]
            current_y = target_min_y + item_height

        return deltas
