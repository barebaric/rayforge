from __future__ import annotations
from typing import Dict, Optional, TYPE_CHECKING
from .base import LayoutStrategy

if TYPE_CHECKING:
    from ...core.matrix import Matrix
    from ...core.item import DocItem
    from ...shared.tasker.context import ExecutionContext


class SpreadHorizontallyStrategy(LayoutStrategy):
    """Distributes items evenly in the horizontal direction."""

    def calculate_deltas(
        self, context: Optional[ExecutionContext] = None
    ) -> Dict[DocItem, Matrix]:
        if len(self.items) < 3:
            return {}

        wps_with_bboxes = []
        for wp in self.items:
            bbox = self._get_item_world_bbox(wp)
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
                from ...core.matrix import Matrix

                deltas[wp] = Matrix.translation(delta_x, 0)

            item_width = bbox[2] - bbox[0]
            current_x = target_min_x + item_width

        return deltas


class SpreadVerticallyStrategy(LayoutStrategy):
    """Distributes items evenly in the vertical direction."""

    def calculate_deltas(
        self, context: Optional[ExecutionContext] = None
    ) -> Dict[DocItem, Matrix]:
        if len(self.items) < 3:
            return {}

        wps_with_bboxes = []
        for wp in self.items:
            bbox = self._get_item_world_bbox(wp)
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
                from ...core.matrix import Matrix

                deltas[wp] = Matrix.translation(0, delta_y)

            item_height = bbox[3] - bbox[1]
            current_y = target_min_y + item_height

        return deltas
