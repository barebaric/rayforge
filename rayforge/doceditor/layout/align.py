from __future__ import annotations
from typing import Dict, Optional, TYPE_CHECKING, Sequence
from ...core.matrix import Matrix
from .base import LayoutStrategy

if TYPE_CHECKING:
    from ...core.item import DocItem
    from ...shared.tasker.context import ExecutionContext


class BboxAlignLeftStrategy(LayoutStrategy):
    """Aligns the left edges of the selection's bounding boxes."""

    def calculate_deltas(
        self, context: Optional[ExecutionContext] = None
    ) -> Dict[DocItem, Matrix]:
        target_x: float
        if len(self.items) == 1:
            # For a single item, align to the world origin's left edge.
            target_x = 0.0
        else:
            bbox = self._get_selection_world_bbox()
            if not bbox:
                return {}
            target_x = bbox[0]  # Align to selection's left edge

        deltas = {}
        for wp in self.items:
            wp_bbox = self._get_item_world_bbox(wp)
            if not wp_bbox:
                continue
            delta_x = target_x - wp_bbox[0]
            if abs(delta_x) > 1e-6:
                deltas[wp] = Matrix.translation(delta_x, 0)
        return deltas


class BboxAlignCenterStrategy(LayoutStrategy):
    """Horizontally centers the selection's bounding boxes."""

    def __init__(
        self,
        items: Sequence[DocItem],
        surface_width_mm: Optional[float] = None,
    ):
        super().__init__(items)
        self.surface_width_mm = surface_width_mm

    def calculate_deltas(
        self, context: Optional[ExecutionContext] = None
    ) -> Dict[DocItem, Matrix]:
        target_center_x: float
        if len(self.items) == 1 and self.surface_width_mm is not None:
            target_center_x = self.surface_width_mm / 2
        else:
            bbox = self._get_selection_world_bbox()
            if not bbox:
                return {}
            target_center_x = bbox[0] + (bbox[2] - bbox[0]) / 2

        deltas = {}
        for wp in self.items:
            wp_bbox = self._get_item_world_bbox(wp)
            if not wp_bbox:
                continue
            wp_center_x = wp_bbox[0] + (wp_bbox[2] - wp_bbox[0]) / 2
            delta_x = target_center_x - wp_center_x
            if abs(delta_x) > 1e-6:
                deltas[wp] = Matrix.translation(delta_x, 0)
        return deltas


class BboxAlignRightStrategy(LayoutStrategy):
    """Aligns the right edges of the selection's bounding boxes."""

    def __init__(
        self,
        items: Sequence[DocItem],
        surface_width_mm: Optional[float] = None,
    ):
        super().__init__(items)
        self.surface_width_mm = surface_width_mm

    def calculate_deltas(
        self, context: Optional[ExecutionContext] = None
    ) -> Dict[DocItem, Matrix]:
        target_x: float
        if len(self.items) == 1 and self.surface_width_mm is not None:
            target_x = self.surface_width_mm
        else:
            bbox = self._get_selection_world_bbox()
            if not bbox:
                return {}
            target_x = bbox[2]  # Right edge of collective box

        deltas = {}
        for wp in self.items:
            wp_bbox = self._get_item_world_bbox(wp)
            if not wp_bbox:
                continue
            delta_x = target_x - wp_bbox[2]
            if abs(delta_x) > 1e-6:
                deltas[wp] = Matrix.translation(delta_x, 0)
        return deltas


class BboxAlignTopStrategy(LayoutStrategy):
    """Aligns the top edges of the selection's bounding boxes."""

    def __init__(
        self,
        items: Sequence[DocItem],
        surface_height_mm: Optional[float] = None,
    ):
        super().__init__(items)
        self.surface_height_mm = surface_height_mm

    def calculate_deltas(
        self, context: Optional[ExecutionContext] = None
    ) -> Dict[DocItem, Matrix]:
        target_y: float
        if len(self.items) == 1 and self.surface_height_mm is not None:
            target_y = self.surface_height_mm
        else:
            bbox = self._get_selection_world_bbox()
            if not bbox:
                return {}
            target_y = bbox[3]  # Top edge of collective box

        deltas = {}
        for wp in self.items:
            wp_bbox = self._get_item_world_bbox(wp)
            if not wp_bbox:
                continue
            delta_y = target_y - wp_bbox[3]
            if abs(delta_y) > 1e-6:
                deltas[wp] = Matrix.translation(0, delta_y)
        return deltas


class BboxAlignMiddleStrategy(LayoutStrategy):
    """Vertically centers the selection's bounding boxes."""

    def __init__(
        self,
        items: Sequence[DocItem],
        surface_height_mm: Optional[float] = None,
    ):
        super().__init__(items)
        self.surface_height_mm = surface_height_mm

    def calculate_deltas(
        self, context: Optional[ExecutionContext] = None
    ) -> Dict[DocItem, Matrix]:
        target_center_y: float
        if len(self.items) == 1 and self.surface_height_mm is not None:
            target_center_y = self.surface_height_mm / 2
        else:
            bbox = self._get_selection_world_bbox()
            if not bbox:
                return {}
            target_center_y = bbox[1] + (bbox[3] - bbox[1]) / 2

        deltas = {}
        for wp in self.items:
            wp_bbox = self._get_item_world_bbox(wp)
            if not wp_bbox:
                continue
            wp_center_y = wp_bbox[1] + (wp_bbox[3] - wp_bbox[1]) / 2
            delta_y = target_center_y - wp_center_y
            if abs(delta_y) > 1e-6:
                deltas[wp] = Matrix.translation(0, delta_y)
        return deltas


class BboxAlignBottomStrategy(LayoutStrategy):
    """Aligns the bottom edges of the selection's bounding boxes."""

    def calculate_deltas(
        self, context: Optional[ExecutionContext] = None
    ) -> Dict[DocItem, Matrix]:
        target_y: float
        if len(self.items) == 1:
            target_y = 0.0
        else:
            bbox = self._get_selection_world_bbox()
            if not bbox:
                return {}
            target_y = bbox[1]  # Bottom edge of collective box

        deltas = {}
        for wp in self.items:
            wp_bbox = self._get_item_world_bbox(wp)
            if not wp_bbox:
                continue
            delta_y = target_y - wp_bbox[1]
            if abs(delta_y) > 1e-6:
                deltas[wp] = Matrix.translation(0, delta_y)
        return deltas
