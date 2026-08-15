from __future__ import annotations

import math
from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import TYPE_CHECKING

from blinker import Signal
from raygeo.geo import Matrix
from raygeo.geo.types import Rect

from ...context import get_context
from ...core.group import Group
from ...core.item import DocItem
from ...core.layer import Layer
from ...core.stock import StockItem
from ...core.workpiece import WorkPiece
from ...machine.models.machine_panel import PanelOrientation

if TYPE_CHECKING:
    from ...shared.tasker.context import ExecutionContext
    from ...shared.tasker.manager import TaskManager


class LayoutStrategy(ABC):
    """
    Abstract base class for alignment and distribution strategies.

    Each strategy calculates the necessary transformation deltas to apply
    to a list of DocItems to achieve a specific layout.
    """

    def __init__(self, items: Sequence[DocItem], **kwargs):
        if not items:
            raise ValueError("LayoutStrategy requires at least one item.")
        # Filter out items that are descendants of other items in the selection
        # to avoid applying transformations multiple times up the hierarchy.
        self.items = self._filter_descendants(list(items))
        if not self.items:
            raise ValueError(
                "LayoutStrategy requires at least one item after filtering."
            )
        self.error_reported = Signal()

    @staticmethod
    def _filter_descendants(items: Sequence[DocItem]) -> list[DocItem]:
        """
        Given a list of DocItems, returns a new list containing only the
        top-level items from the original list. If an item is a descendant
        of another item in the list, it is excluded.
        """
        # Create a set of all items for efficient lookup.
        item_set = set(items)
        top_level_items = []

        for item in items:
            is_descendant = False
            p = item.parent
            while p:
                if p in item_set:
                    is_descendant = True
                    break
                p = p.parent
            if not is_descendant:
                top_level_items.append(item)
        return top_level_items

    @staticmethod
    def _get_item_world_bbox(
        item: DocItem,
    ) -> Rect | None:
        """
        Calculates the axis-aligned bounding box (min_x, min_y, max_x, max_y)
        of a single DocItem (WorkPiece, Group, or StockItem) in world (mm)
        coordinates.
        """

        items_to_measure = []
        if isinstance(item, WorkPiece):
            items_to_measure.append(item)
        elif isinstance(item, (Group, Layer)):
            items_to_measure.extend(item.get_descendants(of_type=WorkPiece))
        elif isinstance(item, StockItem):
            items_to_measure.append(item)
        else:
            return None

        if not items_to_measure:
            return None

        all_corners = []
        for sub_item in items_to_measure:
            transform = sub_item.get_world_transform()
            # Each workpiece's local geometry is a 1x1 unit square
            local_corners = [(0, 0), (1, 0), (1, 1), (0, 1)]
            all_corners.extend(
                [transform.transform_point(p) for p in local_corners]
            )

        if not all_corners:
            return None

        min_x = min(p[0] for p in all_corners)
        min_y = min(p[1] for p in all_corners)
        max_x = max(p[0] for p in all_corners)
        max_y = max(p[1] for p in all_corners)
        return (min_x, min_y, max_x, max_y)

    def _get_selection_world_bbox(
        self,
    ) -> Rect | None:
        """
        Calculates the collective world-space bounding box for all
        items. Returns (min_x, min_y, max_x, max_y).
        """
        overall_min_x, overall_max_x = float("inf"), float("-inf")
        overall_min_y, overall_max_y = float("inf"), float("-inf")

        for item in self.items:
            bbox = self._get_item_world_bbox(item)
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

    # -- Panel-aware helpers ------------------------------------------
    #
    # The document model lives in WORLD space while the 2D canvas
    # presents PANEL space (the optional 90-degree presentation
    # rotation). Layout operations act on what the user sees, so the
    # bounding boxes and targets are computed in PANEL space and the
    # resulting deltas are un-rotated back into WORLD space before they
    # are applied to the item matrices.

    def _get_panel(self):
        """The machine's display panel, or ``None`` when no machine is
        configured."""
        machine = get_context().machine
        if machine is None:
            return None
        return getattr(machine, "panel", None)

    def _get_item_panel_bbox(self, item: DocItem) -> Rect | None:
        """Panel-space bounding box of a single item.

        Projects the item's WORLD bounding box through the presentation
        rotation. Falls back to the world box when the panel is
        unavailable or unrotated.
        """
        bbox = self._get_item_world_bbox(item)
        if bbox is None:
            return None
        panel = self._get_panel()
        if panel is None or not self._is_panel_rotated(panel):
            return bbox
        return panel.world_bbox_to_panel(bbox)

    def _get_selection_panel_bbox(self) -> Rect | None:
        """Collective PANEL-space bounding box for all items."""
        overall: Rect | None = None
        for item in self.items:
            bbox = self._get_item_panel_bbox(item)
            if bbox is None:
                continue
            if overall is None:
                overall = bbox
            else:
                overall = (
                    min(overall[0], bbox[0]),
                    min(overall[1], bbox[1]),
                    max(overall[2], bbox[2]),
                    max(overall[3], bbox[3]),
                )
        return overall

    @staticmethod
    def _is_panel_rotated(panel) -> bool:
        """True when the panel presents a rotated bed."""
        orientation = getattr(panel, "orientation", None)
        return isinstance(orientation, PanelOrientation) and (
            orientation is not PanelOrientation.NATIVE
        )

    def _world_delta(self, dx: float, dy: float) -> Matrix:
        """Delta matrix for a PANEL-space translation.

        The delta is un-rotated into WORLD space so it composes with the
        item's canonical matrix the same way it would move on screen.
        """
        panel = self._get_panel()
        if panel is not None and self._is_panel_rotated(panel):
            wx, wy = panel.panel_delta_to_world(dx, dy)
            return Matrix.translation(wx, wy)
        return Matrix.translation(dx, dy)

    @abstractmethod
    def calculate_deltas(
        self, context: ExecutionContext | None = None
    ) -> dict[DocItem, Matrix]:
        """
        Calculates the required delta transformation matrix for each
        item.

        Returns:
            A dictionary mapping each DocItem to a delta Matrix that,
            when pre-multiplied with the item's current matrix, will
            move it to the target position.
        """

    async def calculate_deltas_async(
        self,
        context: ExecutionContext | None = None,
        task_manager: TaskManager | None = None,
    ) -> dict[DocItem, Matrix]:
        """
        Asynchronous version of calculate_deltas.

        Default implementation raises NotImplementedError. Subclasses can
        override this to provide async implementations.

        Returns:
            A dictionary mapping each DocItem to a delta Matrix.
        """
        raise NotImplementedError(
            "This layout strategy does not support async calculation"
        )
