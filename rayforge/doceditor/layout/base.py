from __future__ import annotations
import math
from abc import ABC, abstractmethod
from typing import Dict, Optional, Sequence, Tuple, TYPE_CHECKING
from ...core.matrix import Matrix
from ...core.item import DocItem
from ...core.workpiece import WorkPiece
from ...core.group import Group

if TYPE_CHECKING:
    from ...shared.tasker.context import ExecutionContext


class LayoutStrategy(ABC):
    """
    Abstract base class for alignment and distribution strategies.

    Each strategy calculates the necessary transformation deltas to apply
    to a list of DocItems to achieve a specific layout.
    """

    def __init__(self, items: Sequence[DocItem]):
        if not items:
            raise ValueError("LayoutStrategy requires at least one item.")
        # Filter out items that are descendants of other items in the selection
        # to avoid applying transformations multiple times up the hierarchy.
        self.items = self._filter_descendants(list(items))
        if not self.items:
            raise ValueError(
                "LayoutStrategy requires at least one item after filtering."
            )

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
    ) -> Optional[Tuple[float, float, float, float]]:
        """
        Calculates the axis-aligned bounding box (min_x, min_y, max_x, max_y)
        of a single DocItem (WorkPiece or Group) in world (mm) coordinates.
        """

        items_to_measure = []
        if isinstance(item, WorkPiece):
            items_to_measure.append(item)
        elif isinstance(item, Group):
            # For a group, get all descendant workpieces to measure
            items_to_measure.extend(item.get_descendants(of_type=WorkPiece))
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
    ) -> Optional[Tuple[float, float, float, float]]:
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

    @abstractmethod
    def calculate_deltas(
        self, context: Optional[ExecutionContext] = None
    ) -> Dict[DocItem, Matrix]:
        """
        Calculates the required delta transformation matrix for each
        item.

        Returns:
            A dictionary mapping each DocItem to a delta Matrix that,
            when pre-multiplied with the item's current matrix, will
            move it to the target position.
        """
        pass
