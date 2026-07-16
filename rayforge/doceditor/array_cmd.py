"""
ArrayCmd: the document command handler for the Array / Pattern tool.

It duplicates a (possibly multi-layer) selection of items into a
regular pattern. Each duplicate is placed on the same layer as its
source item, preserving the document's layer structure.
"""

from __future__ import annotations

import logging
import math
from gettext import gettext as _
from typing import TYPE_CHECKING, List, Optional, Sequence

from raygeo.geo import Matrix
from raygeo.geo.types import Rect

from ..core.group import Group
from ..core.item import DocItem
from ..core.undo import ListItemCommand
from ..core.workpiece import WorkPiece
from .array import ArrayParams, make_array_strategy
from .layout.base import LayoutStrategy
from .transform_cmd import TransformCmd

if TYPE_CHECKING:
    from .editor import DocEditor

logger = logging.getLogger(__name__)


class ArrayCmd:
    """Handles undoable creation of item arrays."""

    def __init__(self, editor: "DocEditor"):
        self._editor = editor

    # ------------------------------------------------------------------
    # Pure helpers (no mutation) - shared with the live preview.
    # ------------------------------------------------------------------
    @staticmethod
    def _get_top_level_items(
        items: Sequence[DocItem],
    ) -> List[DocItem]:
        """Returns only the top-level items from a selection.

        If an item and one of its ancestors are both selected, only the
        ancestor is arrayed, so descendants are not duplicated twice.
        """
        if not items:
            return []
        item_set = set(items)
        top_level: List[DocItem] = []
        for item in items:
            ancestor = item.parent
            while ancestor:
                if ancestor in item_set:
                    break
                ancestor = ancestor.parent
            else:
                top_level.append(item)
        return top_level

    @staticmethod
    def _compute_unit_bbox(items: Sequence[DocItem]) -> Optional[Rect]:
        """Collective world-space bbox of the selection as one unit."""
        min_x = min_y = math.inf
        max_x = max_y = -math.inf
        for item in items:
            bbox = LayoutStrategy._get_item_world_bbox(item)
            if not bbox:
                continue
            min_x = min(min_x, bbox[0])
            min_y = min(min_y, bbox[1])
            max_x = max(max_x, bbox[2])
            max_y = max(max_y, bbox[3])
        if math.isinf(min_x):
            return None
        return (min_x, min_y, max_x, max_y)

    def compute_plan(
        self,
        source_items: Sequence[DocItem],
        params: ArrayParams,
    ) -> List[Matrix]:
        """Returns the list of world-space delta matrices for the array.

        This is pure: it neither reads nor mutates document state beyond
        the source items' current transforms. Instance 0 is the identity
        (the original selection). Used by the live preview.
        """
        top_level = self._get_top_level_items(source_items)
        unit_bbox = self._compute_unit_bbox(top_level)
        if unit_bbox is None:
            return []
        strategy = make_array_strategy(unit_bbox, params)
        return strategy.calculate_placements()

    def get_selection_bbox(
        self, source_items: Sequence[DocItem]
    ) -> Optional[Rect]:
        """Returns the collective world bbox of the selection, or None.

        Convenience wrapper for the live preview.
        """
        return self._compute_unit_bbox(self._get_top_level_items(source_items))

    # ------------------------------------------------------------------
    # Commit path (undoable mutation).
    # ------------------------------------------------------------------
    def create_array(
        self,
        source_items: Sequence[DocItem],
        params: ArrayParams,
    ) -> List[DocItem]:
        """Duplicates the selection into the array in one transaction.

        The original selection is kept in place as the first (identity)
        instance; one copy per remaining instance is created, each on
        the same layer as its source item.

        Returns the newly created top-level items.
        """
        top_level = self._get_top_level_items(source_items)
        if not top_level:
            return []

        deltas = self.compute_plan(top_level, params)
        if len(deltas) <= 1:
            return []

        history = self._editor.history_manager
        created: List[DocItem] = []

        with history.transaction(_("Create Array")) as t:
            for delta in deltas:
                if _is_identity(delta):
                    # The original selection occupies this cell.
                    continue
                for item in top_level:
                    layer = _get_item_layer(item)
                    if layer is None:
                        logger.warning(
                            "Item '%s' has no layer; skipping.", item.name
                        )
                        continue
                    copy = self._make_copy_at_delta(item, delta)
                    cmd = ListItemCommand(
                        owner_obj=layer,
                        item=copy,
                        undo_command="remove_child",
                        redo_command="add_child",
                        name=_("Create array copy"),
                    )
                    t.execute(cmd)
                    created.append(copy)
        return created

    @staticmethod
    def _make_copy_at_delta(item: DocItem, delta_world: Matrix) -> DocItem:
        """Duplicates ``item`` and applies a world-space delta to the copy.

        The copy keeps the source item's parent layer, so the world
        delta is converted back into the copy's local matrix relative
        to that unchanged parent.
        """
        copy = item.duplicate()
        old_world = item.get_world_transform()
        new_world = delta_world @ old_world
        copy.matrix = TransformCmd._world_to_local_matrix(item, new_world)
        return copy


def _is_identity(matrix: Matrix) -> bool:
    """True if the matrix has no practical effect."""
    return matrix.is_identity()


def _get_item_layer(item: DocItem):
    """Returns the layer owning ``item`` (WorkPiece or Group), or None."""
    if isinstance(item, (WorkPiece, Group)):
        return item.layer
    return None
