from __future__ import annotations
import logging
from typing import TYPE_CHECKING, List, Tuple
from ..core.item import DocItem
from ..core.matrix import Matrix
from ..undo import ChangePropertyCommand

if TYPE_CHECKING:
    from .editor import DocEditor

logger = logging.getLogger(__name__)


class TransformCmd:
    """Handles undoable transformations of document items."""

    def __init__(self, editor: "DocEditor"):
        self._editor = editor

    def create_transform_transaction(
        self,
        changes: List[Tuple[DocItem, Matrix, Matrix]],
    ):
        """
        Creates a single, undoable transaction for a list of matrix changes
        that have already been calculated.

        Args:
            changes: A list of tuples, where each tuple contains
                (DocItem_to_change, old_matrix, new_matrix).
        """
        history_manager = self._editor.history_manager
        if not changes:
            return

        logger.debug(
            f"Creating transform transaction for {len(changes)} item(s)."
        )

        with history_manager.transaction(_("Transform item(s)")) as t:
            for doc_item, old_matrix, new_matrix in changes:
                cmd = ChangePropertyCommand(
                    target=doc_item,
                    property_name="matrix",
                    new_value=new_matrix,
                    old_value=old_matrix,
                )
                t.execute(cmd)

    def nudge_items(
        self,
        items: List[DocItem],
        dx_mm: float,
        dy_mm: float,
    ):
        """
        Moves a list of DocItems by a given delta in world coordinates,
        creating a single undoable transaction for the operation.

        Args:
            items: The list of DocItems to move.
            dx_mm: The distance to move along the X-axis in millimeters.
            dy_mm: The distance to move along the Y-axis in millimeters.
        """
        history_manager = self._editor.history_manager
        if not items or (dx_mm == 0.0 and dy_mm == 0.0):
            return

        with history_manager.transaction(_("Move item(s)")) as t:
            for item in items:
                old_matrix = item.matrix.copy()
                # Nudge must be pre-multiplied to apply the translation in
                # world space, not local space.
                delta = Matrix.translation(dx_mm, dy_mm)
                new_matrix = delta @ old_matrix
                cmd = ChangePropertyCommand(
                    target=item,
                    property_name="matrix",
                    new_value=new_matrix,
                    old_value=old_matrix,
                )
                t.execute(cmd)
