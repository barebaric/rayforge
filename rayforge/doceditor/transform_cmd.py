from __future__ import annotations
import logging
from typing import TYPE_CHECKING, List, Dict
from ..core.item import DocItem
from ..core.matrix import Matrix
from ..undo import ChangePropertyCommand
from ..workbench.elements.group import GroupElement

if TYPE_CHECKING:
    from .editor import DocEditor
    from ..workbench.canvas import CanvasElement

logger = logging.getLogger(__name__)


class TransformCmd:
    """Handles undoable transformations of document items."""

    def __init__(self, editor: "DocEditor"):
        self._editor = editor

    def finalize_interactive_transform(
        self,
        elements: List["CanvasElement"],
        transform_start_states: Dict["CanvasElement", dict],
    ):
        """
        Finalizes an interactive transform by collecting all matrix changes
        from view elements and committing them to the data model in a single,
        undoable transaction.

        Args:
            elements: The list of elements that were directly manipulated.
            transform_start_states: A dictionary mapping elements to their
                initial state (e.g., matrix) at the start of the transform.
        """
        history_manager = self._editor.history_manager
        logger.debug(
            f"Finalizing transform for {len(elements)} element(s)."
            " Creating undo transaction."
        )

        # Step 1: Collect all elements that may have changed.
        # The view (e.g., ShrinkWrapGroup) updates its hierarchy automatically.
        # We just need to find all the elements that have actually changed.
        affected_elements = set()
        for element in elements:
            affected_elements.add(element)
            parent = element.parent
            while isinstance(parent, GroupElement):
                affected_elements.add(parent)
                parent = parent.parent

        # Step 2: Create commands for all changes found.
        commands_to_execute = []
        for element in affected_elements:
            if (
                not isinstance(element.data, DocItem)
                or element not in transform_start_states
                or "matrix" not in transform_start_states[element]
            ):
                continue

            docitem: DocItem = element.data
            start_matrix = transform_start_states[element]["matrix"]
            new_matrix = element.transform

            if start_matrix != new_matrix:
                cmd = ChangePropertyCommand(
                    target=docitem,
                    property_name="matrix",
                    new_value=new_matrix.copy(),
                    old_value=start_matrix,
                )
                commands_to_execute.append(cmd)

        # Step 3: Execute all commands in a single transaction.
        if commands_to_execute:
            with history_manager.transaction(_("Transform item(s)")) as t:
                for cmd in commands_to_execute:
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
