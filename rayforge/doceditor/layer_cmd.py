from __future__ import annotations
import logging
from typing import TYPE_CHECKING, Optional, List, cast
from ..core.layer import Layer
from ..core.workpiece import WorkPiece
from ..undo import Command

if TYPE_CHECKING:
    from ..workbench.surface import WorkSurface

logger = logging.getLogger(__name__)


class MoveWorkpiecesLayerCommand(Command):
    """
    An undoable command to move one or more workpieces to a different layer.
    """

    def __init__(
        self,
        canvas: "WorkSurface",
        workpieces: List[WorkPiece],
        new_layer: Layer,
        old_layer: Layer,
        name: Optional[str] = None,
    ):
        super().__init__(name)
        self.canvas = canvas
        self.workpieces = workpieces
        self.new_layer = new_layer
        self.old_layer = old_layer
        if not name:
            self.name = _("Move to another layer")

    def _move(self, from_layer: Layer, to_layer: Layer):
        """The core logic for moving workpieces, view-first."""
        from ..workbench.elements.layer import LayerElement

        from_layer_elem = cast(
            Optional[LayerElement], self.canvas.find_by_data(from_layer)
        )
        to_layer_elem = cast(
            Optional[LayerElement], self.canvas.find_by_data(to_layer)
        )

        if not from_layer_elem or not to_layer_elem:
            logger.warning("Could not find layer elements for move operation.")
            return

        # Step 1 & 2: UI-first, flicker-free re-parenting
        elements_to_move = []
        for wp in self.workpieces:
            wp_elem = self.canvas.find_by_data(wp)
            if wp_elem:
                elements_to_move.append(wp_elem)

        for elem in elements_to_move:
            to_layer_elem.add(elem)

        # Enforce correct Z-order immediately after the move.
        from_layer_elem.sort_children_by_z_order()
        to_layer_elem.sort_children_by_z_order()
        self.canvas.queue_draw()

        # Step 3: Update the model
        # The signals will trigger a non-destructive reconciliation.
        for wp in self.workpieces:
            if wp.parent:
                wp.parent.remove_child(wp)
            to_layer.add_child(wp)

    def execute(self):
        """Executes the command, moving workpieces to the new layer."""
        self._move(self.old_layer, self.new_layer)

    def undo(self):
        """Undoes the command, moving workpieces back to the old layer."""
        self._move(self.new_layer, self.old_layer)


def move_selected_to_adjacent_layer(surface: "WorkSurface", direction: int):
    """
    Creates an undoable command to move selected workpieces to the
    next or previous layer.

    Args:
        surface: The WorkSurface instance containing the selection.
        direction: 1 for the next layer (down), -1 for the previous (up).
    """
    selected_wps = surface.get_selected_workpieces()
    if not selected_wps:
        return

    doc = surface.doc
    layers = doc.layers
    if len(layers) <= 1:
        return

    # Assume all selected workpieces are on the same layer, which is a
    # reasonable constraint for this operation.
    current_layer = selected_wps[0].layer
    if not current_layer:
        return

    try:
        current_index = layers.index(current_layer)
        # Wrap around the layer list
        new_index = (current_index + direction + len(layers)) % len(layers)
        new_layer = layers[new_index]

        cmd = MoveWorkpiecesLayerCommand(
            surface, selected_wps, new_layer, current_layer
        )
        doc.history_manager.execute(cmd)

    except ValueError:
        logger.warning(
            f"Layer '{current_layer.name}' not found in document's layer list."
        )
