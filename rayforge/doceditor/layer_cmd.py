from __future__ import annotations
import logging
from typing import TYPE_CHECKING, Optional, List
from ..core.layer import Layer
from ..core.workpiece import WorkPiece
from ..undo import Command
from ..core.stocklayer import StockLayer

if TYPE_CHECKING:
    from ..workbench.surface import WorkSurface
    from .editor import DocEditor

logger = logging.getLogger(__name__)


class MoveWorkpiecesLayerCommand(Command):
    """
    An undoable command to move one or more workpieces to a different layer.
    """

    def __init__(
        self,
        workpieces: List[WorkPiece],
        new_layer: Layer,
        old_layer: Layer,
        name: Optional[str] = None,
    ):
        super().__init__(name)
        self.workpieces = workpieces
        self.new_layer = new_layer
        self.old_layer = old_layer
        if not name:
            self.name = _("Move to another layer")

    def _move(self, from_layer: Layer, to_layer: Layer):
        """The core logic for moving workpieces, model-only."""
        # The UI will react to the model changes automatically through signals.
        # The DocItem.add_child() method handles removing the child from its
        # previous parent.
        for wp in self.workpieces:
            to_layer.add_child(wp)

    def execute(self):
        """Executes the command, moving workpieces to the new layer."""
        self._move(self.old_layer, self.new_layer)

    def undo(self):
        """Undoes the command, moving workpieces back to the old layer."""
        self._move(self.new_layer, self.old_layer)


class LayerCmd:
    """Handles commands related to layer manipulation."""

    def __init__(self, editor: "DocEditor"):
        self._editor = editor

    def move_selected_to_adjacent_layer(
        self, surface: "WorkSurface", direction: int
    ):
        """
        Creates an undoable command to move selected workpieces to the
        next or previous valid (non-stock) layer, preserving the selection.

        Args:
            surface: The WorkSurface instance containing the selection.
            direction: 1 for the next layer (down), -1 for the previous (up).
        """
        selected_wps = surface.get_selected_workpieces()
        if not selected_wps:
            return

        doc = self._editor.doc
        # A valid target for a WorkPiece is any layer that is NOT a StockLayer.
        workpiece_layers = [
            layer for layer in doc.layers if not isinstance(layer, StockLayer)
        ]

        if len(workpiece_layers) <= 1:
            # Not enough valid layers to move between.
            return

        # Assume all selected workpieces are on the same layer, which is a
        # reasonable constraint for this operation.
        current_layer = selected_wps[0].layer
        if not current_layer:
            return

        try:
            # Find the index of the current layer within the *filtered* list.
            current_index = workpiece_layers.index(current_layer)

            # Wrap around the filtered layer list.
            new_index = (
                current_index + direction + len(workpiece_layers)
            ) % len(workpiece_layers)
            new_layer = workpiece_layers[new_index]

            # 1. Create the model-only command.
            cmd = MoveWorkpiecesLayerCommand(
                selected_wps, new_layer, current_layer
            )

            # 2. Execute the command. The history manager updates the model,
            #    which triggers signals that cause the UI to destructively
            #    rebuild the moved elements in a new layer element.
            self._editor.history_manager.execute(cmd)

            # 3. After the model and UI have been updated, explicitly
            #    re-apply the selection to the newly created UI elements by
            #    telling the surface to select the same model objects again.
            surface.select_items(selected_wps)

        except ValueError:
            # This can happen if the current layer is not in the filtered list,
            # which would be an inconsistent state, but we should handle it.
            logger.warning(
                f"Layer '{current_layer.name}' not found in document's "
                "workpiece layer list."
            )
