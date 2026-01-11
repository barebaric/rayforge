from __future__ import annotations
import logging
from typing import TYPE_CHECKING, Optional, List
from ..core.layer import Layer
from ..core.undo import Command, ChangePropertyCommand
from ..core.undo.list_cmd import ReorderListCommand
from ..core.workpiece import WorkPiece

if TYPE_CHECKING:
    from ..ui_gtk.canvas2d.surface import WorkSurface
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


class AddLayerAndSetActiveCommand(Command):
    """
    An undoable command to add a new layer and set it as the active layer.
    """

    def __init__(
        self,
        editor: "DocEditor",
        new_layer: Optional[Layer] = None,
        name: str = "Add layer",
    ):
        super().__init__(name=name)
        self._editor = editor
        self.new_layer = new_layer or self._create_default_layer()
        self._old_active_layer: Optional[Layer] = None

    def _create_default_layer(self) -> Layer:
        """Creates a new layer with a default, unique name."""
        # Find a unique default name for the new layer
        base_name = _("Layer")
        existing_names = {layer.name for layer in self._editor.doc.layers}
        highest_num = 0
        for name in existing_names:
            if name.startswith(base_name):
                try:
                    num_part = name[len(base_name) :].strip()
                    if num_part.isdigit():
                        highest_num = max(highest_num, int(num_part))
                except ValueError:
                    continue  # Ignore names that don't parse correctly

        new_name = f"{base_name} {highest_num + 1}"
        return Layer(name=new_name)

    def execute(self):
        """Adds the layer and makes it active."""
        self._old_active_layer = self._editor.doc.active_layer
        new_list = self._editor.doc.layers + [self.new_layer]
        cmd = ReorderListCommand(
            target_obj=self._editor.doc,
            list_property_name="layers",
            new_list=new_list,
            setter_method_name="set_layers",
        )
        cmd.execute()
        self._editor.doc.active_layer = self.new_layer
        self._editor.doc.update_stock_visibility()

    def undo(self):
        """Removes the layer and restores the previous active layer."""
        new_list = [
            g for g in self._editor.doc.layers if g is not self.new_layer
        ]
        cmd = ReorderListCommand(
            target_obj=self._editor.doc,
            list_property_name="layers",
            new_list=new_list,
            setter_method_name="set_layers",
        )
        cmd.execute()
        if self._old_active_layer in self._editor.doc.layers:
            self._editor.doc.active_layer = self._old_active_layer
        self._editor.doc.update_stock_visibility()


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
        workpiece_layers = list(doc.layers)

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

    def add_layer_and_set_active(self, new_layer: Optional[Layer] = None):
        """Adds a new layer to the document and sets it as the active layer."""
        cmd = AddLayerAndSetActiveCommand(self._editor, new_layer)
        self._editor.history_manager.execute(cmd)

    def rename_layer(self, layer: Layer, new_name: str):
        """Renames a layer with an undoable command."""
        if new_name == layer.name:
            return
        cmd = ChangePropertyCommand(
            target=layer,
            property_name="name",
            new_value=new_name,
            setter_method_name="set_name",
            name=_("Rename layer"),
        )
        self._editor.history_manager.execute(cmd)

    def set_layer_visibility(self, layer: Layer, visible: bool):
        """Sets the visibility of a layer with an undoable command."""
        if visible == layer.visible:
            return
        cmd = ChangePropertyCommand(
            target=layer,
            property_name="visible",
            new_value=visible,
            setter_method_name="set_visible",
            name=_("Toggle layer visibility"),
        )
        self._editor.history_manager.execute(cmd)

    def set_layer_stock_item(
        self, layer: Layer, stock_item_uid: Optional[str]
    ):
        """Assigns a stock item to a layer with an undoable command."""
        if stock_item_uid == layer.stock_item_uid:
            return
        cmd = ChangePropertyCommand(
            target=layer,
            property_name="stock_item_uid",
            new_value=stock_item_uid,
            setter_method_name="set_stock_item_uid",
            name=_("Assign stock material"),
        )
        self._editor.history_manager.execute(cmd)
        # Update stock visibility when stock assignment changes
        if layer.active:
            self._editor.doc.update_stock_visibility()

    def set_active_layer(self, layer: Layer):
        """Sets the active layer."""
        if self._editor.doc.active_layer is layer:
            return
        old_layer = self._editor.doc.active_layer
        cmd = ChangePropertyCommand(
            target=self._editor.doc,
            property_name="active_layer",
            new_value=layer,
            old_value=old_layer,
            name=_("Set active layer"),
        )
        self._editor.history_manager.execute(cmd)

    def delete_layer(self, layer: Layer):
        """Deletes a layer with an undoable command."""
        new_list = [g for g in self._editor.doc.layers if g is not layer]
        cmd = ReorderListCommand(
            target_obj=self._editor.doc,
            list_property_name="layers",
            new_list=new_list,
            setter_method_name="set_layers",
            name=_("Remove layer '{name}'").format(name=layer.name),
        )
        self._editor.history_manager.execute(cmd)

    def reorder_layers(self, new_order: List[Layer]):
        """Reorders layers with an undoable command."""
        cmd = ReorderListCommand(
            target_obj=self._editor.doc,
            list_property_name="layers",
            new_list=new_order,
            setter_method_name="set_layers",
            name=_("Reorder layers"),
        )
        self._editor.history_manager.execute(cmd)
