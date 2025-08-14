import logging
from typing import List, Optional, TypeVar, Iterable
from blinker import Signal
from ..undo import HistoryManager
from .workpiece import WorkPiece
from .layer import Layer
from .item import DocItem


logger = logging.getLogger(__name__)

# For generic type hinting in add_child
T = TypeVar("T", bound="DocItem")


class Doc(DocItem):
    """
    Represents a loaded Rayforge document. Serves as the root of the
    document's object tree.
    """

    def __init__(self):
        super().__init__()
        self.history_manager = HistoryManager()
        self.active_layer_changed = Signal()
        self.job_assembly_invalidated = Signal()

        self._active_layer_index: int = 0

        # A new document starts with one empty layer.
        layer = Layer(_("Layer 1"))
        self.add_child(layer)

    @property
    def doc(self) -> "Doc":
        """The root Doc object is itself."""
        return self

    @property
    def layers(self) -> List[Layer]:
        """Returns a list of all child items that are Layers."""
        return [child for child in self.children if isinstance(child, Layer)]

    def __iter__(self):
        """Iterates through all workpieces in all layers."""
        return (wp for layer in self.layers for wp in layer.workpieces)

    @property
    def workpieces(self) -> List[WorkPiece]:
        """Returns a list of all workpieces from all layers."""
        return list(self)

    def get_all_workpieces(self) -> List["WorkPiece"]:
        """
        Recursively finds and returns a flattened list of all WorkPiece
        objects contained within this document.
        """
        all_wps = []
        for child in self.children:
            all_wps.extend(child.get_all_workpieces())
        return all_wps

    def add_workpiece(self, workpiece: WorkPiece):
        """Adds a workpiece to the currently active layer."""
        self.active_layer.add_workpiece(workpiece)

    def remove_workpiece(self, workpiece: WorkPiece):
        """Removes a workpiece from the layer that owns it."""
        if workpiece.parent and isinstance(workpiece.parent, Layer):
            workpiece.parent.remove_child(workpiece)

    @property
    def active_layer(self) -> Layer:
        """Returns the currently active layer."""
        return self.layers[self._active_layer_index]

    @active_layer.setter
    def active_layer(self, layer: Layer):
        """Sets the active layer by instance."""
        try:
            new_index = self.layers.index(layer)
            if self._active_layer_index != new_index:
                self._active_layer_index = new_index
                self.updated.send(self)
                self.active_layer_changed.send(self)
        except ValueError:
            logger.warning("Attempted to set a non-existent layer as active.")

    def _on_layer_post_transformer_changed(self, sender):
        """Special-case bubbling for a non-standard signal."""
        self.job_assembly_invalidated.send(self)

    def add_child(self, child: T, index: Optional[int] = None) -> T:
        if isinstance(child, Layer):
            child.post_step_transformer_changed.connect(
                self._on_layer_post_transformer_changed
            )
        super().add_child(child, index)
        return child

    def remove_child(self, child: DocItem):
        if isinstance(child, Layer):
            child.post_step_transformer_changed.disconnect(
                self._on_layer_post_transformer_changed
            )
        super().remove_child(child)

    def set_children(self, new_children: Iterable[DocItem]):
        old_layers = self.layers
        for layer in old_layers:
            layer.post_step_transformer_changed.disconnect(
                self._on_layer_post_transformer_changed
            )

        new_layers = [c for c in new_children if isinstance(c, Layer)]
        for layer in new_layers:
            layer.post_step_transformer_changed.connect(
                self._on_layer_post_transformer_changed
            )
        super().set_children(new_children)

    def add_layer(self, layer: Layer):
        self.add_child(layer)

    def remove_layer(self, layer: Layer):
        # Prevent removing the last layer.
        if layer not in self.layers or len(self.layers) <= 1:
            return
        self.remove_child(layer)

        # Ensure active_layer_index remains valid
        if self._active_layer_index >= len(self.layers):
            self._active_layer_index = len(self.layers) - 1
            self.active_layer_changed.send(self)

    def set_layers(self, layers: List[Layer]):
        # A document must always have at least one layer.
        if not layers:
            raise ValueError("Workpiece layer list cannot be empty.")

        # Preserve the active layer if it still exists in the new list
        current_active = self.active_layer
        old_active_index = self._active_layer_index
        try:
            new_active_index = layers.index(current_active)
        except ValueError:
            new_active_index = 0  # Default to first layer

        self.set_children(layers)

        self._active_layer_index = new_active_index
        if old_active_index != self._active_layer_index:
            self.active_layer_changed.send(self)

    def has_workpiece(self):
        return bool(self.workpieces)

    def has_result(self):
        # A result is possible if there's a workpiece and at least one
        # workflow (in any layer) has steps.
        return self.has_workpiece() and any(
            layer.workflow.has_steps() for layer in self.layers
        )
