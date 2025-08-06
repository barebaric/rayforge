import logging
from typing import List
from blinker import Signal
from ..undo import HistoryManager
from .workpiece import WorkPiece
from .layer import Layer


logger = logging.getLogger(__name__)


class Doc:
    """
    Represents a loaded Rayforge document.
    """

    def __init__(self):
        self.history_manager = HistoryManager()
        self.changed = Signal()
        self.active_layer_changed = Signal()
        self.descendant_added = Signal()
        self.descendant_removed = Signal()
        self.descendant_updated = Signal()

        self.layers: List[Layer] = []
        self._layer_ref_for_pyreverse: Layer
        self._active_layer_index: int = 0

        # A new document starts with one empty layer. The application
        # controller (e.g., MainWindow) is responsible for populating it
        # with a default step.
        layer = Layer(self, _("Layer 1"))
        self.add_layer(layer)

    def __iter__(self):
        """Iterates through all workpieces in all layers."""
        return (wp for layer in self.layers for wp in layer.workpieces)

    @property
    def workpieces(self) -> List[WorkPiece]:
        """Returns a list of all workpieces from all layers."""
        return list(self)

    def add_workpiece(self, workpiece: WorkPiece):
        """Adds a workpiece to the currently active layer."""
        self.active_layer.add_workpiece(workpiece)

    def remove_workpiece(self, workpiece: WorkPiece):
        """Removes a workpiece from the layer that owns it."""
        if workpiece.layer and workpiece.layer in self.layers:
            workpiece.layer.remove_workpiece(workpiece)

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
                self.changed.send(self)
                self.active_layer_changed.send(self)
        except ValueError:
            logger.warning("Attempted to set a non-existent layer as active.")

    def _on_layer_changed(self, sender):
        """A single handler for generic changes in layers."""
        self.changed.send(self)

    def _on_descendant_added(self, sender, *, origin):
        self.descendant_added.send(self, origin=origin)

    def _on_descendant_removed(self, sender, *, origin):
        self.descendant_removed.send(self, origin=origin)

    def _on_descendant_updated(self, sender, *, origin):
        self.descendant_updated.send(self, origin=origin)

    def add_layer(self, layer: Layer):
        self.layers.append(layer)
        layer.changed.connect(self._on_layer_changed)
        layer.descendant_added.connect(self._on_descendant_added)
        layer.descendant_removed.connect(self._on_descendant_removed)
        layer.descendant_updated.connect(self._on_descendant_updated)
        self.descendant_added.send(self, origin=layer)
        self.changed.send(self)

    def remove_layer(self, layer: Layer):
        # Prevent removing the last layer.
        if layer not in self.layers or len(self.layers) <= 1:
            return
        layer.changed.disconnect(self._on_layer_changed)
        layer.descendant_added.disconnect(self._on_descendant_added)
        layer.descendant_removed.disconnect(self._on_descendant_removed)
        layer.descendant_updated.disconnect(self._on_descendant_updated)
        self.layers.remove(layer)
        self.descendant_removed.send(self, origin=layer)

        # Ensure active_layer_index remains valid
        if self._active_layer_index >= len(self.layers):
            self._active_layer_index = len(self.layers) - 1
            self.active_layer_changed.send(self)

        self.changed.send(self)

    def set_layers(self, layers: List[Layer]):
        # A document must always have at least one layer.
        if not layers:
            raise ValueError("Workpiece layer list cannot be empty.")

        old_layers = set(self.layers)
        new_layers = set(layers)

        # Preserve the active layer if it still exists in the new list
        current_active = self.active_layer
        old_active_index = self._active_layer_index
        try:
            new_active_index = layers.index(current_active)
        except ValueError:
            new_active_index = 0  # Default to first layer

        for layer in old_layers:
            layer.changed.disconnect(self._on_layer_changed)
            layer.descendant_added.disconnect(self._on_descendant_added)
            layer.descendant_removed.disconnect(self._on_descendant_removed)
            layer.descendant_updated.disconnect(self._on_descendant_updated)

        self.layers = list(layers)
        for layer in self.layers:
            layer.changed.connect(self._on_layer_changed)
            layer.descendant_added.connect(self._on_descendant_added)
            layer.descendant_removed.connect(self._on_descendant_removed)
            layer.descendant_updated.connect(self._on_descendant_updated)

        for layer in old_layers - new_layers:
            self.descendant_removed.send(self, origin=layer)
        for layer in new_layers - old_layers:
            self.descendant_added.send(self, origin=layer)

        self._active_layer_index = new_active_index
        self.changed.send(self)
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
