import logging
from typing import List, Optional, TypeVar, Iterable, Dict, TYPE_CHECKING
from blinker import Signal
from ..undo import HistoryManager
from .workpiece import WorkPiece
from .layer import Layer
from .item import DocItem
from .import_source import ImportSource

if TYPE_CHECKING:
    from .stock import StockItem


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
        self.import_sources: Dict[str, ImportSource] = {}

        # A new document starts with one empty workpiece layer
        workpiece_layer = Layer(_("Layer 1"))
        self.add_child(workpiece_layer)

        # The new workpiece layer should be active by default
        self._active_layer_index: int = 0

    @classmethod
    def from_dict(cls, data: Dict) -> "Doc":
        """Deserializes the document from a dictionary."""
        doc = cls()
        doc.uid = data.get("uid", doc.uid)

        # Clear the default layer created by __init__
        doc.set_children([])
        # Reset the index to a safe value before adding new layers
        doc._active_layer_index = -1

        layers = [Layer.from_dict(d) for d in data.get("children", [])]
        stock_items = [
            StockItem.from_dict(d) for d in data.get("stock_items", [])
        ]
        doc.set_children(layers + stock_items)

        doc._active_layer_index = data.get("active_layer_index", 0)

        import_sources = {
            uid: ImportSource.from_dict(src_data)
            for uid, src_data in data.get("import_sources", {}).items()
        }
        doc.import_sources = import_sources

        return doc

    @property
    def stock_items(self) -> List["StockItem"]:
        """Returns a list of all child items that are StockItems."""
        from .stock import (
            StockItem,
        )  # Lazy import to avoid circular dependency

        return [
            child for child in self.children if isinstance(child, StockItem)
        ]

    @stock_items.setter
    def stock_items(self, new_stock_items: List["StockItem"]):
        """
        Replaces the existing stock items with a new list, preserving order,
        while leaving other child types (like Layers) untouched.
        """
        from .stock import StockItem  # Lazy import

        # Create a new children list containing only the non-stock items
        other_children = [
            child
            for child in self.children
            if not isinstance(child, StockItem)
        ]

        # Combine the non-stock items with the new list of stock items
        new_children_list = other_children + new_stock_items

        # Use set_children to correctly handle signal connection/disconnection
        self.set_children(new_children_list)
        self.updated.send(self)

    def get_import_source_by_uid(self, uid: str) -> Optional[ImportSource]:
        """
        Retrieves an ImportSource from the document's registry by its UID.
        """
        return self.import_sources.get(uid)

    def to_dict(self) -> Dict:
        """Serializes the document and its children to a dictionary."""
        return {
            "uid": self.uid,
            "type": "doc",
            "active_layer_index": self._active_layer_index,
            "children": [child.to_dict() for child in self.layers],
            "stock_items": [
                stock_item.to_dict() for stock_item in self.stock_items
            ],
            "import_sources": {
                uid: source.to_dict()
                for uid, source in self.import_sources.items()
            },
        }

    def add_import_source(self, source: ImportSource):
        """Adds or updates an ImportSource in the document's registry."""
        if not isinstance(source, ImportSource):
            raise TypeError("Only ImportSource objects can be added.")
        self.import_sources[source.uid] = source

    @property
    def doc(self) -> "Doc":
        """The root Doc object is itself."""
        return self

    def add_stock_item(self, stock_item: "StockItem"):
        """Adds a stock item to the document."""
        self.add_child(stock_item)
        self.updated.send(self)

    def remove_stock_item(self, stock_item: "StockItem"):
        """Removes a stock item from the document."""
        self.remove_child(stock_item)
        self.updated.send(self)

    def get_stock_item_by_uid(self, uid: str) -> Optional["StockItem"]:
        """Retrieves a stock item by its UID."""
        for stock_item in self.stock_items:
            if stock_item.uid == uid:
                return stock_item
        return None

    @property
    def layers(self) -> List[Layer]:
        """Returns a list of all child items that are Layers."""
        return [child for child in self.children if isinstance(child, Layer)]

    @property
    def all_workpieces(self) -> List[WorkPiece]:
        """
        Recursively finds and returns a flattened list of all WorkPiece
        objects contained within this document.
        """
        wps = []
        for layer in self.layers:
            wps.extend(layer.all_workpieces)
        return wps

    def add_workpiece(self, workpiece: WorkPiece):
        """Adds a workpiece to the currently active layer."""
        self.active_layer.add_workpiece(workpiece)

    def remove_workpiece(self, workpiece: WorkPiece):
        """Removes a workpiece from the layer that owns it."""
        if workpiece.parent:
            workpiece.parent.remove_child(workpiece)

    def get_top_level_items(self) -> List["DocItem"]:
        """
        Returns a list of all top-level, user-facing items in the document by
        querying each layer for its content.
        """
        top_items = []
        for layer in self.layers:
            top_items.extend(layer.get_content_items())
        return top_items

    @property
    def active_layer(self) -> Layer:
        """Returns the currently active layer."""
        if not self.layers:
            raise IndexError("Document has no layers.")
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
                self.update_stock_visibility()
        except ValueError:
            logger.warning("Attempted to set a non-existent layer as active.")

    def _on_layer_per_step_transformer_changed(self, sender):
        """Special-case bubbling for a non-standard signal."""
        self.job_assembly_invalidated.send(self)

    def add_child(self, child: T, index: Optional[int] = None) -> T:
        if isinstance(child, Layer):
            child.per_step_transformer_changed.connect(
                self._on_layer_per_step_transformer_changed
            )
        super().add_child(child, index)
        return child

    def remove_child(self, child: DocItem):
        if isinstance(child, Layer):
            if child.workflow:
                child.per_step_transformer_changed.disconnect(
                    self._on_layer_per_step_transformer_changed
                )
        super().remove_child(child)

    def set_children(self, new_children: Iterable[DocItem]):
        new_children_list = list(new_children)

        old_layers = self.layers
        for layer in old_layers:
            # Ensure the layer has a workflow before disconnecting
            if layer.workflow:
                layer.per_step_transformer_changed.disconnect(
                    self._on_layer_per_step_transformer_changed
                )

        new_layers = [c for c in new_children_list if isinstance(c, Layer)]
        for layer in new_layers:
            layer.per_step_transformer_changed.connect(
                self._on_layer_per_step_transformer_changed
            )
        super().set_children(new_children_list)

    def add_layer(self, layer: Layer):
        self.add_child(layer)

    def remove_layer(self, layer: Layer):
        if layer not in self.layers:
            return

        if len(self.layers) <= 1:
            msg = "A document must have at least one workpiece layer."
            logger.warning(msg)
            return

        # Safely adjust active layer index before removal
        old_active_layer = self.active_layer
        layers_before_remove = self.layers
        layer_index_to_remove = layers_before_remove.index(layer)

        # Remove the child. This will trigger signals.
        self.remove_child(layer)

        # After removal, the list of layers is shorter. We need to ensure
        # _active_layer_index is still valid.
        if old_active_layer is layer:
            # The active layer was deleted. Choose the one before it, or 0.
            new_index = max(0, layer_index_to_remove - 1)
            self._active_layer_index = new_index
            self.active_layer_changed.send(self)
        elif layer_index_to_remove < self._active_layer_index:
            # A layer before the active one was removed, so the active index
            # must shift.
            self._active_layer_index -= 1
            # The active layer instance hasn't changed, so no change signal
            # needed.

    def set_layers(self, layers: List[Layer]):
        new_layers_list = list(layers)

        # A document must always have at least one workpiece layer.
        if len(new_layers_list) < 1:
            raise ValueError(
                "A document must have at least one workpiece layer."
            )

        # Preserve the active layer if it still exists in the new list
        old_active_layer = None
        if self.layers and self._active_layer_index >= 0:
            old_active_layer = self.active_layer

        try:
            if old_active_layer:
                new_active_index = new_layers_list.index(old_active_layer)
            else:
                new_active_index = 0
        except ValueError:
            # The old active layer is not in the new list, so pick a default.
            new_active_index = 0

        # IMPORTANT: Update the active index BEFORE calling set_children.
        self._active_layer_index = new_active_index

        # CRITICAL CHANGE: Preserve non-layer children (e.g., stock items)
        current_stock_items = self.stock_items
        new_children_list = new_layers_list + current_stock_items
        self.set_children(new_children_list)

        # After the state is consistent, send the active_layer_changed signal
        # if the active layer instance has actually changed.
        if old_active_layer is not self.active_layer:
            self.active_layer_changed.send(self)

    def has_workpiece(self):
        return bool(self.all_workpieces)

    def has_result(self):
        # A result is possible if there's a workpiece and at least one
        # workflow (in any layer) has steps.
        return self.has_workpiece() and any(
            layer.workflow and layer.workflow.has_steps()
            for layer in self.layers
        )

    def update_stock_visibility(self):
        """
        Updates stock item visibility based on the active layer.
        Only the stock item assigned to the active layer will be visible.
        """
        active_layer = self.active_layer
        active_stock_uid = (
            active_layer.stock_item_uid if active_layer else None
        )

        for stock_item in self.stock_items:
            stock_item.set_visible(stock_item.uid == active_stock_uid)
