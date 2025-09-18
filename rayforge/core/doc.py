import logging
from typing import List, Optional, TypeVar, Iterable, Dict, TYPE_CHECKING
from blinker import Signal
from ..undo import HistoryManager
from .workpiece import WorkPiece
from .layer import Layer
from .item import DocItem
from .stocklayer import StockLayer
from .import_source import ImportSource

if TYPE_CHECKING:
    pass


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

        # A new document starts with a stock layer and one empty workpiece
        # layer. The stock layer is added first to appear at the bottom of
        # the UI list.
        stock_layer = StockLayer()
        self.add_child(stock_layer)

        workpiece_layer = Layer(_("Layer 1"))
        self.add_child(workpiece_layer)

        # The new workpiece layer should be active by default, not the stock.
        # The `layers` property will return [StockLayer, Layer], so index 1
        # is correct.
        self._active_layer_index: int = 1

    def to_dict(self) -> Dict:
        """Serializes the document and its children to a dictionary."""
        return {
            "uid": self.uid,
            "type": "doc",
            "active_layer_index": self._active_layer_index,
            "children": [child.to_dict() for child in self.children],
            "import_sources": {
                uid: source.to_dict()
                for uid, source in self.import_sources.items()
            },
        }

    @property
    def doc(self) -> "Doc":
        """The root Doc object is itself."""
        return self

    @property
    def stock_layer(self) -> Optional["StockLayer"]:
        """Returns the StockLayer if one exists in the document."""
        for child in self.children:
            if isinstance(child, StockLayer):
                return child
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
            # This correctly skips StockLayer, as its all_workpieces is empty
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
        if isinstance(child, StockLayer):
            if self.stock_layer is not None:
                raise ValueError("A document can only have one StockLayer.")
            # StockLayer doesn't have a workflow, so no signal to connect
        elif isinstance(child, Layer):
            child.post_step_transformer_changed.connect(
                self._on_layer_post_transformer_changed
            )
        super().add_child(child, index)
        return child

    def remove_child(self, child: DocItem):
        if isinstance(child, StockLayer):
            logger.warning("The StockLayer cannot be removed.")
            return

        if isinstance(child, Layer):
            # StockLayer is a subclass, but we already handled it above.
            if child.workflow:
                child.post_step_transformer_changed.disconnect(
                    self._on_layer_post_transformer_changed
                )
        super().remove_child(child)

    def set_children(self, new_children: Iterable[DocItem]):
        new_children_list = list(new_children)
        stock_layer_count = sum(
            isinstance(c, StockLayer) for c in new_children_list
        )
        if stock_layer_count > 1:
            raise ValueError("A document can only have one StockLayer.")

        old_layers = self.layers
        for layer in old_layers:
            if not isinstance(layer, StockLayer):
                # Ensure the layer has a workflow before disconnecting
                if layer.workflow:
                    layer.post_step_transformer_changed.disconnect(
                        self._on_layer_post_transformer_changed
                    )

        new_layers = [c for c in new_children_list if isinstance(c, Layer)]
        for layer in new_layers:
            if not isinstance(layer, StockLayer):
                layer.post_step_transformer_changed.connect(
                    self._on_layer_post_transformer_changed
                )
        super().set_children(new_children_list)

    def add_layer(self, layer: Layer):
        self.add_child(layer)

    def remove_layer(self, layer: Layer):
        if layer not in self.layers:
            return

        new_layers = [la for la in self.layers if la is not layer]
        try:
            self.set_layers(new_layers)
        except ValueError as e:
            # Log the failure if an internal API call tries to do this.
            logger.warning(f"Layer removal failed: {e}")

    def set_layers(self, layers: List[Layer]):
        new_layers_list = list(layers)

        # A document must always have at least one regular workpiece layer.
        regular_layer_count = sum(
            1 for layer in new_layers_list if not isinstance(layer, StockLayer)
        )
        if regular_layer_count < 1:
            raise ValueError(
                "A document must have at least one workpiece layer."
            )

        # Preserve the active layer if it still exists in the new list
        old_active_layer = self.active_layer
        try:
            new_active_index = new_layers_list.index(old_active_layer)
        except ValueError:
            # The old active layer is not in the new list, so pick a default.
            # Fall back to the first regular layer in the new list.
            try:
                first_regular_layer = next(
                    la
                    for la in new_layers_list
                    if not isinstance(la, StockLayer)
                )
                new_active_index = new_layers_list.index(first_regular_layer)
            except (StopIteration, ValueError):
                # This should be unreachable due to the check above, but as a
                # failsafe, just pick the first item.
                new_active_index = 0

        # IMPORTANT: Update the active index BEFORE calling set_children.
        # set_children fires signals that can cause UI updates, which will
        # query `active_layer`. If the index is not updated first, it can
        # be out of bounds for the new, shorter list of layers, causing an
        # IndexError.
        self._active_layer_index = new_active_index

        self.set_children(new_layers_list)

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
