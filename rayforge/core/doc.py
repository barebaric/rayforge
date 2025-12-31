import logging
from typing import List, Optional, TypeVar, Iterable, Dict, TYPE_CHECKING, cast
from blinker import Signal
from ..core.undo import HistoryManager
from .asset import IAsset
from .item import DocItem
from .layer import Layer
from .source_asset import SourceAsset
from .workpiece import WorkPiece

if TYPE_CHECKING:
    from .sketcher.sketch import Sketch
    from .stock import StockItem
    from .stock_asset import StockAsset


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

        # Asset Management
        self.assets: Dict[str, IAsset] = {}
        self.asset_order: List[str] = []

        # A new document starts with one empty workpiece layer
        workpiece_layer = Layer(_("Layer 1"))
        self.add_child(workpiece_layer)

        # The new workpiece layer should be active by default
        self._active_layer_index: int = 0

    @classmethod
    def from_dict(cls, data: Dict) -> "Doc":
        """Deserializes the document from a dictionary."""
        from .stock import StockItem
        from .stock_asset import StockAsset
        from .sketcher.sketch import Sketch
        from .geo import Geometry
        from .matrix import Matrix
        from .source_asset import SourceAsset

        # --- Polymorphic Deserialization Factories ---
        asset_class_map = {
            "stock": StockAsset,
            "sketch": Sketch,
            "source": SourceAsset,
        }
        item_class_map = {"layer": Layer, "stockitem": StockItem}

        def _deserialize_asset(asset_data: Dict) -> IAsset:
            asset_type = asset_data.get("type")
            asset_class = None
            if asset_type:
                asset_class = asset_class_map.get(asset_type)
            if not asset_class:
                raise TypeError(f"Unknown asset type '{asset_type}'")
            return asset_class.from_dict(asset_data)

        def _deserialize_item(item_data: Dict) -> DocItem:
            item_type = item_data.get("type")
            item_class = None
            if item_type:
                item_class = item_class_map.get(item_type)
            if not item_class:
                raise TypeError(f"Unknown document item type '{item_type}'")
            return item_class.from_dict(item_data)

        doc = cls()
        doc.uid = data.get("uid", doc.uid)

        # Clear the default layer created by __init__
        doc.set_children([])
        doc._active_layer_index = -1

        # Load assets first from unified list
        for asset_data in data.get("assets", []):
            doc.add_asset(_deserialize_asset(asset_data))

        # Legacy Asset Loading (from separate dictionaries)
        stock_assets_data = data.get("stock_assets", {})
        for uid, sa_data in stock_assets_data.items():
            doc.add_asset(StockAsset.from_dict(sa_data))
        sketches_data = data.get("sketches", {})
        for uid, s_data in sketches_data.items():
            doc.add_asset(Sketch.from_dict(s_data))
        source_assets_data = data.get("source_assets", {})
        for uid, src_data in source_assets_data.items():
            doc.add_asset(SourceAsset.from_dict(src_data))

        # Load children (Layers and StockItems) from unified list
        children = []
        children_data = data.get("children", [])
        for d in children_data:
            children.append(_deserialize_item(d))

        # Legacy stock item loading (from separate list)
        stock_items_data = data.get("stock_items", [])
        for d in stock_items_data:
            if "geometry" in d and "stock_asset_uid" not in d:
                # Legacy format: create a StockAsset from item data
                asset = StockAsset(name=d.get("name", "Stock"))
                asset.geometry = (
                    Geometry.from_dict(d["geometry"])
                    if "geometry" in d and d["geometry"]
                    else Geometry()
                )
                asset.thickness = d.get("thickness")
                asset.material_uid = d.get("material_uid")
                doc.add_asset(asset)

                # Create a StockItem instance linked to the new asset
                item = StockItem(
                    stock_asset_uid=asset.uid, name=d.get("name", "Stock")
                )
                item.uid = d["uid"]
                item.matrix = Matrix.from_list(d["matrix"])
                item.visible = d.get("visible", True)
                children.append(item)
            else:
                children.append(StockItem.from_dict(d))

        doc.set_children(children)
        doc._active_layer_index = data.get("active_layer_index", 0)

        return doc

    @property
    def stock_items(self) -> List["StockItem"]:
        """Returns a list of all child items that are StockItems."""
        from .stock import StockItem

        return [
            child for child in self.children if isinstance(child, StockItem)
        ]

    def get_source_asset_by_uid(self, uid: str) -> Optional[SourceAsset]:
        """
        Retrieves a SourceAsset from the document's registry by its UID.
        """
        return self.source_assets.get(uid)

    def to_dict(self) -> Dict:
        """Serializes the document and its children to a dictionary."""
        return {
            "uid": self.uid,
            "type": "doc",
            "active_layer_index": self._active_layer_index,
            "children": [child.to_dict() for child in self.children],
            "assets": [asset.to_dict() for asset in self.get_all_assets()],
        }

    def add_asset(
        self, asset: IAsset, index: Optional[int] = None, silent: bool = False
    ):
        """
        Adds or updates an asset in the document's unified registry and
        maintains its order.
        """
        if not isinstance(asset, IAsset):
            raise TypeError("Only IAsset objects can be added.")
        if asset.uid in self.assets:
            return  # Asset already exists

        self.assets[asset.uid] = asset
        if index is None:
            self.asset_order.append(asset.uid)
        else:
            self.asset_order.insert(index, asset.uid)

        if not silent:
            self.updated.send(self)

    def remove_asset_by_uid(self, uid: str):
        """Removes an asset from the document by its UID."""
        if self.assets.pop(uid, None):
            try:
                self.asset_order.remove(uid)
            except ValueError:
                logger.warning(
                    f"Asset UID {uid} was in asset dict but not in order list."
                )
            self.updated.send(self)

    def remove_asset(self, asset: "IAsset"):
        """Removes an asset from the document."""
        self.remove_asset_by_uid(asset.uid)

    def get_asset_by_uid(self, uid: str) -> Optional[IAsset]:
        """Retrieves any asset from the document's registry by its UID."""
        return self.assets.get(uid)

    def set_asset_order(self, new_order_uids: List[str]):
        """Sets the canonical order for all assets."""
        if set(new_order_uids) != set(self.assets.keys()):
            raise ValueError(
                "New order list must contain all and only existing asset UIDs."
            )
        self.asset_order = new_order_uids
        self.updated.send(self)

    def get_all_assets(self) -> List[IAsset]:
        """Returns a unified list of all assets in the canonical order."""
        return [
            self.assets[uid] for uid in self.asset_order if uid in self.assets
        ]

    @property
    def source_assets(self) -> Dict[str, "SourceAsset"]:
        """
        Returns a dictionary of all SourceAssets for compatibility.
        NOTE: The order of this dictionary is not guaranteed.
        """
        return {
            uid: cast(SourceAsset, asset)
            for uid, asset in self.assets.items()
            if asset.asset_type_name == "source"
        }

    @property
    def stock_assets(self) -> Dict[str, "StockAsset"]:
        """
        Returns a dictionary of all StockAssets for compatibility.
        NOTE: The order of this dictionary is not guaranteed.
        """
        from .stock_asset import StockAsset

        return {
            uid: cast(StockAsset, asset)
            for uid, asset in self.assets.items()
            if asset.asset_type_name == "stock"
        }

    @property
    def sketches(self) -> Dict[str, "Sketch"]:
        """
        Returns a dictionary of all Sketches for compatibility.
        NOTE: The order of this dictionary is not guaranteed.
        """
        from .sketcher.sketch import Sketch

        return {
            uid: cast(Sketch, asset)
            for uid, asset in self.assets.items()
            if asset.asset_type_name == "sketch"
        }

    @property
    def doc(self) -> "Doc":
        """The root Doc object is itself."""
        return self

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

        self._active_layer_index = new_active_index
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
        # workflow (in any layer) has at least one visible step.
        return self.has_workpiece() and any(
            step.visible
            for layer in self.layers
            if layer.workflow
            for step in layer.workflow.steps
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
