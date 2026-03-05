from __future__ import annotations
from typing import Dict, Any, Optional, TYPE_CHECKING, Tuple, cast
from gettext import gettext as _
from .geo import Geometry
from .item import DocItem
from .matrix import Matrix

if TYPE_CHECKING:
    from .asset import IAsset
    from .material import Material
    from .stock_asset import StockAsset


class StockItem(DocItem):
    """
    Represents an instance of a stock material in the document.

    It is a first-class document item that can be transformed. It references
    a StockAsset for its defining properties like geometry and material.
    """

    def __init__(self, stock_asset_uid: str, name: str = "Stock"):
        super().__init__(name=name)
        self.stock_asset_uid: str = stock_asset_uid
        self.visible: bool = True
        self.extra: Dict[str, Any] = {}

    def depends_on_asset(self, asset: "IAsset") -> bool:
        """Checks if this stock item is an instance of the given asset."""
        return self.stock_asset_uid == asset.uid

    @property
    def stock_asset(self) -> Optional["StockAsset"]:
        """Retrieves the StockAsset this item is an instance of."""
        doc = self.doc
        if doc:
            return cast(
                "Optional[StockAsset]",
                doc.get_asset_by_uid(self.stock_asset_uid),
            )
        return None

    @property
    def natural_size(self) -> Tuple[float, float]:
        """
        Returns the natural size of the stock item, defined by its
        referenced StockAsset's geometry bounding box.
        """
        asset = self.stock_asset
        if asset:
            return asset.get_natural_size()
        return (1.0, 1.0)  # Fallback

    def to_dict(self) -> Dict[str, Any]:
        """Serializes the StockItem to a dictionary."""
        result = {
            "uid": self.uid,
            "type": "stockitem",  # Discriminator for deserialization
            "name": self.name,
            "matrix": self.matrix.to_list(),
            "stock_asset_uid": self.stock_asset_uid,
            "visible": self.visible,
        }
        result.update(self.extra)
        return result

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "StockItem":
        """
        Deserializes a dictionary into a StockItem instance.
        Assumes the new format with 'stock_asset_uid'. Legacy file handling
        is performed in Doc.from_dict.
        """
        known_keys = {
            "uid",
            "type",
            "name",
            "matrix",
            "stock_asset_uid",
            "visible",
        }
        extra = {k: v for k, v in data.items() if k not in known_keys}

        new_item = cls(
            name=data.get("name", "Stock"),
            stock_asset_uid=data["stock_asset_uid"],
        )
        new_item.uid = data["uid"]
        new_item.matrix = Matrix.from_list(data["matrix"])
        new_item.visible = data.get("visible", True)
        new_item.extra = extra

        return new_item

    def duplicate(self) -> "StockItem":
        """
        Creates a deep copy of this StockItem with a new UID.

        This also creates a duplicate of the associated StockAsset,
        ensuring the new item is completely independent.
        """
        from .stock_asset import StockAsset

        old_asset = self.stock_asset
        if old_asset:
            asset_dict = old_asset.to_dict()
            new_asset = StockAsset.from_dict(asset_dict)
            new_asset.uid = str(__import__("uuid").uuid4())
            new_asset.name = _("{name} (copy)").format(name=old_asset.name)
        else:
            new_asset = None

        item_dict = self.to_dict()
        new_item = StockItem.from_dict(item_dict)
        new_item.uid = str(__import__("uuid").uuid4())
        new_item.name = _("{name} (copy)").format(name=self.name)

        if new_asset:
            new_item.stock_asset_uid = new_asset.uid
            if self.doc:
                self.doc.add_asset(new_asset, silent=True)

        return new_item

    def set_name(self, name: str):
        """Setter method for use with undo commands."""
        if self.name != name:
            self.name = name
            self.updated.send(self)

    # --- Delegated Properties for backward compatibility and convenience ---

    @property
    def thickness(self) -> Optional[float]:
        """Delegates thickness access to the StockAsset."""
        asset = self.stock_asset
        return asset.thickness if asset else None

    @property
    def material_uid(self) -> Optional[str]:
        """Delegates material_uid access to the StockAsset."""
        asset = self.stock_asset
        return asset.material_uid if asset else None

    @property
    def geometry(self) -> "Geometry":
        """Delegates geometry access to the StockAsset."""
        asset = self.stock_asset
        return asset.geometry if asset else Geometry()

    def get_world_geometry(self) -> "Geometry":
        """
        Returns the geometry transformed to world space.
        """
        geo = self.geometry
        if geo.is_empty():
            return geo
        world_geo = geo.copy()
        world_transform = self.get_world_transform()
        world_geo.transform(world_transform.to_4x4_numpy())
        return world_geo

    def get_world_rect_geometry(self) -> "Geometry":
        """
        Returns a rectangle geometry in world space based on the bbox.
        """
        x, y, w, h = self.bbox
        geo = Geometry()
        if w > 0 and h > 0:
            geo.move_to(x, y)
            geo.line_to(x + w, y)
            geo.line_to(x + w, y + h)
            geo.line_to(x, y + h)
            geo.close_path()
        return geo

    @property
    def display_icon_name(self) -> str:
        """Delegates display_icon_name access to the StockAsset."""
        asset = self.stock_asset
        return asset.display_icon_name if asset else "dialog-error-symbolic"

    def get_default_size(self, *args, **kwargs) -> tuple[float, float]:
        """Delegates size calculation to the StockAsset."""
        asset = self.stock_asset
        if asset:
            return asset.get_natural_size()
        return 1.0, 1.0

    @property
    def material(self) -> Optional["Material"]:
        """
        Gets the Material object for this stock item via its StockAsset.

        Returns:
            Material instance or None if not set or not found
        """
        asset = self.stock_asset
        return asset.material if asset else None

    def set_visible(self, visible: bool):
        """Sets the visibility of the stock item."""
        if self.visible == visible:
            return
        self.visible = visible
        self.updated.send(self)

    def get_natural_aspect_ratio(self) -> Optional[float]:
        """
        Returns the aspect ratio of the stock's geometry bounding box
        from its StockAsset.
        """
        asset = self.stock_asset
        if not asset or asset.geometry.is_empty():
            return None
        w, h = asset.get_natural_size()
        return w / h if h > 1e-9 else None

    def get_current_aspect_ratio(self) -> Optional[float]:
        """
        Returns the aspect ratio of the stock's current world-space size.
        """
        w, h = self.size
        return w / h if h > 1e-9 else None
