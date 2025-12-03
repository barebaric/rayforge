from __future__ import annotations
from typing import Dict, Any, Optional, TYPE_CHECKING, Tuple, cast
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
        return {
            "uid": self.uid,
            "type": "stockitem",  # Discriminator for deserialization
            "name": self.name,
            "matrix": self.matrix.to_list(),
            "stock_asset_uid": self.stock_asset_uid,
            "visible": self.visible,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "StockItem":
        """
        Deserializes a dictionary into a StockItem instance.
        Assumes the new format with 'stock_asset_uid'. Legacy file handling
        is performed in Doc.from_dict.
        """
        new_item = cls(
            name=data.get("name", "Stock"),
            stock_asset_uid=data["stock_asset_uid"],
        )
        new_item.uid = data["uid"]
        new_item.matrix = Matrix.from_list(data["matrix"])
        new_item.visible = data.get("visible", True)

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
