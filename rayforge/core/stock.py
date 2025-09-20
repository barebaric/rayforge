from __future__ import annotations
from typing import Dict, Any, Optional

from .item import DocItem
from .geo import Geometry
from .matrix import Matrix


class StockItem(DocItem):
    """
    Represents a piece of stock material in the document.

    It is a first-class document item with a shape defined by vector
    geometry. It can be transformed like any other DocItem.
    """

    def __init__(
        self, geometry: Optional[Geometry] = None, name: str = "Stock"
    ):
        super().__init__(name=name)
        self.geometry: Geometry = (
            geometry if geometry is not None else Geometry()
        )

        # If geometry is provided, set the initial matrix to match its
        # size.
        if not self.geometry.is_empty():
            min_x, min_y, max_x, max_y = self.geometry.rect()
            width = max_x - min_x
            height = max_y - min_y
            # Set the internal _matrix directly to avoid firing signals during
            # construction, which is good practice.
            self._matrix = Matrix.scale(width, height)

    def to_dict(self) -> Dict[str, Any]:
        """Serializes the StockItem to a dictionary."""
        return {
            "uid": self.uid,
            "type": "stockitem",  # Discriminator for deserialization
            "name": self.name,
            "matrix": self.matrix.to_list(),
            "geometry": self.geometry.to_dict(),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "StockItem":
        """Deserializes a dictionary into a StockItem instance."""
        geometry = (
            Geometry.from_dict(data["geometry"])
            if "geometry" in data and data["geometry"]
            else None
        )
        new_item = cls(name=data.get("name", "Stock"), geometry=geometry)
        new_item.uid = data["uid"]
        new_item.matrix = Matrix.from_list(data["matrix"])

        return new_item

    def get_natural_aspect_ratio(self) -> Optional[float]:
        """
        Returns the aspect ratio of the stock's geometry bounding box.
        """
        if self.geometry.is_empty():
            return None
        min_x, min_y, max_x, max_y = self.geometry.rect()
        width = max_x - min_x
        height = max_y - min_y
        return width / height if height > 1e-9 else None

    def get_current_aspect_ratio(self) -> Optional[float]:
        """
        Returns the aspect ratio of the stock's current world-space size.
        """
        w, h = self.size
        return w / h if h > 1e-9 else None

    def get_default_size(self, *args, **kwargs) -> tuple[float, float]:
        """
        Returns the natural size of the stock's geometry bounding box.
        Ignores container bounds as the stock's size is intrinsic.
        """
        if self.geometry.is_empty():
            return 1.0, 1.0  # Fallback for empty geometry
        min_x, min_y, max_x, max_y = self.geometry.rect()
        width = max_x - min_x
        height = max_y - min_y
        return width, height
