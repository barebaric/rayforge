from __future__ import annotations
from typing import Dict, Any, Optional, TYPE_CHECKING

from .item import DocItem
from .geo import Geometry
from .matrix import Matrix

if TYPE_CHECKING:
    from .material import Material


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
        self.thickness: Optional[float] = None
        self.material_uid: Optional[str] = None
        self.visible: bool = True

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
            "thickness": self.thickness,
            "material_uid": self.material_uid,
            "visible": self.visible,
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
        new_item.thickness = data.get("thickness")
        new_item.material_uid = data.get("material_uid")
        new_item.visible = data.get("visible", True)

        return new_item

    def set_name(self, name: str):
        """Setter method for use with undo commands."""
        if self.name != name:
            self.name = name
            self.updated.send(self)

    def set_thickness(self, value: Optional[float]):
        """Setter method for use with undo commands."""
        if self.thickness != value:
            self.thickness = value
            self.updated.send(self)

    @property
    def material(self) -> Optional["Material"]:
        """
        Gets the Material object for this stock item.

        Returns:
            Material instance or None if not set or not found
        """
        if not self.material_uid:
            return None

        # Import here to avoid circular imports
        from ..config import material_mgr

        return material_mgr.get_material_or_none(self.material_uid)

    def set_material(self, material_uid: str):
        """
        Setter method for use with undo commands.

        Args:
            material_uid: The UID of the material to set
        """
        if self.material_uid != material_uid:
            self.material_uid = material_uid
            self.updated.send(self)

    def set_visible(self, visible: bool):
        """Sets the visibility of the stock item."""
        if self.visible == visible:
            return
        self.visible = visible
        self.updated.send(self)

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
