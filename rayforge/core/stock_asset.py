from __future__ import annotations
import uuid
from typing import Dict, Any, Optional, TYPE_CHECKING
from blinker import Signal

from ..context import get_context
from .geo import Geometry
from .asset import IAsset

if TYPE_CHECKING:
    from .material import Material


class StockAsset(IAsset):
    """
    Represents a stock material definition in the asset library.

    This is not a DocItem and does not exist in the document hierarchy.
    It defines the properties of a stock that can be instanced as a StockItem.
    """

    def __init__(
        self, name: str = "Stock", geometry: Optional[Geometry] = None
    ):
        self.uid: str = str(uuid.uuid4())
        self._name: str = name
        self.geometry: Geometry = (
            geometry if geometry is not None else Geometry()
        )
        self.thickness: Optional[float] = None
        self.material_uid: Optional[str] = None
        self.updated = Signal()

    @property
    def name(self) -> str:
        """The user-facing name of the asset."""
        return self._name

    @name.setter
    def name(self, value: str):
        """Sets the asset name and sends an update signal if changed."""
        if self._name != value:
            self._name = value
            self.updated.send(self)

    @property
    def asset_type_name(self) -> str:
        """The machine-readable type name for the asset list."""
        return "stock"

    @property
    def display_icon_name(self) -> str:
        """The icon name for the asset list."""
        return "stock-symbolic"

    @property
    def is_reorderable(self) -> bool:
        """Whether this asset type supports reordering in the asset list."""
        return True

    @property
    def is_draggable_to_canvas(self) -> bool:
        """Whether this asset can be dragged from the list onto the canvas."""
        return False

    def to_dict(self) -> Dict[str, Any]:
        """Serializes the StockAsset to a dictionary."""
        return {
            "uid": self.uid,
            "type": self.asset_type_name,  # For polymorphic deserialization
            "name": self.name,
            "geometry": self.geometry.to_dict(),
            "thickness": self.thickness,
            "material_uid": self.material_uid,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "StockAsset":
        """Deserializes a dictionary into a StockAsset instance."""
        geometry = (
            Geometry.from_dict(data["geometry"])
            if "geometry" in data and data["geometry"]
            else None
        )
        asset = cls(name=data.get("name", "Stock"), geometry=geometry)
        asset.uid = data["uid"]
        asset.thickness = data.get("thickness")
        asset.material_uid = data.get("material_uid")
        return asset

    def set_thickness(self, value: Optional[float]):
        """Setter method for use with undo commands."""
        if self.thickness != value:
            self.thickness = value
            self.updated.send(self)

    @property
    def material(self) -> Optional["Material"]:
        """
        Gets the Material object for this stock asset.

        Returns:
            Material instance or None if not set or not found
        """
        if not self.material_uid:
            return None

        context = get_context()
        material_mgr = context.material_mgr
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

    def get_natural_size(self) -> tuple[float, float]:
        """
        Returns the natural size of the stock's geometry bounding box.
        """
        if self.geometry.is_empty():
            return 1.0, 1.0  # Fallback for empty geometry
        min_x, min_y, max_x, max_y = self.geometry.rect()
        width = max_x - min_x
        height = max_y - min_y
        return width, height
