from __future__ import annotations
import logging
import uuid
from typing import Dict, Any, Optional, TYPE_CHECKING, ClassVar
from gettext import gettext as _
from blinker import Signal
from ..context import get_context
from .geo import Geometry
from .asset import IAsset

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from .material import Material


class StockAsset(IAsset):
    """
    Represents a stock material definition in the asset library.

    This is not a DocItem and does not exist in the document hierarchy.
    It defines the properties of a stock that can be instanced as a StockItem.
    """

    is_addable: ClassVar[bool] = True
    asset_type_name: ClassVar[str] = "stock"
    display_icon_name: ClassVar[str] = "stock-symbolic"
    is_reorderable: ClassVar[bool] = True
    is_draggable_to_canvas: ClassVar[bool] = True
    type_display_name: ClassVar[str] = _("Stock Material")
    can_edit: ClassVar[bool] = True
    add_action: ClassVar[Optional[str]] = "add-stock"
    activate_action: ClassVar[Optional[str]] = None
    edit_item_action: ClassVar[Optional[str]] = "edit-stock-item"

    def __init__(
        self, name: str = "Stock", geometry: Optional[Geometry] = None
    ):
        self._uid: str = str(uuid.uuid4())
        self._name: str = name
        self.geometry: Geometry = (
            geometry if geometry is not None else Geometry()
        )
        self.thickness: Optional[float] = None
        self.material_uid: Optional[str] = None
        self._hidden: bool = False
        self._updated = Signal()
        self.extra: Dict[str, Any] = {}

    @property
    def uid(self) -> str:
        """The unique identifier of the asset instance."""
        return self._uid

    @uid.setter
    def uid(self, value: str) -> None:
        """Set the unique identifier. Used for deserialization."""
        self._uid = value

    @property
    def updated(self) -> Signal:
        """Signal emitted when the stock asset changes."""
        return self._updated

    @property
    def name(self) -> str:
        """The user-facing name of the asset."""
        return self._name

    @name.setter
    def name(self, value: str):
        """Sets the asset name and sends an update signal if changed."""
        if self._name != value:
            self._name = value
            self._updated.send(self)

    def to_dict(self) -> Dict[str, Any]:
        """Serializes the StockAsset to a dictionary."""
        result = {
            "uid": self.uid,
            "type": self.asset_type_name,
            "name": self.name,
            "geometry": self.geometry.to_dict(),
            "thickness": self.thickness,
            "material_uid": self.material_uid,
            "hidden": self._hidden,
        }
        result.update(self.extra)
        return result

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "StockAsset":
        """Deserializes a dictionary into a StockAsset instance."""
        known_keys = {
            "uid",
            "type",
            "name",
            "geometry",
            "thickness",
            "material_uid",
            "hidden",
        }
        extra = {k: v for k, v in data.items() if k not in known_keys}

        geometry = (
            Geometry.from_dict(data["geometry"])
            if "geometry" in data and data["geometry"]
            else None
        )
        asset = cls(name=data.get("name", "Stock"), geometry=geometry)
        asset.uid = data["uid"]
        asset.thickness = data.get("thickness")
        asset.material_uid = data.get("material_uid")
        asset._hidden = data.get("hidden", False)
        asset.extra = extra
        return asset

    def set_thickness(self, value: Optional[float]):
        """Setter method for use with undo commands."""
        if self.thickness != value:
            self.thickness = value
            self._updated.send(self)

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
            self._updated.send(self)

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

    @property
    def hidden(self) -> bool:
        """Indicates if this asset should be hidden from the UI."""
        return self._hidden

    @hidden.setter
    def hidden(self, value: bool):
        """Sets the hidden state and sends an update signal if changed."""
        if self._hidden != value:
            self._hidden = value
            self._updated.send(self)

    def set_hidden(self, value: bool):
        """Setter method for use with undo commands."""
        self.hidden = value

    def get_thumbnail(self, size: int) -> Optional[bytes]:
        """Returns a PNG thumbnail of the stock geometry."""
        try:
            return self.geometry.to_png(size)
        except Exception:
            logger.exception("Failed to generate stock thumbnail")
            return None
