from __future__ import annotations
import uuid
from pathlib import Path
from typing import Optional, Dict, Any, TYPE_CHECKING
from dataclasses import dataclass, field

from .asset import IAsset

if TYPE_CHECKING:
    from ..image.base_renderer import Renderer


@dataclass
class SourceAsset(IAsset):
    """
    An immutable data record for a raw imported file and its base render.
    This is stored once per file in the document's central asset registry.
    """

    source_file: Path
    original_data: bytes
    renderer: "Renderer"
    base_render_data: Optional[bytes] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    width_px: Optional[int] = None
    height_px: Optional[int] = None
    width_mm: float = 0.0
    height_mm: float = 0.0
    uid: str = field(default_factory=lambda: str(uuid.uuid4()))
    _name: str = field(init=False, repr=False)

    def __post_init__(self):
        self._name = self.source_file.name

    # --- IAsset Protocol Implementation ---

    @property
    def name(self) -> str:
        """The user-facing name of the asset instance."""
        return self._name

    @name.setter
    def name(self, value: str) -> None:
        """Sets the asset name. Provided for protocol compatibility."""
        self._name = value

    @property
    def asset_type_name(self) -> str:
        """A unique, machine-readable name like "source" or "sketch"."""
        return "source"

    @property
    def display_icon_name(self) -> str:
        """The name of the icon representing the asset type."""
        return "image-x-generic-symbolic"

    @property
    def is_reorderable(self) -> bool:
        """Indicates if this asset type supports manual reordering."""
        return False

    @property
    def is_draggable_to_canvas(self) -> bool:
        """Indicates if this asset can be dragged onto the canvas."""
        return True

    @property
    def hidden(self) -> bool:
        """Indicates if this asset should be hidden from the UI."""
        return True

    def to_dict(self) -> Dict[str, Any]:
        """Serializes the SourceAsset to a dictionary."""
        return {
            "uid": self.uid,
            "type": self.asset_type_name,
            "name": self.name,
            "source_file": str(self.source_file),
            "original_data": self.original_data,
            "base_render_data": self.base_render_data,
            "renderer_name": self.renderer.__class__.__name__,
            "metadata": self.metadata,
            "width_px": self.width_px,
            "height_px": self.height_px,
            "width_mm": self.width_mm,
            "height_mm": self.height_mm,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SourceAsset":
        """Deserializes a dictionary into a SourceAsset instance."""
        from ..image import renderer_by_name

        renderer = renderer_by_name[data["renderer_name"]]

        instance = cls(
            uid=data["uid"],
            source_file=Path(data["source_file"]),
            original_data=data["original_data"],
            base_render_data=data.get("base_render_data"),
            renderer=renderer,
            metadata=data.get("metadata", {}),
            width_px=data.get("width_px"),
            height_px=data.get("height_px"),
            width_mm=data.get("width_mm", 0.0),
            height_mm=data.get("height_mm", 0.0),
        )
        if "name" in data:
            instance.name = data["name"]
        return instance
