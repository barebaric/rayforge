from __future__ import annotations
import base64
import logging
import pyvips
import uuid
from collections import OrderedDict
from blinker import Signal
from dataclasses import dataclass, field
from gettext import gettext as _
from pathlib import Path
from typing import ClassVar, Dict, Any, Optional, TYPE_CHECKING

from .asset import IAsset

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from ..image.base_renderer import Renderer


@dataclass
class SourceAsset(IAsset):
    """
    An immutable data record for a raw imported file and its base render.
    This is stored once per file in the document's central asset registry.
    """

    is_addable: ClassVar[bool] = False
    asset_type_name: ClassVar[str] = "source"
    display_icon_name: ClassVar[str] = "image-x-generic-symbolic"
    is_reorderable: ClassVar[bool] = False
    is_draggable_to_canvas: ClassVar[bool] = True
    type_display_name: ClassVar[str] = _("Source")
    can_edit: ClassVar[bool] = False
    add_action: ClassVar[Optional[str]] = None
    activate_action: ClassVar[Optional[str]] = None
    edit_item_action: ClassVar[Optional[str]] = None

    source_file: Path
    original_data: bytes = field(repr=False)
    renderer: "Renderer"
    base_render_data: Optional[bytes] = field(default=None, repr=False)
    thumbnail_data: Optional[bytes] = field(default=None, repr=False)
    metadata: Dict[str, Any] = field(default_factory=dict)
    width_px: Optional[int] = None
    height_px: Optional[int] = None
    width_mm: float = 0.0
    height_mm: float = 0.0
    _uid: str = field(init=False, default_factory=lambda: str(uuid.uuid4()))
    _name: str = field(init=False, repr=False)
    _hidden: bool = field(init=False, default=False)
    _updated: Signal = field(init=False, default_factory=Signal)
    extra: Dict[str, Any] = field(default_factory=dict)
    _base_image_cache: OrderedDict = field(
        init=False, default_factory=OrderedDict, repr=False
    )

    _BASE_IMAGE_CACHE_MAX_SIZE: ClassVar[int] = 10

    def __post_init__(self):
        self._name = self.source_file.name

    def get_cached_base_image(
        self, data_id: int, width: int, height: int
    ) -> Optional[pyvips.Image]:
        key = (data_id, width, height)
        return self._base_image_cache.get(key)

    def cache_base_image(
        self, data_id: int, width: int, height: int, image: pyvips.Image
    ) -> None:
        key = (data_id, width, height)
        if key in self._base_image_cache:
            self._base_image_cache.move_to_end(key)
        else:
            self._base_image_cache[key] = image
        while len(self._base_image_cache) > self._BASE_IMAGE_CACHE_MAX_SIZE:
            self._base_image_cache.popitem(last=False)

    def clear_base_image_cache(self) -> None:
        self._base_image_cache.clear()

    @property
    def uid(self) -> str:
        """The unique identifier of the asset instance."""
        return self._uid

    @property
    def updated(self) -> Signal:
        return self._updated

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
    def hidden(self) -> bool:
        """Indicates if this asset should be hidden from the UI."""
        return self._hidden

    @hidden.setter
    def hidden(self, value: bool):
        """Sets the hidden state."""
        self._hidden = value

    def get_thumbnail(self, size: int) -> Optional[bytes]:
        """Returns a PNG thumbnail of the rendered image."""
        try:
            if self.thumbnail_data:
                return self._scale_png(self.thumbnail_data, size)
            return None
        except Exception:
            logger.exception("Failed to generate source thumbnail")
            return None

    def _scale_png(self, png_data: bytes, size: int) -> Optional[bytes]:
        image = pyvips.Image.pngload_buffer(png_data)
        aspect = image.width / image.height
        if aspect > 1:
            new_width = size
            new_height = int(size / aspect)
        else:
            new_height = size
            new_width = int(size * aspect)
        scale = min(new_width / image.width, new_height / image.height)
        linear = image.colourspace("scrgb")
        resized = linear.resize(scale)
        image = resized.colourspace("srgb")
        return image.pngsave_buffer()

    def to_dict(self) -> Dict[str, Any]:
        """Serializes SourceAsset to a dictionary."""
        result = {
            "uid": self.uid,
            "type": self.asset_type_name,
            "name": self.name,
            "source_file": str(self.source_file),
            "original_data": base64.b64encode(self.original_data).decode(
                "utf-8"
            ),
            "base_render_data": (
                base64.b64encode(self.base_render_data).decode("utf-8")
                if self.base_render_data
                else None
            ),
            "thumbnail_data": (
                base64.b64encode(self.thumbnail_data).decode("utf-8")
                if self.thumbnail_data
                else None
            ),
            "renderer_name": self.renderer.__class__.__name__,
            "metadata": self.metadata,
            "width_px": self.width_px,
            "height_px": self.height_px,
            "width_mm": self.width_mm,
            "height_mm": self.height_mm,
            "hidden": self._hidden,
        }
        result.update(self.extra)
        return result

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SourceAsset":
        """Deserializes a dictionary into a SourceAsset instance."""
        from ..image import renderer_registry
        from ..image.base_renderer import UnknownRenderer

        known_keys = {
            "uid",
            "type",
            "name",
            "source_file",
            "original_data",
            "base_render_data",
            "thumbnail_data",
            "renderer_name",
            "metadata",
            "width_px",
            "height_px",
            "width_mm",
            "height_mm",
            "hidden",
        }
        extra = {k: v for k, v in data.items() if k not in known_keys}

        renderer = renderer_registry.get(data["renderer_name"])
        if renderer is None:
            logger.warning(
                f"Unknown renderer: {data['renderer_name']}. "
                f"Using UnknownRenderer as fallback."
            )
            renderer = UnknownRenderer()

        original_data = base64.b64decode(data["original_data"])
        base_render_data = (
            base64.b64decode(data["base_render_data"])
            if data.get("base_render_data")
            else None
        )
        thumbnail_data = (
            base64.b64decode(data["thumbnail_data"])
            if data.get("thumbnail_data")
            else None
        )

        instance = cls(
            source_file=Path(data["source_file"]),
            original_data=original_data,
            base_render_data=base_render_data,
            thumbnail_data=thumbnail_data,
            renderer=renderer,
            metadata=data.get("metadata", {}),
            width_px=data.get("width_px"),
            height_px=data.get("height_px"),
            width_mm=data.get("width_mm", 0.0),
            height_mm=data.get("height_mm", 0.0),
        )
        if "uid" in data:
            instance._uid = data["uid"]
        if "name" in data:
            instance.name = data["name"]
        if "hidden" in data:
            instance._hidden = data["hidden"]
        instance.extra = extra
        return instance
