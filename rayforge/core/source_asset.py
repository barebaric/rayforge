from __future__ import annotations

import base64
import hashlib
import logging
import uuid
from collections import OrderedDict
from dataclasses import dataclass, field
from gettext import gettext as _
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar

import pyvips
from blinker import Signal

from .asset import IAsset

logger = logging.getLogger(__name__)

SOURCE_FILE_SHA256_KEY = "source_file_sha256"
SOURCE_FILE_SIZE_KEY = "source_file_size"

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
    add_action: ClassVar[str | None] = None
    activate_action: ClassVar[str | None] = None
    edit_item_action: ClassVar[str | None] = None

    source_file: Path
    original_data: bytes = field(repr=False)
    renderer: Renderer
    base_render_data: bytes | None = field(default=None, repr=False)
    thumbnail_data: bytes | None = field(default=None, repr=False)
    metadata: dict[str, Any] = field(default_factory=dict)
    width_px: int | None = None
    height_px: int | None = None
    width_mm: float = 0.0
    height_mm: float = 0.0
    auto_reload: bool = False
    _uid: str = field(init=False, default_factory=lambda: str(uuid.uuid4()))
    _name: str = field(init=False, repr=False)
    _hidden: bool = field(init=False, default=False)
    _updated: Signal = field(init=False, default_factory=Signal)
    extra: dict[str, Any] = field(default_factory=dict)
    _base_image_cache: OrderedDict = field(
        init=False, default_factory=OrderedDict, repr=False
    )

    _BASE_IMAGE_CACHE_MAX_SIZE: ClassVar[int] = 10

    def __post_init__(self):
        self._name = self.source_file.name

    def get_cached_base_image(
        self, data_id: int, width: int, height: int
    ) -> pyvips.Image | None:
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

    def set_source_file_fingerprint(self, data: bytes) -> None:
        """
        Records the SHA-256 and size of the file as it was on disk.

        Importers that convert a file before embedding it (for example EPS
        to PDF) call this with the original bytes, so the file on disk can
        still be recognised although ``original_data`` holds the
        converted data.
        """
        self.metadata[SOURCE_FILE_SHA256_KEY] = hashlib.sha256(
            data
        ).hexdigest()
        self.metadata[SOURCE_FILE_SIZE_KEY] = len(data)

    @property
    def source_file_sha256(self) -> str:
        """
        The SHA-256 (hex) of the imported file as it was on disk. Falls
        back to the embedded data, which is the file itself for every
        source that was not converted, and for older projects.
        """
        digest = self.metadata.get(SOURCE_FILE_SHA256_KEY)
        if isinstance(digest, str):
            return digest
        return hashlib.sha256(self.original_data).hexdigest()

    @property
    def source_file_size(self) -> int:
        """The size in bytes of the imported file as it was on disk."""
        size = self.metadata.get(SOURCE_FILE_SIZE_KEY)
        if isinstance(size, int):
            return size
        return len(self.original_data)

    def matches_source_file(self, data: bytes) -> bool:
        """True if the bytes are those of the file this asset came from."""
        if len(data) != self.source_file_size:
            return False
        return hashlib.sha256(data).hexdigest() == self.source_file_sha256

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

    def get_thumbnail(self, size: int) -> bytes | None:
        """Returns a PNG thumbnail of the rendered image."""
        try:
            if self.thumbnail_data:
                return self._scale_png(self.thumbnail_data, size)
            return None
        except Exception:
            logger.exception("Failed to generate source thumbnail")
            return None

    def _scale_png(self, png_data: bytes, size: int) -> bytes | None:
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

    def to_dict(self) -> dict[str, Any]:
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
            "auto_reload": self.auto_reload,
        }
        result.update(self.extra)
        return result

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SourceAsset:
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
            "auto_reload",
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
            auto_reload=data.get("auto_reload", False),
        )
        if "uid" in data:
            instance._uid = data["uid"]
        if "name" in data:
            instance.name = data["name"]
        if "hidden" in data:
            instance._hidden = data["hidden"]
        instance.extra = extra
        return instance
