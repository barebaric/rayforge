"""
The barcode asset: a document asset that provides the geometry of a QR
code or barcode generated from a template string.
"""

import logging
import uuid
from gettext import gettext as _
from typing import Any, ClassVar

from blinker import Signal
from raygeo.geo import Geometry

from rayforge.core.asset import IAsset
from rayforge.core.expression.expression_map import ExpressionMap
from rayforge.core.geometry_provider import IGeometryProvider
from rayforge.image.base_renderer import Renderer
from rayforge.image.geo_renderer import render_geometry_to_png
from rayforge.image.registry import renderer_registry
from rayforge.image.structures import FillRenderData, FillStyle

from .encoders import (
    MIN_QUIET_ZONE,
    QrErrorCorrection,
    Symbology,
    encode,
)
from .geometry import (
    BarcodeStyle,
    barcode_geometry,
    loops_to_geometry,
)

logger = logging.getLogger(__name__)

RENDERER_NAME = "SketchRenderer"
CACHE_KEY = 0
PLACEHOLDER_MM = 10.0
_BLACK = (0.0, 0.0, 0.0, 1.0)


class BarcodeAsset(IAsset, IGeometryProvider):
    """
    A QR code or barcode whose content is a template. The template is
    resolved with the parameters the workpiece passes in (for example
    the document's job counter), then encoded and drawn as outlines
    plus a solid fill.
    """

    is_addable: ClassVar[bool] = False
    asset_type_name: ClassVar[str] = "barcode"
    display_icon_name: ClassVar[str] = "barcode-symbolic"
    is_reorderable: ClassVar[bool] = True
    is_draggable_to_canvas: ClassVar[bool] = True
    type_display_name: ClassVar[str] = _("Barcode")
    can_edit: ClassVar[bool] = False
    add_action: ClassVar[str | None] = None
    activate_action: ClassVar[str | None] = None
    edit_item_action: ClassVar[str | None] = None

    def __init__(
        self,
        name: str | None = None,
        content: str = "",
        symbology: Symbology = Symbology.QR,
        error_correction: QrErrorCorrection = QrErrorCorrection.M,
        module_mm: float = 0.5,
        bar_height_mm: float = 15.0,
        quiet_zone: int | None = None,
        show_text: bool = False,
        invert: bool = False,
    ):
        self._uid = str(uuid.uuid4())
        self._name = name or _("Barcode")
        self._updated = Signal()
        self.content = content
        self.symbology = symbology
        self.error_correction = error_correction
        self.module_mm = module_mm
        self.bar_height_mm = bar_height_mm
        self.quiet_zone = quiet_zone
        self.show_text = show_text
        self.invert = invert
        self._error: str | None = None

    @property
    def uid(self) -> str:
        return self._uid

    @property
    def name(self) -> str:
        return self._name

    @name.setter
    def name(self, value: str) -> None:
        if value != self._name:
            self._name = value
            self._updated.send(self)

    @property
    def updated(self) -> Signal:
        return self._updated

    @property
    def hidden(self) -> bool:
        return False

    @property
    def provider_type_name(self) -> str:
        return self.asset_type_name

    @property
    def renderer(self) -> Renderer | None:
        return renderer_registry.get_by_name(RENDERER_NAME)

    @property
    def error(self) -> str | None:
        """Why the last content could not be encoded, if it could not."""
        return self._error

    @property
    def effective_quiet_zone(self) -> int:
        if self.quiet_zone is not None:
            return self.quiet_zone
        return MIN_QUIET_ZONE[self.symbology]

    def get_template_sources(self) -> list[str]:
        return [self.content]

    def resolve(self, params: dict[str, Any] | None = None) -> str:
        """The content with its ``{...}`` placeholders resolved."""
        return ExpressionMap(dict(params or {})).format(self.content)

    def _resolved(
        self, params: dict[str, Any] | None, cache: dict | None
    ) -> str:
        cached = cache.get(CACHE_KEY) if cache is not None else None
        if cached and cached[0] == self.content:
            return cached[1]
        text = self.resolve(params)
        if cache is not None:
            cache[CACHE_KEY] = (self.content, text)
        return text

    def _style(self) -> BarcodeStyle:
        return BarcodeStyle(
            module_mm=self.module_mm,
            bar_height_mm=self.bar_height_mm,
            quiet_zone=self.effective_quiet_zone,
            show_text=self.show_text,
            invert=self.invert,
        )

    def get_geometry(
        self,
        params: dict[str, Any] | None = None,
        *,
        resolved_text_cache: dict | None = None,
    ) -> tuple[Geometry, list[FillRenderData]]:
        text = self._resolved(params, resolved_text_cache)
        try:
            encoded = encode(self.symbology, text, self.error_correction)
        except ValueError as e:
            self._error = str(e)
            logger.info(f"Barcode {self.uid[:8]}: {e}")
            size = PLACEHOLDER_MM
            outline = [(0.0, 0.0), (size, 0.0), (size, size), (0.0, size)]
            return loops_to_geometry([outline]), []
        self._error = None
        geometry = barcode_geometry(encoded, self.symbology, self._style())
        fill = FillRenderData(
            geometry=geometry.copy(), style=FillStyle.SOLID, color=_BLACK
        )
        return geometry, [fill]

    def get_thumbnail(self, size: int) -> bytes | None:
        geometry, _fills = self.get_geometry()
        return render_geometry_to_png(geometry, size)

    def to_dict(self) -> dict[str, Any]:
        return {
            "uid": self._uid,
            "type": self.asset_type_name,
            "name": self._name,
            "content": self.content,
            "symbology": self.symbology.value,
            "error_correction": self.error_correction.value,
            "module_mm": self.module_mm,
            "bar_height_mm": self.bar_height_mm,
            "quiet_zone": self.quiet_zone,
            "show_text": self.show_text,
            "invert": self.invert,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "BarcodeAsset":
        asset = cls(
            name=data.get("name"),
            content=data.get("content", ""),
            symbology=Symbology(data.get("symbology", "qr")),
            error_correction=QrErrorCorrection(
                data.get("error_correction", "M")
            ),
            module_mm=float(data.get("module_mm", 0.5)),
            bar_height_mm=float(data.get("bar_height_mm", 15.0)),
            quiet_zone=data.get("quiet_zone"),
            show_text=bool(data.get("show_text", False)),
            invert=bool(data.get("invert", False)),
        )
        asset._uid = data.get("uid", asset._uid)
        return asset
