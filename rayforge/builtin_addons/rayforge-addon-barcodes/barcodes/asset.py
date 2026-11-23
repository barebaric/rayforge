"""The barcode asset: a QR code or barcode generated from a template."""

import logging
import uuid
from gettext import gettext as _
from typing import Any, ClassVar

from blinker import Signal
from raygeo.barcode import (
    QrErrorCorrection,
    Symbology,
    min_quiet_zone,
)
from raygeo.barcode import (
    encode as encode_barcode,
)
from raygeo.barcode.geometry import BarcodeStyle, to_geometry
from raygeo.geo import Geometry
from raygeo.template import resolve_text as resolve_template

from rayforge.core.asset import IAsset
from rayforge.core.expression import referenced_template_names
from rayforge.core.geometry_provider import IGeometryProvider
from rayforge.core.template_functions import get_template_functions
from rayforge.image.base_renderer import Renderer
from rayforge.image.geo_renderer import render_geometry_fill_to_png
from rayforge.image.structures import FillRenderData, FillStyle

from .renderer import BARCODE_RENDERER

logger = logging.getLogger(__name__)

CACHE_KEY = 0
PLACEHOLDER_MM = 10.0
_BLACK = (0.0, 0.0, 0.0, 1.0)


def _symbology_from_value(value: Any) -> Symbology:
    try:
        return Symbology.from_value(str(value))
    except (TypeError, ValueError):
        return Symbology.QR


def _error_correction_from_value(value: Any) -> QrErrorCorrection:
    try:
        return QrErrorCorrection.from_value(str(value))
    except (TypeError, ValueError):
        return QrErrorCorrection.M


class BarcodeAsset(IAsset, IGeometryProvider):
    """
    A QR code or barcode whose content is a template.

    The content is resolved with the values the workpiece passes in (for
    example the document's job counter), encoded by raygeo and drawn as
    closed outlines with a solid fill.
    """

    is_addable: ClassVar[bool] = True
    asset_type_name: ClassVar[str] = "barcode"
    display_icon_name: ClassVar[str] = "barcode-symbolic"
    is_reorderable: ClassVar[bool] = False
    is_draggable_to_canvas: ClassVar[bool] = True
    type_display_name: ClassVar[str] = _("Barcode")
    can_edit: ClassVar[bool] = True
    add_action: ClassVar[str | None] = "new-barcode-asset"
    activate_action: ClassVar[str | None] = "activate-barcode"
    edit_item_action: ClassVar[str | None] = None

    def __init__(
        self,
        name: str | None = None,
        content: str = "Rayforge",
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
        return BARCODE_RENDERER

    @property
    def error(self) -> str | None:
        """Why the last content could not be encoded, if it could not."""
        return self._error

    @property
    def effective_quiet_zone(self) -> int:
        if self.quiet_zone is not None:
            return self.quiet_zone
        return min_quiet_zone(self.symbology)

    def variables_used(self) -> set[str]:
        return referenced_template_names([self.content])

    def get_reference_point(self) -> tuple[float, float]:
        return (0.0, 0.0)

    def get_geometry(
        self,
        params: dict[str, Any] | None = None,
        *,
        text_values: dict[str, Any] | None = None,
        resolved_text_cache: dict | None = None,
    ) -> tuple[Geometry, list[FillRenderData]]:
        cache = resolved_text_cache if resolved_text_cache is not None else {}
        context = dict(params or {})
        context.update(text_values or {})
        resolved, entry, _volatile = resolve_template(
            self.content,
            context,
            get_template_functions(),
            cache.get(CACHE_KEY),
        )
        cache[CACHE_KEY] = entry

        if not resolved:
            self._error = _("The content could not be resolved")
            return self._placeholder()

        try:
            encoded = encode_barcode(
                resolved, self.symbology, self.error_correction
            )
            geometry = to_geometry(
                encoded,
                BarcodeStyle(
                    module_mm=self.module_mm,
                    bar_height_mm=self.bar_height_mm,
                    quiet_zone=self.effective_quiet_zone,
                    show_text=self.show_text,
                    invert=self.invert,
                ),
            )
        except ValueError as exc:
            self._error = str(exc)
            logger.info(f"Barcode {self.uid[:8]}: {exc}")
            return self._placeholder()

        self._error = None
        fill = FillRenderData(
            geometry=geometry.copy(), style=FillStyle.SOLID, color=_BLACK
        )
        return geometry, [fill]

    def _placeholder(self) -> tuple[Geometry, list[FillRenderData]]:
        size = PLACEHOLDER_MM
        geometry = Geometry()
        geometry.move_to(0.0, 0.0)
        geometry.line_to(size, 0.0)
        geometry.line_to(size, size)
        geometry.line_to(0.0, size)
        geometry.close_path()
        return geometry, []

    def get_thumbnail(self, size: int) -> bytes | None:
        try:
            values = {name: 1 for name in self.variables_used()}
            geometry, _fills = self.get_geometry(text_values=values)
            return render_geometry_fill_to_png(geometry, size)
        except Exception:
            logger.exception("Failed to generate barcode thumbnail")
            return None

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
            content=data.get("content", "Rayforge"),
            symbology=_symbology_from_value(data.get("symbology", "qr")),
            error_correction=_error_correction_from_value(
                data.get("error_correction", "M")
            ),
            module_mm=float(data.get("module_mm", 0.5)),
            bar_height_mm=float(data.get("bar_height_mm", 15.0)),
            quiet_zone=(
                int(data["quiet_zone"])
                if data.get("quiet_zone") is not None
                else None
            ),
            show_text=bool(data.get("show_text", False)),
            invert=bool(data.get("invert", False)),
        )
        asset._uid = data.get("uid", asset._uid)
        return asset
