import logging
from gettext import gettext as _
from pathlib import Path
from typing import ClassVar

from raygeo.geo import Geometry

from ...core.source_asset import SourceAsset
from ..base_importer import ImporterFeature
from ..dxf.importer import DxfImporter
from ..engine import NormalizationEngine
from ..structures import (
    ImportManifest,
    LayerGeometry,
    LayerInfo,
    ParsingResult,
)
from .parser import HpglDrawing, parse_hpgl
from .renderer import HPGL_RENDERER

logger = logging.getLogger(__name__)


def _layer_id(pen: int) -> str:
    return f"pen-{pen}"


class HpglImporter(DxfImporter):
    """
    Imports HP-GL plot files (.plt, .hpgl). Each pen becomes a layer that
    can be selected in the import dialog. Layer filtering and splitting
    are inherited from the DXF importer, which works on the same
    per-layer, Y-up, millimetre geometry.
    """

    label = "HPGL plot files"
    mime_types = ("application/vnd.hp-hpgl", "application/vnd.hp-HPGL")
    extensions = (".plt", ".hpgl", ".hpg", ".hgl")
    features: ClassVar[set[ImporterFeature]] = {
        ImporterFeature.DIRECT_VECTOR,
        ImporterFeature.LAYER_SELECTION,
    }

    def __init__(self, data: bytes, source_file: Path | None = None):
        super().__init__(data, source_file)
        self._drawing: HpglDrawing | None = None

    def _get_drawing(self) -> HpglDrawing:
        if self._drawing is None:
            self._drawing = parse_hpgl(self.raw_data)
            if self._drawing.unsupported:
                names = ", ".join(sorted(self._drawing.unsupported))
                self.add_warning(
                    _(
                        "Some HPGL instructions are not supported and "
                        "were skipped: {names}"
                    ).format(names=names)
                )
            if not self._drawing.geometries_by_pen:
                self.add_error(
                    _("{filename} contains no HPGL drawing commands.").format(
                        filename=self.source_file.name
                    )
                )
            self._geometries_by_layer = {
                _layer_id(pen): geo
                for pen, geo in self._drawing.geometries_by_pen.items()
            }
        return self._drawing

    def _bounds(self) -> tuple[float, float, float, float] | None:
        merged = self._merged_geometry()
        if merged is None:
            return None
        min_x, min_y, max_x, max_y = merged.rect()
        return min_x, min_y, max_x - min_x, max_y - min_y

    def scan(self) -> ImportManifest:
        drawing = self._get_drawing()
        bounds = self._bounds()
        layers = [
            LayerInfo(
                id=_layer_id(pen),
                name=_("Pen {number}").format(number=pen),
                feature_count=len(geo.split_into_contours()),
            )
            for pen, geo in drawing.geometries_by_pen.items()
        ]
        return ImportManifest(
            title=self.source_file.name,
            layers=layers,
            natural_size_mm=(bounds[2], bounds[3]) if bounds else None,
            warnings=self._warnings,
            errors=self._errors,
        )

    def parse(self) -> ParsingResult | None:
        self._get_drawing()
        bounds = self._bounds()
        if bounds is None:
            return None

        temp_result = ParsingResult(
            document_bounds=bounds,
            native_unit_to_mm=1.0,
            is_y_down=False,
            layers=[],
            world_frame_of_reference=bounds,
            background_world_transform=None,  # type: ignore
        )
        bg_item = NormalizationEngine.calculate_layout_item(
            bounds, temp_result
        )
        return ParsingResult(
            document_bounds=bounds,
            native_unit_to_mm=1.0,
            is_y_down=False,
            layers=[
                self._layer_geometry(layer_id, geo)
                for layer_id, geo in self._geometries_by_layer.items()
                if layer_id is not None
            ],
            world_frame_of_reference=bounds,
            background_world_transform=bg_item.world_matrix,
        )

    @staticmethod
    def _layer_geometry(layer_id: str, geo: Geometry) -> LayerGeometry:
        min_x, min_y, max_x, max_y = geo.rect()
        return LayerGeometry(
            layer_id=layer_id,
            name=layer_id,
            content_bounds=(min_x, min_y, max_x - min_x, max_y - min_y),
        )

    def create_source_asset(self, parse_result: ParsingResult) -> SourceAsset:
        _x, _y, w, h = parse_result.document_bounds
        return SourceAsset(
            source_file=self.source_file,
            original_data=self.raw_data,
            renderer=HPGL_RENDERER,
            metadata={"is_vector": True},
            thumbnail_data=self._render_thumbnail(),
            width_mm=w,
            height_mm=h,
        )
