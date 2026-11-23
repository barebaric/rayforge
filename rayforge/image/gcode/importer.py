import logging
from gettext import gettext as _
from pathlib import Path
from typing import ClassVar

from raygeo.gcode import GcodeParseResult, parse_gcode
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
from .renderer import GCODE_RENDERER

logger = logging.getLogger(__name__)

LAYER_ID = "__default__"


class GcodeImporter(DxfImporter):
    """
    Imports G-code programs as plain cutting geometry. Rapid moves and
    feed moves at spindle speed zero are treated as travel and excluded.
    Unsupported instructions are reported as warnings. Layer filtering
    and splitting are inherited from the DXF importer, which works on
    the same Y-up, millimetre geometry.
    """

    label = "G-code files"
    mime_types = ("text/x.gcode", "application/x-gcode")
    extensions = (".gcode", ".ngc", ".nc", ".tap", ".gc")
    # System databases map .nc to NetCDF and .tap to other formats, so
    # a guessed type must never be recorded for a G-code asset.
    native_mime_type = "text/x.gcode"
    features: ClassVar[set[ImporterFeature]] = {
        ImporterFeature.DIRECT_VECTOR,
    }

    def __init__(self, data: bytes, source_file: Path | None = None):
        super().__init__(data, source_file)
        self._result: GcodeParseResult | None = None
        self._geometry: Geometry | None = None

    def _get_result(self) -> GcodeParseResult:
        if self._result is None:
            text = self.raw_data.decode("utf-8", errors="replace")
            result = parse_gcode(text)
            if result.unsupported:
                names = ", ".join(result.unsupported)
                self.add_warning(
                    _(
                        "Some G-code instructions are not supported and "
                        "were skipped: {names}"
                    ).format(names=names)
                )
            geometry = result.geometry
            self._geometry = geometry
            if geometry.is_empty():
                self.add_error(
                    _("{filename} contains no G-code cut moves.").format(
                        filename=self.source_file.name
                    )
                )
            self._result = result
            self._geometries_by_layer = {LAYER_ID: geometry}
        return self._result

    def _get_geometry(self) -> Geometry:
        self._get_result()
        assert self._geometry is not None
        return self._geometry

    def _layer_name(self) -> str:
        return self.source_file.stem or _("G-code")

    def _bounds(self) -> tuple[float, float, float, float] | None:
        merged = self._merged_geometry()
        if merged is None:
            return None
        min_x, min_y, max_x, max_y = merged.rect()
        return min_x, min_y, max_x - min_x, max_y - min_y

    def scan(self) -> ImportManifest:
        geometry = self._get_geometry()
        bounds = self._bounds()
        layers = []
        if not geometry.is_empty():
            layers.append(
                LayerInfo(
                    id=LAYER_ID,
                    name=self._layer_name(),
                    feature_count=len(geometry.split_into_contours()),
                )
            )
        return ImportManifest(
            title=self.source_file.name,
            layers=layers,
            natural_size_mm=(bounds[2], bounds[3]) if bounds else None,
            warnings=self._warnings,
            errors=self._errors,
        )

    def parse(self) -> ParsingResult | None:
        self._get_result()
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
                LayerGeometry(
                    layer_id=LAYER_ID,
                    name=self._layer_name(),
                    content_bounds=bounds,
                )
            ],
            world_frame_of_reference=bounds,
            background_world_transform=bg_item.world_matrix,
        )

    def create_source_asset(self, parse_result: ParsingResult) -> SourceAsset:
        _x, _y, w, h = parse_result.document_bounds
        return SourceAsset(
            source_file=self.source_file,
            original_data=self.raw_data,
            renderer=GCODE_RENDERER,
            metadata={"is_vector": True},
            thumbnail_data=self._render_thumbnail(),
            width_mm=w,
            height_mm=h,
        )
