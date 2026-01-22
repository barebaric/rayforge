import warnings
from typing import Optional
import logging
from pathlib import Path

with warnings.catch_warnings():
    warnings.simplefilter("ignore", DeprecationWarning)
    try:
        import pyvips
    except ImportError:
        raise ImportError("The BMP importer requires the pyvips library.")

from ...core.geo import Geometry
from ...core.source_asset import SourceAsset
from ...core.vectorization_spec import TraceSpec, VectorizationSpec
from ..base_importer import (
    Importer,
    ImporterFeature,
)
from ..tracing import trace_surface
from .. import image_util
from ..structures import (
    ParsingResult,
    LayerGeometry,
    VectorizationResult,
    ImportManifest,
)
from .parser import parse_bmp
from .renderer import BMP_RENDERER

logger = logging.getLogger(__name__)


class BmpImporter(Importer):
    label = "BMP files"
    mime_types = ("image/bmp",)
    extensions = (".bmp",)
    features = {ImporterFeature.BITMAP_TRACING}

    def __init__(self, data: bytes, source_file: Optional[Path] = None):
        super().__init__(data, source_file)
        self._image: Optional[pyvips.Image] = None

    def scan(self) -> ImportManifest:
        """
        Scans the BMP header to extract dimensions and calculate physical size.
        """
        try:
            parsed_data = parse_bmp(self.raw_data)
            if not parsed_data:
                return ImportManifest(
                    title=self.source_file.name,
                    warnings=[
                        "Could not parse BMP header. File may be unsupported."
                    ],
                )

            _, width, height, dpi_x, dpi_y = parsed_data
            dpi_x = dpi_x or 96.0
            dpi_y = dpi_y or 96.0

            width_mm = float(width) * (25.4 / dpi_x)
            height_mm = float(height) * (25.4 / dpi_y)

            return ImportManifest(
                title=self.source_file.name,
                natural_size_mm=(width_mm, height_mm),
            )
        except Exception as e:
            logger.warning(f"BMP scan failed for {self.source_file.name}: {e}")
            return ImportManifest(
                title=self.source_file.name,
                warnings=[
                    "An unexpected error occurred while scanning the BMP file."
                ],
            )

    def create_source_asset(self, parse_result: ParsingResult) -> SourceAsset:
        """
        Creates a SourceAsset for BMP import.
        """
        _, _, w_px, h_px = parse_result.page_bounds
        width_mm = w_px * parse_result.native_unit_to_mm
        height_mm = h_px * parse_result.native_unit_to_mm

        return SourceAsset(
            source_file=self.source_file,
            original_data=self.raw_data,
            renderer=BMP_RENDERER,
            width_px=int(w_px),
            height_px=int(h_px),
            width_mm=width_mm,
            height_mm=height_mm,
        )

    def vectorize(
        self,
        parse_result: ParsingResult,
        spec: VectorizationSpec,
    ) -> VectorizationResult:
        assert self._image is not None, "parse() must be called first"
        assert isinstance(spec, TraceSpec), (
            "BmpImporter only supports TraceSpec"
        )

        surface = image_util.vips_rgba_to_cairo_surface(self._image)
        geometries_list = trace_surface(surface, spec)
        merged_geometry = Geometry()
        for geo in geometries_list:
            merged_geometry.extend(geo)
        # For now, all traced geometry goes into a single "layer"
        return VectorizationResult(
            geometries_by_layer={None: merged_geometry},
            source_parse_result=parse_result,
        )

    def parse(self) -> Optional[ParsingResult]:
        """
        Phase 1: Parsing.

        Parses the BMP file and returns a ParsingResult containing geometric
        facts about the image in its native coordinate system (pixels). The
        parsed pyvips.Image is stored in self._image.
        """
        parsed_data = parse_bmp(self.raw_data)
        if not parsed_data:
            self._image = None
            return None

        rgba_bytes, width, height, dpi_x, dpi_y = parsed_data
        dpi_x = dpi_x or 96.0
        dpi_y = dpi_y or 96.0

        try:
            image = pyvips.Image.new_from_memory(
                rgba_bytes, width, height, 4, "uchar"
            )
            image = image.copy(
                interpretation=pyvips.Interpretation.SRGB,
                xres=dpi_x / 25.4,
                yres=dpi_y / 25.4,
            )
            self._image = image
        except pyvips.Error as e:
            logger.error(
                "Failed to create pyvips image from parsed BMP data: %s", e
            )
            self._image = None
            return None

        # Calculate unit conversion (pixels to mm)
        native_unit_to_mm = 25.4 / dpi_x

        # Page bounds are the full image dimensions
        page_bounds = (0.0, 0.0, float(width), float(height))

        # For tracing, geometries are traced from the full image, so content
        # bounds should match page bounds to ensure correct alignment.
        content_bounds = page_bounds

        # BMP is a single-layer format, use a default layer ID
        layer_id = "__default__"

        parse_result = ParsingResult(
            page_bounds=page_bounds,
            native_unit_to_mm=native_unit_to_mm,
            is_y_down=True,
            layers=[
                LayerGeometry(
                    layer_id=layer_id,
                    name=layer_id,
                    content_bounds=content_bounds,
                )
            ],
        )

        return parse_result
