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
from ..assembler import ItemAssembler
from ..base_importer import (
    Importer,
    ImportPayload,
    ImporterFeature,
    ImportManifest,
)
from ..engine import NormalizationEngine
from ..tracing import trace_surface
from .. import image_util
from ..structures import ParsingResult, LayerGeometry, VectorizationResult
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

    def get_doc_items(
        self, vectorization_spec: Optional["VectorizationSpec"] = None
    ) -> Optional[ImportPayload]:
        if not isinstance(vectorization_spec, TraceSpec):
            logger.error("BmpImporter requires a TraceSpec to trace.")
            return None

        # Phase 2 (Parse): Get ParsingResult and set self._image
        parse_result = self.parse()
        if not parse_result or not self._image:
            logger.error(
                "BMP file could not be parsed. It may be compressed or in an "
                "unsupported format."
            )
            return None

        # Create the SourceAsset with dimensions from parsing
        _, _, w, h = parse_result.page_bounds
        width_mm = w * parse_result.native_unit_to_mm
        height_mm = h * parse_result.native_unit_to_mm

        source = SourceAsset(
            source_file=self.source_file,
            original_data=self.raw_data,
            renderer=BMP_RENDERER,
            width_px=int(w),
            height_px=int(h),
            width_mm=width_mm,
            height_mm=height_mm,
        )

        # Phase 3 (Vectorize): Trace the image to get vector geometry
        vec_result = self.vectorize(parse_result, vectorization_spec)

        # Phase 4 (Layout): Calculate layout plan from vectorized geometry
        engine = NormalizationEngine()
        plan = engine.calculate_layout(vec_result, vectorization_spec)
        if not plan:
            logger.warning("Layout plan is empty; no items will be created.")
            return ImportPayload(source=source, items=[])

        # Phase 5 (Assemble): Create DocItems from the layout plan
        assembler = ItemAssembler()
        items = assembler.create_items(
            source_asset=source,
            layout_plan=plan,
            spec=vectorization_spec,
            source_name=self.source_file.stem,
            geometries=vec_result.geometries_by_layer,
        )

        return ImportPayload(source=source, items=items)

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
                LayerGeometry(layer_id=layer_id, content_bounds=content_bounds)
            ],
        )

        return parse_result
