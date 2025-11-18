import warnings
from typing import Optional
import logging

with warnings.catch_warnings():
    warnings.simplefilter("ignore", DeprecationWarning)
    try:
        import pyvips
    except ImportError:
        raise ImportError("The BMP importer requires the pyvips library.")

from ...core.source_asset import SourceAsset
from ...core.vectorization_spec import TraceSpec, VectorizationSpec
from ..base_importer import Importer, ImportPayload
from ..tracing import trace_surface
from .. import image_util
from .parser import parse_bmp
from .renderer import BMP_RENDERER

logger = logging.getLogger(__name__)


class BmpImporter(Importer):
    label = "BMP files"
    mime_types = ("image/bmp",)
    extensions = (".bmp",)
    is_bitmap = True

    def get_doc_items(
        self, vectorization_spec: Optional["VectorizationSpec"] = None
    ) -> Optional[ImportPayload]:
        if not isinstance(vectorization_spec, TraceSpec):
            logger.error("BmpImporter requires a TraceSpec to trace.")
            return None

        # Step 1: Use the parser to get clean pixel data and metadata.
        parsed_data = parse_bmp(self.raw_data)
        if not parsed_data:
            logger.error(
                "BMP file could not be parsed. It may be compressed or in an "
                "unsupported format."
            )
            return None

        rgba_bytes, width, height, dpi_x, dpi_y = parsed_data
        dpi_x = dpi_x or 96.0
        dpi_y = dpi_y or 96.0

        # Calculate physical dimensions based on parsed DPI
        width_mm = float(width) * (25.4 / dpi_x)
        height_mm = float(height) * (25.4 / dpi_y)

        try:
            # Step 2: Create a clean pyvips image from the RGBA buffer.
            image = pyvips.Image.new_from_memory(
                rgba_bytes, width, height, 4, "uchar"
            )
            # Explicitly set the color interpretation and resolution.
            image = image.copy(
                interpretation=pyvips.Interpretation.SRGB,
                xres=dpi_x / 25.4,  # px/mm
                yres=dpi_y / 25.4,
            )
        except pyvips.Error as e:
            logger.error(
                "Failed to create pyvips image from parsed BMP data: %s", e
            )
            return None

        # Step 3: Create the SourceAsset with dimensions
        source = SourceAsset(
            source_file=self.source_file,
            original_data=self.raw_data,
            renderer=BMP_RENDERER,
            width_px=width,
            height_px=height,
            width_mm=width_mm,
            height_mm=height_mm,
        )

        # Step 4: Convert to a Cairo surface for tracing.
        surface = image_util.vips_rgba_to_cairo_surface(image)

        # Step 5: Trace the surface and create the WorkPiece(s).
        geometries = trace_surface(surface, vectorization_spec)

        # Step 6: Use helper to create a single, masked workpiece
        items = image_util.create_single_workpiece_from_trace(
            geometries,
            source,
            image,
            vectorization_spec,
            self.source_file.stem,
        )

        return ImportPayload(source=source, items=items)
