import warnings
from typing import Optional, List
import logging
import cairo

with warnings.catch_warnings():
    warnings.simplefilter("ignore", DeprecationWarning)
    try:
        import pyvips
    except ImportError:
        raise ImportError("The BMP importer requires the pyvips library.")

from ...core.item import DocItem
from ...core.workpiece import WorkPiece
from ..base_importer import Importer
from ...core.vectorization_config import TraceConfig
from ...core.geo import Geometry
from ...shared.util.tracing import trace_surface
from .renderer import BMP_RENDERER
from .parser import parse_bmp

logger = logging.getLogger(__name__)


class BmpImporter(Importer):
    label = "BMP files"
    mime_types = ("image/bmp",)
    extensions = (".bmp",)

    def get_doc_items(
        self, vector_config: Optional["TraceConfig"] = None
    ) -> Optional[List[DocItem]]:
        if not vector_config:
            logger.error("BmpImporter requires a vector_config to trace.")
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

        try:
            # Step 2: Create a clean pyvips image from the RGBA buffer.
            image = pyvips.Image.new_from_memory(
                rgba_bytes, width, height, 4, "uchar"
            )
        except pyvips.Error as e:
            logger.error(
                "Failed to create pyvips image from parsed BMP data: %s", e
            )
            return None

        # Step 3: Proceed with the known-good pyvips image for tracing.
        width_mm = width * (25.4 / dpi_x)
        height_mm = height * (25.4 / dpi_y)
        pixels_per_mm = (width / width_mm, height / height_mm)

        b, g, r, a = image[2], image[1], image[0], image[3]
        bgra_image = b.bandjoin([g, r, a])
        cairo_mem_buffer: memoryview = bgra_image.write_to_memory()

        surface = cairo.ImageSurface.create_for_data(
            cairo_mem_buffer,
            cairo.FORMAT_ARGB32,
            width,
            height,
            width * 4,
        )

        geometries = trace_surface(surface, pixels_per_mm)
        if not geometries:
            return []

        combined_geo = Geometry()
        for geo in geometries:
            combined_geo.commands.extend(geo.commands)

        final_wp = WorkPiece(
            source_file=self.source_file,
            renderer=BMP_RENDERER,
            vectors=combined_geo,
            data=self.raw_data,
        )
        final_wp.set_size(width_mm, height_mm)
        final_wp.pos = (0, 0)
        return [final_wp]
