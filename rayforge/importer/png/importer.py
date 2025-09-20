import warnings
from typing import Optional, List
import logging
import cairo

with warnings.catch_warnings():
    warnings.simplefilter("ignore", DeprecationWarning)
    import pyvips

from ...core.item import DocItem
from ...core.workpiece import WorkPiece
from ..base_importer import Importer
from ...core.vectorization_config import TraceConfig
from ...core.geo import Geometry
from ...shared.util.tracing import trace_surface
from .renderer import PNG_RENDERER


logger = logging.getLogger(__name__)


class PngImporter(Importer):
    label = "PNG files"
    mime_types = ("image/png",)
    extensions = (".png",)

    def get_doc_items(
        self, vector_config: Optional["TraceConfig"] = None
    ) -> Optional[List[DocItem]]:
        if not vector_config:
            # In the new async model, raster importers MUST receive a config.
            logger.error("PngImporter requires a vector_config to trace.")
            return None

        try:
            image = pyvips.Image.pngload_buffer(self.raw_data)
        except pyvips.Error:
            return None  # Return None if the PNG data is invalid

        # 1. Prepare image for tracing by converting to a Cairo surface
        if image.bands < 4:
            image = image.bandjoin(255)  # Ensure alpha channel
        b, g, r, a = image[2], image[1], image[0], image[3]
        bgra_image = b.bandjoin([g, r, a])
        mem_buffer: memoryview = bgra_image.write_to_memory()
        surface = cairo.ImageSurface.create_for_data(
            mem_buffer,
            cairo.FORMAT_ARGB32,
            image.width,
            image.height,
            image.width * 4,
        )

        # 2. Determine physical size of the entire image
        width_mm, height_mm = image.width, image.height
        try:
            # VIPS stores resolution as pixels per millimeter
            xres = image.get("xres")
            yres = image.get("yres")
            width_mm = image.width / xres
            height_mm = image.height / yres
        except pyvips.Error:
            # Fallback to a standard 96 DPI if no resolution is set
            width_mm = image.width * (25.4 / 96.0)
            height_mm = image.height * (25.4 / 96.0)

        pixels_per_mm = (image.width / width_mm, image.height / height_mm)

        # 3. Trace the surface. The returned vectors are already correctly
        #    positioned relative to the top-left of the image.
        geometries = trace_surface(surface, pixels_per_mm)
        if not geometries:
            return []

        # 4. Combine all traced paths into a single Geometry object.
        #    DO NOT normalize or transform them.
        combined_geo = Geometry()
        for geo in geometries:
            combined_geo.commands.extend(geo.commands)

        # 5. Create the final workpiece. It represents the ENTIRE image.
        final_wp = WorkPiece(
            source_file=self.source_file,
            renderer=PNG_RENDERER,
            vectors=combined_geo,
            data=self.raw_data,
        )
        # The workpiece's size is the full physical size of the PNG.
        final_wp.set_size(width_mm, height_mm)
        # The initial position is (0,0). FileCmd will center it later.
        final_wp.pos = (0, 0)

        return [final_wp]
