from typing import Optional, List
import logging
from ...core.item import DocItem
from ...core.workpiece import WorkPiece
from ...core.vectorization_config import TraceConfig
from ...core.geometry import Geometry
from ...shared.util.tracing import trace_surface
from ..base_importer import Importer
from .renderer import SVG_RENDERER

logger = logging.getLogger(__name__)

# A standard fallback conversion factor for pixel units when no other
# context is provided. Corresponds to 96 DPI. (1 inch / 96 px) * 25.4 mm/inch
MM_PER_PX_FALLBACK = 25.4 / 96.0


class SvgImporter(Importer):
    label = "SVG files"
    mime_types = ("image/svg+xml",)
    extensions = (".svg",)

    def get_doc_items(
        self, vector_config: Optional["TraceConfig"] = None
    ) -> Optional[List[DocItem]]:
        # Create a workpiece first to handle potential errors gracefully.
        wp = WorkPiece(
            source_file=self.source_file,
            renderer=SVG_RENDERER,
            data=self.raw_data,
        )

        try:
            # Get the SVG's natural size in millimeters.
            size_mm = SVG_RENDERER.get_natural_size(
                wp, px_factor=MM_PER_PX_FALLBACK
            )
            if size_mm and size_mm[0] and size_mm[1]:
                wp.set_size(size_mm[0], size_mm[1])
            else:
                size_mm = None
        except Exception:
            return [wp]

        # If tracing is not requested or fails, return the bitmap workpiece.
        if not vector_config:
            return [wp]

        if not size_mm:
            return [wp]

        w_mm, h_mm = size_mm
        # Render at a reasonable resolution for tracing
        w_px, h_px = 2048, 2048
        surface = SVG_RENDERER.render_to_pixels(wp, w_px, h_px)
        if not surface:
            return [wp]

        pixels_per_mm = (w_px / w_mm, h_px / h_mm)

        geometries = trace_surface(surface, pixels_per_mm)
        if not geometries:
            return [wp]

        # Combine all traced paths into a single Geometry object.
        combined_geo = Geometry()
        for geo in geometries:
            combined_geo.commands.extend(geo.commands)

        # Update the workpiece with the generated vectors.
        wp.vectors = combined_geo
        return [wp]
