from typing import Optional, List
from ...core.item import DocItem
from ...core.workpiece import WorkPiece
from ...core.vectorization_config import TraceConfig
from ...core.geometry import Geometry
from ...shared.util.tracing import trace_surface
from ..base_importer import Importer
from .renderer import PDF_RENDERER
import logging

logger = logging.getLogger(__name__)


class PdfImporter(Importer):
    label = "PDF files"
    mime_types = ("application/pdf",)
    extensions = (".pdf",)

    def get_doc_items(
        self, vector_config: Optional["TraceConfig"] = None
    ) -> Optional[List[DocItem]]:
        # Create a workpiece first to handle potential errors gracefully.
        wp = WorkPiece(
            source_file=self.source_file,
            renderer=PDF_RENDERER,
            data=self.raw_data,
        )

        try:
            # Get the PDF's natural size in millimeters.
            size_mm = PDF_RENDERER.get_natural_size(wp)
            if size_mm and size_mm[0] and size_mm[1]:
                wp.set_size(size_mm[0], size_mm[1])
            else:
                size_mm = None
        except Exception:
            # If size can't be determined (e.g., invalid PDF), return the
            # workpiece without a size. The UI can handle this.
            return [wp]

        # If tracing is not requested or fails, return the bitmap workpiece.
        if not vector_config:
            return [wp]

        if not size_mm:
            return [wp]  # Cannot trace without a size

        w_mm, h_mm = size_mm
        # Render at a reasonable resolution for tracing
        w_px = 2048
        h_px = int(w_px * (h_mm / w_mm)) if w_mm > 0 else 2048
        surface = PDF_RENDERER.render_to_pixels(wp, w_px, h_px)
        if not surface:
            return [wp]  # Fallback to bitmap if render fails

        pixels_per_mm = (w_px / w_mm, h_px / h_mm) if w_mm and h_mm else (1, 1)
        geometries = trace_surface(surface, pixels_per_mm)
        if not geometries:
            return [wp]  # Fallback to bitmap if trace fails

        # Combine all traced paths into a single Geometry object.
        combined_geo = Geometry()
        for geo in geometries:
            combined_geo.commands.extend(geo.commands)

        # Update the workpiece with the generated vectors.
        wp.vectors = combined_geo

        return [wp]
