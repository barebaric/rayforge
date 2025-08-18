from typing import Optional, List
from ...core.item import DocItem
from ...core.workpiece import WorkPiece
from ..base_importer import Importer
from .renderer import SVG_RENDERER

# A standard fallback conversion factor for pixel units when no other
# context is provided. Corresponds to 96 DPI. (1 inch / 96 px) * 25.4 mm/inch
MM_PER_PX_FALLBACK = 25.4 / 96.0


class SvgImporter(Importer):
    label = "SVG files"
    mime_types = ("image/svg+xml",)
    extensions = (".svg",)

    def get_doc_items(self) -> Optional[List[DocItem]]:
        # The importer's responsibility is simply to create the WorkPiece.
        # Sizing and positioning should be handled by the application layer
        # that calls the importer.
        wp = WorkPiece(
            source_file=self.source_file,
            renderer=SVG_RENDERER,
            data=self.raw_data,
        )

        # Determine the natural size in mm. If the SVG uses 'px', we use a
        # standard fallback DPI to get a sensible default size.
        natural_size_mm = SVG_RENDERER.get_natural_size(
            wp, px_factor=MM_PER_PX_FALLBACK
        )

        if (
            natural_size_mm
            and natural_size_mm[0] is not None
            and natural_size_mm[1] is not None
            and natural_size_mm[0] > 0
            and natural_size_mm[1] > 0
        ):
            wp.set_size(natural_size_mm[0], natural_size_mm[1])

        return [wp]
