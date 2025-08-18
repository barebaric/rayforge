from typing import Optional, List
from ...core.item import DocItem
from ...core.workpiece import WorkPiece
from ..base_importer import Importer
from .renderer import PDF_RENDERER


class PdfImporter(Importer):
    label = "PDF files"
    mime_types = ("application/pdf",)
    extensions = (".pdf",)

    def get_doc_items(self) -> Optional[List[DocItem]]:
        wp = WorkPiece(
            source_file=self.source_file,
            renderer=PDF_RENDERER,
            data=self.raw_data,
        )

        # Get the PDF's natural size in millimeters.
        size_mm = PDF_RENDERER.get_natural_size(wp)

        # Set the workpiece's initial size to its natural size. This is a
        # much better default than scaling to fit the machine.
        if (
            size_mm
            and size_mm[0] is not None
            and size_mm[1] is not None
            and size_mm[0] > 0
            and size_mm[1] > 0
        ):
            wp.set_size(size_mm[0], size_mm[1])

        return [wp]
