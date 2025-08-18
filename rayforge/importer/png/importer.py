import warnings
from typing import Optional, List

with warnings.catch_warnings():
    warnings.simplefilter("ignore", DeprecationWarning)
    import pyvips

from ...core.item import DocItem
from ...core.workpiece import WorkPiece
from ..base_importer import Importer
from .renderer import PNG_RENDERER


class PngImporter(Importer):
    label = "PNG files"
    mime_types = ("image/png",)
    extensions = (".png",)

    def get_doc_items(self) -> Optional[List[DocItem]]:
        # The importer's only job is to create the WorkPiece.
        # It should not have dependencies on application state like `config`.
        try:
            # We only need to check if the data is valid. The renderer will
            # handle the actual loading.
            pyvips.Image.pngload_buffer(self.raw_data)
        except pyvips.Error:
            return None  # Return None if the PNG data is invalid

        wp = WorkPiece(
            source_file=self.source_file,
            renderer=PNG_RENDERER,
            data=self.raw_data,
        )

        # Get the PNG's natural size in millimeters from the renderer.
        size_mm = PNG_RENDERER.get_natural_size(wp)

        # Set the workpiece's initial size to its natural size.
        if (
            size_mm
            and size_mm[0] is not None
            and size_mm[1] is not None
            and size_mm[0] > 0
            and size_mm[1] > 0
        ):
            wp.set_size(size_mm[0], size_mm[1])

        return [wp]
