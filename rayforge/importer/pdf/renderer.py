import cairo
import io
import warnings
import logging
from typing import Optional, Tuple
from pypdf import PdfReader

with warnings.catch_warnings():
    warnings.simplefilter("ignore", DeprecationWarning)
    import pyvips

from ...core.workpiece import WorkPiece
from ..base_renderer import Renderer
from ..shared.util import to_mm

logger = logging.getLogger(__name__)


class PdfRenderer(Renderer):
    """Renders PDF data from a WorkPiece."""

    def render_data_to_vips_image(
        self, data: bytes, width: int, height: int
    ) -> Optional[pyvips.Image]:
        """Renders raw PDF data to a pyvips Image of specific dimensions."""
        if not data:
            return None
        try:
            reader = PdfReader(io.BytesIO(data))
            media_box = reader.pages[0].mediabox
            natural_w_mm = to_mm(float(media_box.width), "pt")

            dpi = (width / natural_w_mm) * 25.4 if natural_w_mm > 0 else 300

            image = pyvips.Image.pdfload_buffer(data, dpi=dpi)
            if not isinstance(image, pyvips.Image) or image.width == 0:
                return None

            h_scale = width / image.width
            v_scale = height / image.height
            return image.resize(h_scale, vscale=v_scale)
        except Exception:
            logger.warning(
                "Failed to render PDF data to vips image.", exc_info=True
            )
            return None

    def _render_to_vips_image(
        self, workpiece: "WorkPiece", width: int, height: int
    ) -> Optional[pyvips.Image]:
        if not workpiece.data:
            return None
        return self.render_data_to_vips_image(workpiece.data, width, height)

    def get_natural_size(
        self, workpiece: "WorkPiece"
    ) -> Optional[Tuple[float, float]]:
        if not workpiece.data:
            return None
        try:
            reader = PdfReader(io.BytesIO(workpiece.data))
            media_box = reader.pages[0].mediabox
            return (
                to_mm(float(media_box.width), "pt"),
                to_mm(float(media_box.height), "pt"),
            )
        except Exception:
            logger.warning(
                "Failed to get natural size from PDF data.", exc_info=True
            )
            return None

    def render_to_pixels(
        self, workpiece: "WorkPiece", width: int, height: int
    ) -> Optional[cairo.ImageSurface]:
        final_image = self.get_or_create_vips_image(workpiece, width, height)
        if not final_image:
            return None
        if final_image.bands < 4:
            final_image = final_image.bandjoin(255)

        b, g, r, a = (
            final_image[2],
            final_image[1],
            final_image[0],
            final_image[3],
        )
        bgra_image = b.bandjoin([g, r, a])
        mem_buffer = bgra_image.write_to_memory()

        return cairo.ImageSurface.create_for_data(
            mem_buffer,
            cairo.FORMAT_ARGB32,
            final_image.width,
            final_image.height,
            final_image.width * 4,
        )


PDF_RENDERER = PdfRenderer()
