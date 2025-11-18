import warnings
import logging
from typing import Optional, TYPE_CHECKING
from ..base_renderer import Renderer

with warnings.catch_warnings():
    warnings.simplefilter("ignore", DeprecationWarning)
    import pyvips

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


class PdfRenderer(Renderer):
    """Renders PDF data."""

    def render_base_image(
        self,
        data: bytes,
        width: int,
        height: int,
        **kwargs,
    ) -> Optional[pyvips.Image]:
        if not data:
            return None

        # For PDFs, we must determine a DPI to request from the loader
        # to achieve the target pixel dimensions.
        # Since we don't have access to metadata here, we rely on the
        # loader's "scale" parameter if available or default DPI handling.
        # pyvips.pdfload_buffer takes a 'scale' parameter which is easier
        # than DPI if we want target pixels, but 'scale' is relative to 72DPI.

        # However, to correctly calculate the DPI/scale, we would need the
        # original PDF point size. Since renderers are now dumb, we rely
        # on a best-effort approach using a standard DPI or we could parse
        # it here locally if strictly necessary.
        # BUT: The architectural decision is that renderers just render.
        # The PdfImporter calculates the required pixel dimensions based on
        # the physical size it knows. The renderer's job is to hit that
        # pixel count.

        # To hit exact pixel count with PDF, we need the original point size.
        # We'll do a lightweight parse here solely for scaling purposes.
        try:
            # We use a very fast peek just for the MediaBox to calc DPI
            from pypdf import PdfReader
            import io

            reader = PdfReader(io.BytesIO(data))
            media_box = reader.pages[0].mediabox
            w_pt = float(media_box.width)
            h_pt = float(media_box.height)

            if w_pt > 0 and h_pt > 0:
                # Target DPI = (pixels / points) * 72
                dpi_x = (width / w_pt) * 72.0
                dpi_y = (height / h_pt) * 72.0
                dpi = max(dpi_x, dpi_y)
            else:
                dpi = 300.0
        except Exception:
            dpi = 300.0  # Fallback

        try:
            image = pyvips.Image.pdfload_buffer(data, dpi=dpi)
            if not isinstance(image, pyvips.Image) or image.width == 0:
                return None
            return image
        except Exception:
            logger.warning(
                "Failed to render PDF data to vips image.", exc_info=True
            )
            return None


PDF_RENDERER = PdfRenderer()
