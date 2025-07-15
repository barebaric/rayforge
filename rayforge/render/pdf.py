import re
import io
from typing import Optional
import warnings
import logging
with warnings.catch_warnings():
    warnings.simplefilter("ignore", DeprecationWarning)
    import pyvips  # type: ignore
from pypdf import PdfReader, PdfWriter
from ..util.unit import to_mm
from .vips import VipsRenderer


logger = logging.getLogger(__name__)


def parse_length(s):
    m = re.match(r"([0-9.]+)\s*([a-z%]*)", s)
    if m:
        return float(m.group(1)), m.group(2) or "pt"
    return float(s), "pt"


class PDFRenderer(VipsRenderer):
    label = 'PDF files'
    mime_types = ('application/pdf',)
    extensions = ('.pdf',)

    @classmethod
    def get_vips_loader(cls):
        return pyvips.Image.pdfload_buffer

    @classmethod
    def get_vips_loader_args(cls):
        return {'background': (255, 255, 255, 0)}

    @classmethod
    def get_natural_size(cls, data, px_factor=0):
        try:
            reader = PdfReader(io.BytesIO(data))
            page = reader.pages[0]
            media_box = page.mediabox
            width_pt = float(media_box.width)
            height_pt = float(media_box.height)
            return (
                to_mm(width_pt, "pt", px_factor),
                to_mm(height_pt, "pt", px_factor),
            )
        except Exception as e:
            logger.error(f"Failed to get natural size from PDF: {e}")
            return None, None

    @classmethod
    def _crop_to_content(cls, data):
        left_pct, top_pct, right_pct, bottom_pct = cls._get_margins(data)

        reader = PdfReader(io.BytesIO(data))
        writer = PdfWriter()

        for page in reader.pages:
            media_box = page.mediabox
            x0 = float(media_box.left)
            y0 = float(media_box.bottom)
            x1 = float(media_box.right)
            y1 = float(media_box.top)
            width_pt = x1 - x0
            height_pt = y1 - y0

            new_x0 = x0 + left_pct * width_pt
            new_x1 = x1 - right_pct * width_pt
            new_y0 = y0 + bottom_pct * height_pt
            new_y1 = y1 - top_pct * height_pt

            # Create a new media box with the cropped dimensions
            page.mediabox.left = new_x0
            page.mediabox.bottom = new_y0
            page.mediabox.right = new_x1
            page.mediabox.top = new_y1

            writer.add_page(page)

        output = io.BytesIO()
        writer.write(output)
        return output.getvalue()

    @classmethod
    def _render_to_vips_image(
        cls, data: bytes, width: int, height: int
    ) -> Optional[pyvips.Image]:
        """
        Specialized PDF implementation that renders at a high DPI and then
        resizes to the exact target dimensions, ensuring sharpness and
        correct non-uniform scaling.
        """
        try:
            nat_w_mm, nat_h_mm = cls.get_natural_size(data)
            if not nat_w_mm or not nat_h_mm or nat_w_mm <= 0 or nat_h_mm <= 0:
                return super()._render_to_vips_image(data, width, height)

            nat_w_in = nat_w_mm / 25.4
            nat_h_in = nat_h_mm / 25.4

            dpi_x = width / nat_w_in
            dpi_y = height / nat_h_in
            target_dpi = max(dpi_x, dpi_y)

            image = pyvips.Image.pdfload_buffer(data, dpi=target_dpi)
            if (
                not isinstance(image, pyvips.Image)
                or image.width == 0
                or image.height == 0
            ):
                return None

            # Now, force a resize to the exact final dimensions.
            h_scale = width / image.width
            v_scale = height / image.height
            return image.resize(h_scale, vscale=v_scale)

        except pyvips.Error as e:
            logger.error(f"Error rendering PDF to vips image: {e}")
            return None
