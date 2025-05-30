import re
import io
import warnings
with warnings.catch_warnings():
    warnings.simplefilter("ignore", DeprecationWarning)
    import pyvips  # type: ignore
from pypdf import PdfReader, PdfWriter
from ..util.unit import to_mm
from .vips import VipsRenderer


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
        reader = PdfReader(io.BytesIO(data))
        page = reader.pages[0]
        media_box = page.mediabox
        width_pt = float(media_box.width)
        height_pt = float(media_box.height)
        return (
            to_mm(width_pt, "pt", px_factor),
            to_mm(height_pt, "pt", px_factor),
        )

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
