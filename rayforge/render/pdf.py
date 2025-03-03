import re
import io
import cairo
from PIL import Image
import pymupdf
from pypdf import PdfReader, PdfWriter
from ..util.unit import to_mm
from .renderer import Renderer


def parse_length(s):
    m = re.match(r"([0-9.]+)\s*([a-z%]*)", s)
    if m:
        return float(m.group(1)), m.group(2) or "pt"
    return float(s), "pt"


class PDFRenderer(Renderer):
    label = 'PDF files'
    mime_types = ('application/pdf',)
    extensions = ('.pdf',)

    @classmethod
    def prepare(cls, data):
        return cls._crop_to_content(data)

    @classmethod
    def get_natural_size(cls, data, px_factor=0):
        reader = PdfReader(io.BytesIO(data))
        page = reader.pages[0]
        media_box = page.mediabox
        width_pt = float(media_box.width)
        height_pt = float(media_box.height)
        return to_mm(width_pt, "pt", px_factor), \
               to_mm(height_pt, "pt", px_factor)

    @classmethod
    def get_aspect_ratio(cls, data):
        width_mm, height_mm = cls.get_natural_size(data)
        return width_mm / height_mm

    @classmethod
    def render_workpiece(cls, data, width=None, height=None):
        return cls._render_data(data, width, height)

    @classmethod
    def _render_data(cls, data, width=None, height=None):
        doc = pymupdf.open(stream=data, filetype="pdf")
        page = doc.load_page(0)
        zoom_x, zoom_y = 1.0, 1.0

        if width or height:
            rect = page.rect
            zoom_x = width / rect.width if width else 1.0
            zoom_y = height / rect.height if height else 1.0

        matrix = pymupdf.Matrix(zoom_x, zoom_y)
        pix = page.get_pixmap(matrix=matrix, alpha=True)

        # Convert the pixmap to a Pillow image
        img = Image.frombytes("RGBA", [pix.width, pix.height], pix.samples)

        # Save the Pillow image to an in-memory PNG buffer
        buffer = io.BytesIO()
        img.save(buffer, format="PNG")
        buffer.seek(0)

        return cairo.ImageSurface.create_from_png(buffer)

    @classmethod
    def _get_margins(cls, data):
        doc = pymupdf.open(stream=data, filetype="pdf")
        page = doc.load_page(0)
        pix = page.get_pixmap(matrix=pymupdf.Matrix(1, 1), alpha=False)
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        gray_img = img.convert("L")

        # Invert to find non-white regions
        inverted = Image.eval(gray_img, lambda x: 255 - x)
        bbox = inverted.getbbox()

        if not bbox:
            return (0.0, 0.0, 0.0, 0.0)

        x_min, y_min, x_max, y_max = bbox
        img_w, img_h = gray_img.size

        left_pct = x_min / img_w
        top_pct = y_min / img_h
        right_pct = (img_w - x_max) / img_w
        bottom_pct = (img_h - y_max) / img_h

        return left_pct, top_pct, right_pct, bottom_pct

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
