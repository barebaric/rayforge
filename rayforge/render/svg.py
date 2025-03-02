import re
import io
import math
import cairo
import cairosvg
import warnings
with warnings.catch_warnings():
    warnings.simplefilter("ignore", DeprecationWarning)
    import pyvips
from xml.etree import ElementTree as ET
from PIL import Image
from .renderer import Renderer


def parse_length(s):
    m = re.match(r"([0-9.]+)\s*([a-z%]*)", s)
    if m:
        return float(m.group(1)), m.group(2) or "px"
    return float(s), "px"


def to_mm(value, unit):
    """Convert a value to millimeters based on its unit."""
    if unit == "cm":
        return value*10
    if unit == "mm":
        return value
    elif unit == "in":
        return value * 25.4  # 1 inch = 25.4 mm
    raise ValueError("Cannot convert to millimeters without DPI information.")


class SVGRenderer(Renderer):
    label = 'SVG files'
    mime_types = ('image/svg+xml',)
    extensions = ('.svg',)

    @classmethod
    def prepare(cls, data):
        return cls._crop_to_content(data)

    @classmethod
    def get_natural_size(cls, data):
        """
        Returns the natural size of the document in mm as a tuple (w, h).
        This is BEFORE cropping the margins.
        """
        # Parse the SVG from the bytestring
        root = ET.fromstring(data)

        # Extract width and height attributes
        width_attr = root.get("width")
        height_attr = root.get("height")

        if not width_attr or not height_attr:
            # SVG does not have width or height attributes.
            return None, None

        width, width_unit = parse_length(width_attr)
        height, height_unit = parse_length(height_attr)

        # Convert to millimeters
        try:
            width_mm = to_mm(width, width_unit)
            height_mm = to_mm(height, height_unit)
        except ValueError:
            return None, None

        return width_mm, height_mm

    @classmethod
    def get_aspect_ratio(cls, data):
        surface = cls._render_data(data)
        return surface.get_width()/surface.get_height()

    @classmethod
    def render_workpiece(cls, data, width=None, height=None):
        return cls._render_data(data, width, height)

    @classmethod
    def render_chunk(cls, data, chunk_width=32000, chunk_height=32000):
        # Render SVG to pyvips image
        vips_image = pyvips.Image.new_from_buffer(data, "")

        # Calculate number of chunks needed
        cols = math.ceil(vips_image.width / chunk_width)
        rows = math.ceil(vips_image.height / chunk_height)

        for row in range(rows):
            for col in range(cols):
                # Calculate chunk boundaries
                left = col * chunk_width
                top = row * chunk_height
                width = min(chunk_width, vips_image.width - left)
                height = min(chunk_height, vips_image.height - top)

                # Extract chunk
                chunk = vips_image.crop(left, top, width, height)

                # Convert to Cairo-compatible format by adding alpha.
                chunk_rgba = chunk.colourspace("srgb").bandjoin(255)
                buf = chunk_rgba.write_to_memory()

                # Create Cairo surface for the chunk
                surface = cairo.ImageSurface.create_for_data(
                    buf,
                    cairo.FORMAT_ARGB32,
                    chunk.width,
                    chunk.height,
                    chunk_rgba.width * 4  # Stride for RGBA
                )

                # Yield surface + position metadata
                yield (surface, (left, top))

    @classmethod
    def _render_data(cls, data, width=None, height=None):
        png_data = cairosvg.svg2png(bytestring=data,
                                    parent_height=height,
                                    output_height=height)
        return cairo.ImageSurface.create_from_png(io.BytesIO(png_data))

    @classmethod
    def _get_margins(cls, data):
        """
        Reliably finding the content width of an SVG is surprisingly hard.
        I tried several modules (svgelements, svg2paths2) and all methods
        failed depending on the content of the SVG.

        So instead I render the SVG to PNG, find the width and height
        of the content in relation to the PNG size, and apply the factor
        agains the viewport size of the SVG to get the actual bounds.
        """
        # Open the image with PIL.
        png_data = cairosvg.svg2png(bytestring=data)
        img = Image.open(io.BytesIO(png_data))

        # If the image has an alpha channel, use it to determine non-
        # transparent pixels.
        if img.mode in ('RGBA', 'LA'):
            bbox = img.split()[-1].getbbox()  # bbox of non-transparent pixels
        else:
            # Otherwise, convert to grayscale and compute bbox.
            bbox = img.convert("L").getbbox()

        # Calculate margin percentages relative to the full image dimensions
        x_min, y_min, x_max, y_max = bbox
        img_w, img_h = img.size
        left_pct = x_min / img_w
        top_pct = y_min / img_h
        right_pct = (img_w - x_max) / img_w
        bottom_pct = (img_h - y_max) / img_h

        return left_pct, top_pct, right_pct, bottom_pct

    @classmethod
    def _crop_to_content(cls, data):
        left_pct, top_pct, right_pct, bottom_pct = cls._get_margins(data)

        root = ET.fromstring(data)

        # Adjust viewBox by applying the margin percentages
        viewbox_str = root.get("viewBox")
        if not viewbox_str:
            # If no viewBox, use width and height as fallback
            width_str = root.get("width")
            height_str = root.get("height")
            if width_str and height_str:
                width = float(width_str)
                height = float(height_str)
                viewbox_str = f"0 0 {width} {height}"
                root.set("viewBox", viewbox_str)
            else:
                return data  # Cannot crop without dimensions

        vb_x, vb_y, vb_w, vb_h = map(float, viewbox_str.split())
        new_x = vb_x + left_pct * vb_w
        new_y = vb_y + top_pct * vb_h
        new_w = vb_w * (1 - left_pct - right_pct)
        new_h = vb_h * (1 - top_pct - bottom_pct)
        root.set("viewBox", f"{new_x} {new_y} {new_w} {new_h}")

        width_str = root.get("width")
        if width_str:
            width_val, unit = parse_length(width_str)
            root.set("width", f"{new_w}{unit}")
        height_str = root.get("height")
        if height_str:
            height_val, unit = parse_length(height_str)
            root.set("height", f"{new_h}{unit}")

        return ET.tostring(root, encoding="unicode")
