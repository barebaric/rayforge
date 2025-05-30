import re
import warnings
with warnings.catch_warnings():
    warnings.simplefilter("ignore", DeprecationWarning)
    import pyvips  # type: ignore
from xml.etree import ElementTree as ET
from ..util.unit import to_mm
from .vips import VipsRenderer


def parse_length(s):
    m = re.match(r"([0-9.]+)\s*([a-z%]*)", s)
    if m:
        return float(m.group(1)), m.group(2) or "px"
    return float(s), "px"


class SVGRenderer(VipsRenderer):
    label = 'SVG files'
    mime_types = ('image/svg+xml',)
    extensions = ('.svg',)

    @classmethod
    def get_vips_loader(cls):
        return pyvips.Image.svgload_buffer

    @classmethod
    def get_natural_size(cls, data, px_factor=0):
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
            width_mm = to_mm(width, width_unit, px_factor=px_factor)
            height_mm = to_mm(height, height_unit, px_factor=px_factor)
        except ValueError:
            return None, None

        return width_mm, height_mm

    @classmethod
    def _crop_to_content(cls, data):
        # Load the image with pyvips to get pixel dimensions
        kwargs = cls.get_vips_loader_args()
        vips_image = cls.get_vips_loader()(data, **kwargs)
        width_px = vips_image.width
        height_px = vips_image.height

        # Get content margins as percentages
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

        # Calculate the percentage equivalent of a 1-pixel margin
        margin_px = 1
        margin_left_pct = margin_px / width_px if width_px > 0 else 0
        margin_top_pct = margin_px / height_px if height_px > 0 else 0
        margin_right_pct = margin_px / width_px if width_px > 0 else 0
        margin_bottom_pct = margin_px / height_px if height_px > 0 else 0

        # Adjust the content margin percentages
        adjusted_left_pct = max(0, left_pct - margin_left_pct)
        adjusted_top_pct = max(0, top_pct - margin_top_pct)
        adjusted_right_pct = max(0, right_pct - margin_right_pct)
        adjusted_bottom_pct = max(0, bottom_pct - margin_bottom_pct)

        # Calculate new viewBox dimensions using adjusted percentages
        new_x = vb_x + adjusted_left_pct * vb_w
        new_y = vb_y + adjusted_top_pct * vb_h
        new_w = vb_w * (1 - adjusted_left_pct - adjusted_right_pct)
        new_h = vb_h * (1 - adjusted_top_pct - adjusted_bottom_pct)

        # Ensure new dimensions are not negative
        new_w = max(0, new_w)
        new_h = max(0, new_h)

        root.set("viewBox", f"{new_x} {new_y} {new_w} {new_h}")

        # Adjust width and height attributes based on the new viewBox size
        width_str = root.get("width")
        if width_str:
            width_val, unit = parse_length(width_str)
            # Scale the original width by the ratio of new_w to vb_w
            new_width_val = width_val * (new_w / vb_w) if vb_w > 0 else new_w
            root.set("width", f"{new_width_val}{unit}")

        height_str = root.get("height")
        if height_str:
            height_val, unit = parse_length(height_str)
            # Scale the original height by the ratio of new_h to vb_h
            new_height_val = height_val * (new_h / vb_h) if vb_h > 0 else new_h
            root.set("height", f"{new_height_val}{unit}")

        return ET.tostring(root, encoding="unicode").encode('utf-8')
