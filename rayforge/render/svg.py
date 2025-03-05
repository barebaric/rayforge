import re
import warnings
with warnings.catch_warnings():
    warnings.simplefilter("ignore", DeprecationWarning)
    import pyvips
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

        return ET.tostring(root, encoding="unicode").encode('utf-8')
