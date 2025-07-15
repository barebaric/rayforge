import re
from typing import Optional
import logging
import warnings
with warnings.catch_warnings():
    warnings.simplefilter("ignore", DeprecationWarning)
    import pyvips  # type: ignore
from xml.etree import ElementTree as ET
from ..util.unit import to_mm
from .vips import VipsRenderer


logger = logging.Logger(__name__)


def parse_length(s):
    if not s:
        return 0.0, "px"
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
    def _get_margins(cls, data: bytes) -> tuple[float, float, float, float]:
        """
        Calculates content margins by rendering the SVG on a fixed-size
        canvas and finding the bounding box of non-transparent pixels.
        """
        measurement_size = 1000.0
        try:
            root_measure = ET.fromstring(data)
            root_measure.set('width', f'{measurement_size}px')
            root_measure.set('height', f'{measurement_size}px')
            root_measure.set('preserveAspectRatio', 'none')
            measure_svg = ET.tostring(root_measure)

            measure_image = pyvips.Image.svgload_buffer(measure_svg)
            if measure_image.bands < 4:
                measure_image = measure_image.bandjoin(255)

            left, top, w, h = measure_image.find_trim()
            if w == 0 or h == 0:
                return 0.0, 0.0, 0.0, 0.0

            return (
                left / measurement_size,
                top / measurement_size,
                (measurement_size - (left + w)) / measurement_size,
                (measurement_size - (top + h)) / measurement_size,
            )
        except (pyvips.Error, ET.ParseError):
            return 0.0, 0.0, 0.0, 0.0

    @classmethod
    def _crop_to_content(cls, data: bytes) -> bytes:
        """
        Crops the SVG to its content by calculating the content margins
        and adjusting the viewBox and dimensions accordingly. This overrides
        the base class's no-op implementation.
        """
        try:
            l, t, r, b = cls._get_margins(data)
            if all(m == 0.0 for m in (l, t, r, b)):
                return data

            root = ET.fromstring(data)
            w_attr = root.get("width")
            h_attr = root.get("height")
            if not w_attr or not h_attr:
                return data

            w_val, w_unit = parse_length(w_attr)
            h_val, h_unit = parse_length(h_attr)
            w_unit = w_unit or "px"
            h_unit = h_unit or "px"

            vb_str = root.get("viewBox")
            if vb_str:
                vb_x, vb_y, vb_w, vb_h = map(float, vb_str.split())
            else:
                vb_x, vb_y, vb_w, vb_h = 0, 0, w_val, h_val

            new_vb_x = vb_x + (l * vb_w)
            new_vb_y = vb_y + (t * vb_h)
            new_vb_w = vb_w * (1 - l - r)
            new_vb_h = vb_h * (1 - t - b)

            new_w = w_val * (1 - l - r)
            new_h = h_val * (1 - t - b)

            root.set("viewBox", f"{new_vb_x} {new_vb_y} {new_vb_w} {new_vb_h}")
            root.set("width", f"{new_w}{w_unit}")
            root.set("height", f"{new_h}{h_unit}")

            return ET.tostring(root, encoding='utf-8')

        except (pyvips.Error, ET.ParseError, ValueError):
            return data

    @classmethod
    def get_natural_size(cls, data, px_factor=0):
        """
        Calculates the natural size of the SVG's actual content by
        reading its declared dimensions and then adjusting for any margins.
        """
        try:
            root = ET.fromstring(data)
            w_attr = root.get("width")
            h_attr = root.get("height")
            if not w_attr or not h_attr:
                return None, None

            w_val, w_unit = parse_length(w_attr)
            h_val, h_unit = parse_length(h_attr)
            w_mm = to_mm(w_val, w_unit, px_factor=px_factor)
            h_mm = to_mm(h_val, h_unit, px_factor=px_factor)
        except (ValueError, ET.ParseError):
            return None, None

        # Adjust dimensions based on actual content margins
        l, t, r, b = cls._get_margins(data)
        content_w_mm = w_mm * (1 - l - r)
        content_h_mm = h_mm * (1 - t - b)

        return content_w_mm, content_h_mm

    @classmethod
    def _render_to_vips_image(
        cls, data: bytes, width: int, height: int
    ) -> Optional[pyvips.Image]:
        """
        Renders the SVG's true content directly to the target size by
        adjusting its viewBox based on pre-calculated margins.
        """
        l, t, r, b = cls._get_margins(data)

        try:
            root = ET.fromstring(data)
            vb_str = root.get("viewBox")
            if vb_str:
                vb_x, vb_y, vb_w, vb_h = map(float, vb_str.split())
            else:
                w_str = root.get("width")
                h_str = root.get("height")
                if not w_str or not h_str:
                    return None
                w_val, _ = parse_length(w_str)
                h_val, _ = parse_length(h_str)
                vb_x, vb_y, vb_w, vb_h = 0, 0, w_val, h_val

            # Calculate new, cropped viewBox
            new_vb_x = vb_x + (l * vb_w)
            new_vb_y = vb_y + (t * vb_h)
            new_vb_w = vb_w * (1 - l - r)
            new_vb_h = vb_h * (1 - t - b)

            # Set attributes for final, direct render
            root.set("viewBox",
                     f"{new_vb_x} {new_vb_y} {new_vb_w} {new_vb_h}")
            root.set("width", f"{width}px")
            root.set("height", f"{height}px")
            root.set("preserveAspectRatio", "none")

            final_svg = ET.tostring(root, encoding='utf-8')
            return pyvips.Image.svgload_buffer(final_svg)
        except (pyvips.Error, ET.ParseError) as e:
            logger.error(f"Final SVG render failed: {e}")
            return None
