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
    label = "SVG files"
    mime_types = ("image/svg+xml",)
    extensions = (".svg",)

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
            root_measure.set("width", f"{measurement_size}px")
            root_measure.set("height", f"{measurement_size}px")
            root_measure.set("preserveAspectRatio", "none")
            measure_svg = ET.tostring(root_measure)

            measure_image = pyvips.Image.svgload_buffer(measure_svg)
            if measure_image.bands < 4:
                measure_image = measure_image.bandjoin(255)

            left_px, top_px, width_px, height_px = measure_image.find_trim()
            if width_px == 0 or height_px == 0:
                return 0.0, 0.0, 0.0, 0.0

            left_margin_ratio = left_px / measurement_size
            top_margin_ratio = top_px / measurement_size
            right_margin_ratio = (
                measurement_size - (left_px + width_px)
            ) / measurement_size
            bottom_margin_ratio = (
                measurement_size - (top_px + height_px)
            ) / measurement_size

            return (
                left_margin_ratio,
                top_margin_ratio,
                right_margin_ratio,
                bottom_margin_ratio,
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
            left_margin, top_margin, right_margin, bottom_margin = (
                cls._get_margins(data)
            )
            if all(
                margin == 0.0
                for margin in (
                    left_margin,
                    top_margin,
                    right_margin,
                    bottom_margin,
                )
            ):
                return data

            root = ET.fromstring(data)
            width_attr = root.get("width")
            height_attr = root.get("height")
            if not width_attr or not height_attr:
                return data

            width_val, width_unit = parse_length(width_attr)
            height_val, height_unit = parse_length(height_attr)
            width_unit = width_unit or "px"
            height_unit = height_unit or "px"

            viewbox_str = root.get("viewBox")
            if viewbox_str:
                viewbox_x, viewbox_y, viewbox_width, viewbox_height = map(
                    float, viewbox_str.split()
                )
            else:
                viewbox_x, viewbox_y, viewbox_width, viewbox_height = (
                    0,
                    0,
                    width_val,
                    height_val,
                )

            new_viewbox_x = viewbox_x + (left_margin * viewbox_width)
            new_viewbox_y = viewbox_y + (top_margin * viewbox_height)
            new_viewbox_width = viewbox_width * (
                1 - left_margin - right_margin
            )
            new_viewbox_height = viewbox_height * (
                1 - top_margin - bottom_margin
            )

            new_width = width_val * (1 - left_margin - right_margin)
            new_height = height_val * (1 - top_margin - bottom_margin)

            root.set(
                "viewBox",
                f"{new_viewbox_x} {new_viewbox_y} "
                f"{new_viewbox_width} {new_viewbox_height}",
            )
            root.set("width", f"{new_width}{width_unit}")
            root.set("height", f"{new_height}{height_unit}")

            return ET.tostring(root, encoding="utf-8")

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
            width_attr = root.get("width")
            height_attr = root.get("height")
            if not width_attr or not height_attr:
                return None, None

            width_val, width_unit = parse_length(width_attr)
            height_val, height_unit = parse_length(height_attr)
            width_mm = to_mm(width_val, width_unit, px_factor=px_factor)
            height_mm = to_mm(height_val, height_unit, px_factor=px_factor)
        except (ValueError, ET.ParseError):
            return None, None

        # Adjust dimensions based on actual content margins
        left_margin, top_margin, right_margin, bottom_margin = (
            cls._get_margins(data)
        )
        content_w_mm = width_mm * (1 - left_margin - right_margin)
        content_h_mm = height_mm * (1 - top_margin - bottom_margin)

        return content_w_mm, content_h_mm

    @classmethod
    def _render_to_vips_image(
        cls, data: bytes, width: int, height: int
    ) -> Optional[pyvips.Image]:
        """
        Renders the SVG's true content directly to the target size by
        adjusting its viewBox based on pre-calculated margins.
        """
        left_margin, top_margin, right_margin, bottom_margin = (
            cls._get_margins(data)
        )

        try:
            root = ET.fromstring(data)
            viewbox_str = root.get("viewBox")
            if viewbox_str:
                viewbox_x, viewbox_y, viewbox_width, viewbox_height = map(
                    float, viewbox_str.split()
                )
            else:
                width_str = root.get("width")
                height_str = root.get("height")
                if not width_str or not height_str:
                    return None
                width_val, _ = parse_length(width_str)
                height_val, _ = parse_length(height_str)
                viewbox_x, viewbox_y, viewbox_width, viewbox_height = (
                    0,
                    0,
                    width_val,
                    height_val,
                )

            # Calculate new, cropped viewBox
            new_viewbox_x = viewbox_x + (left_margin * viewbox_width)
            new_viewbox_y = viewbox_y + (top_margin * viewbox_height)
            new_viewbox_width = viewbox_width * (
                1 - left_margin - right_margin
            )
            new_viewbox_height = viewbox_height * (
                1 - top_margin - bottom_margin
            )

            # Set attributes for final, direct render
            root.set(
                "viewBox",
                f"{new_viewbox_x} {new_viewbox_y} "
                f"{new_viewbox_width} {new_viewbox_height}",
            )
            root.set("width", f"{width}px")
            root.set("height", f"{height}px")
            root.set("preserveAspectRatio", "none")

            final_svg = ET.tostring(root, encoding="utf-8")
            return pyvips.Image.svgload_buffer(final_svg)
        except (pyvips.Error, ET.ParseError) as e:
            logger.error(f"Final SVG render failed: {e}")
            return None
