import re
from typing import Optional, Tuple, Dict
import logging
import warnings
with warnings.catch_warnings():
    warnings.simplefilter("ignore", DeprecationWarning)
    import pyvips
from xml.etree import ElementTree as ET
from .util import to_mm
from .renderer import Renderer, CAIRO_MAX_DIMENSION
import cairo

logger = logging.getLogger(__name__)


def parse_length(s):
    if not s:
        return 0.0, "px"
    m = re.match(r"([0-9.]+)\s*([a-z%]*)", s)
    if m:
        return float(m.group(1)), m.group(2) or "px"
    return float(s), "px"


class SVGRenderer(Renderer):
    label = "SVG files"
    mime_types = ("image/svg+xml",)
    extensions = (".svg",)

    def __init__(self, data: bytes):
        """
        Initializes the renderer.
        """
        self.raw_data = data
        self._margin_cache: Optional[Tuple[float, float, float, float]] = None
        self._natural_size_cache: Dict[
            float, Tuple[Optional[float], Optional[float]]
        ] = {}
        self._vips_image_cache: Dict[Tuple[int, int], pyvips.Image] = {}

    def _get_margins(self) -> Tuple[float, float, float, float]:
        """
        Calculates the empty margins around the SVG content.
        The result is cached in an instance attribute as it's expensive and
        never changes for the lifetime of the instance.
        """
        if self._margin_cache is not None:
            return self._margin_cache

        measurement_size = 1000.0
        try:
            root_measure = ET.fromstring(self.raw_data)
            root_measure.set("width", f"{measurement_size}px")
            root_measure.set("height", f"{measurement_size}px")
            root_measure.set("preserveAspectRatio", "none")
            measure_svg = ET.tostring(root_measure)

            measure_image = pyvips.Image.svgload_buffer(measure_svg)
            if measure_image.bands < 4:
                measure_image = measure_image.bandjoin(255)

            left_px, top_px, width_px, height_px = measure_image.find_trim()
            if width_px == 0 or height_px == 0:
                self._margin_cache = 0.0, 0.0, 0.0, 0.0
                return self._margin_cache

            left = left_px / measurement_size
            top = top_px / measurement_size
            right = (
                measurement_size - (left_px + width_px)
            ) / measurement_size
            bottom = (
                measurement_size - (top_px + height_px)
            ) / measurement_size

            self._margin_cache = left, top, right, bottom
        except (pyvips.Error, ET.ParseError):
            self._margin_cache = 0.0, 0.0, 0.0, 0.0
        return self._margin_cache

    def get_natural_size(
        self, px_factor: float = 0.0
    ) -> Tuple[Optional[float], Optional[float]]:
        """
        Calculates the natural size, correctly caching results for different
        px_factor values in an instance dictionary.
        """
        if px_factor in self._natural_size_cache:
            return self._natural_size_cache[px_factor]

        try:
            root = ET.fromstring(self.raw_data)
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

        left_margin, top_margin, right_margin, bottom_margin = (
            self._get_margins()
        )
        content_w_mm = width_mm * (1 - left_margin - right_margin)
        content_h_mm = height_mm * (1 - top_margin - bottom_margin)

        result = content_w_mm, content_h_mm
        self._natural_size_cache[px_factor] = result
        return result

    def get_aspect_ratio(self) -> float:
        """
        Calculates aspect ratio on-the-fly using the cached get_natural_size.
        """
        w, h = self.get_natural_size()
        if w and h and h > 0:
            return w / h
        return 1.0

    def _render_to_vips_image(
        self, width: int, height: int
    ) -> Optional[pyvips.Image]:
        """
        Renders the SVG to a vips image, caching the result in an instance
        dictionary. This is an expensive operation.
        """
        key = (width, height)
        if key in self._vips_image_cache:
            return self._vips_image_cache[key]

        try:
            # First, get the margins of the original SVG
            left_margin, top_margin, right_margin, bottom_margin = (
                self._get_margins()
            )

            # Now, create a modified SVG in memory that we will render
            # only once.
            root = ET.fromstring(self.raw_data)
            viewbox_str = root.get("viewBox")
            if viewbox_str:
                viewbox_x, viewbox_y, viewbox_width, viewbox_height = map(
                    float, viewbox_str.split()
                )
            else:
                w_str, h_str = root.get("width"), root.get("height")
                if not w_str or not h_str:
                    return None
                w_val, _ = parse_length(w_str)
                h_val, _ = parse_length(h_str)
                viewbox_x, viewbox_y, viewbox_width, viewbox_height = (
                    0,
                    0,
                    w_val,
                    h_val,
                )

            # Adjust the viewBox to crop out the empty margins
            new_viewbox_x = viewbox_x + (left_margin * viewbox_width)
            new_viewbox_y = viewbox_y + (top_margin * viewbox_height)
            new_viewbox_width = viewbox_width * (
                1 - left_margin - right_margin
            )
            new_viewbox_height = viewbox_height * (
                1 - top_margin - bottom_margin
            )

            root.set(
                "viewBox",
                f"{new_viewbox_x} {new_viewbox_y} "
                f"{new_viewbox_width} {new_viewbox_height}",
            )
            root.set("width", f"{width}px")
            root.set("height", f"{height}px")
            root.set("preserveAspectRatio", "none")

            # Perform the render operation
            final_svg_data = ET.tostring(root, encoding="utf-8")
            final_image = pyvips.Image.svgload_buffer(final_svg_data)

            self._vips_image_cache[key] = final_image
            return final_image
        except (pyvips.Error, ET.ParseError) as e:
            logger.error(f"Final SVG render failed: {e}")
            return None

    def render_to_pixels(
        self, width: int, height: int
    ) -> Optional[cairo.ImageSurface]:
        final_image = self._render_to_vips_image(width, height)
        render_width, render_height = width, height

        if render_width <= 0 or render_height <= 0:
            return None

        # If the requested render size exceeds Cairo's hard limit, we must
        # scale it down to prevent a crash. The UI layer will scale the
        # resulting (smaller) surface back up, resulting in pixelation,
        # which is an acceptable trade-off at extreme zoom levels.
        if (
            render_width > CAIRO_MAX_DIMENSION
            or render_height > CAIRO_MAX_DIMENSION
        ):
            scale_factor = 1.0
            if render_width > CAIRO_MAX_DIMENSION:
                scale_factor = CAIRO_MAX_DIMENSION / render_width
            if render_height > CAIRO_MAX_DIMENSION:
                scale_factor = min(
                    scale_factor, CAIRO_MAX_DIMENSION / render_height
                )

            new_width = int(render_width * scale_factor)
            new_height = int(render_height * scale_factor)

            render_width = max(1, new_width)
            render_height = max(1, new_height)

            final_image = self._render_to_vips_image(
                render_width, render_height
            )

        if not isinstance(final_image, pyvips.Image):
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
