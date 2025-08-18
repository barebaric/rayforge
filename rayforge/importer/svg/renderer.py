import cairo
import warnings
from typing import Optional, Tuple
from xml.etree import ElementTree as ET

with warnings.catch_warnings():
    warnings.simplefilter("ignore", DeprecationWarning)
    import pyvips

from ...core.workpiece import WorkPiece
from ..base_renderer import Renderer
from ..shared.util import parse_length, to_mm


class SvgRenderer(Renderer):
    """Renders SVG data from a WorkPiece."""

    def _get_margins(
        self, workpiece: "WorkPiece"
    ) -> Tuple[float, float, float, float]:
        # Margin calculation logic moved here from old importer
        measurement_size = 1000.0
        if not workpiece.data:
            return 0.0, 0.0, 0.0, 0.0
        try:
            root = ET.fromstring(workpiece.data)
            root.set("width", f"{measurement_size}px")
            root.set("height", f"{measurement_size}px")
            root.set("preserveAspectRatio", "none")

            img = pyvips.Image.svgload_buffer(ET.tostring(root))
            if img.bands < 4:
                img = img.bandjoin(255)

            left, top, w, h = img.find_trim()
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

    def get_natural_size(
        self, workpiece: "WorkPiece", px_factor: float = 0.0
    ) -> Optional[Tuple[float, float]]:
        if not workpiece.data:
            return None
        try:
            root = ET.fromstring(workpiece.data)
            w_str = root.get("width")
            h_str = root.get("height")
            if not w_str or not h_str:
                return None

            w_val, w_unit = parse_length(w_str)
            h_val, h_unit = parse_length(h_str)

            w_mm = to_mm(w_val, w_unit, px_factor=px_factor)
            h_mm = to_mm(h_val, h_unit, px_factor=px_factor)

        except (ValueError, ET.ParseError):
            return None

        left, top, right, bottom = self._get_margins(workpiece)
        return w_mm * (1 - left - right), h_mm * (1 - top - bottom)

    def _render_to_vips_image(
        self, workpiece: "WorkPiece", width: int, height: int
    ) -> Optional[pyvips.Image]:
        if not workpiece.data:
            return None
        try:
            left, top, right, bottom = self._get_margins(workpiece)
            root = ET.fromstring(workpiece.data)

            vb_str = root.get("viewBox")
            if vb_str:
                vb_x, vb_y, vb_w, vb_h = map(float, vb_str.split())
            else:
                w_str, h_str = root.get("width"), root.get("height")
                if not w_str or not h_str:
                    return None
                w_val, _ = parse_length(w_str)
                h_val, _ = parse_length(h_str)
                vb_x, vb_y, vb_w, vb_h = 0, 0, w_val, h_val

            new_vb_x = vb_x + (left * vb_w)
            new_vb_y = vb_y + (top * vb_h)
            new_vb_w = vb_w * (1 - left - right)
            new_vb_h = vb_h * (1 - top - bottom)
            root.set("viewBox", f"{new_vb_x} {new_vb_y} {new_vb_w} {new_vb_h}")

            root.set("width", f"{width}px")
            root.set("height", f"{height}px")
            root.set("preserveAspectRatio", "none")

            return pyvips.Image.svgload_buffer(ET.tostring(root))
        except (pyvips.Error, ET.ParseError, ValueError, TypeError):
            return None

    def render_to_pixels(
        self, workpiece: "WorkPiece", width: int, height: int
    ) -> Optional[cairo.ImageSurface]:
        final_image = self.get_or_create_vips_image(workpiece, width, height)
        if not final_image:
            return None

        if final_image.bands < 4:
            final_image = final_image.bandjoin(255)

        # Vips RGBA -> Cairo BGRA
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


SVG_RENDERER = SvgRenderer()
