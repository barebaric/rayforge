import cairo
import warnings
from typing import Optional, Tuple

with warnings.catch_warnings():
    warnings.simplefilter("ignore", DeprecationWarning)
    import pyvips

from ...core.workpiece import WorkPiece
from ..base_renderer import Renderer


class PngRenderer(Renderer):
    """Renders PNG data from a WorkPiece."""

    def get_natural_size(
        self, workpiece: "WorkPiece"
    ) -> Optional[Tuple[float, float]]:
        if not workpiece.data:
            return None
        try:
            image = pyvips.Image.pngload_buffer(
                workpiece.data, access=pyvips.Access.RANDOM
            )
        except pyvips.Error:
            return None

        try:
            # xres and yres are pixels per millimeter in VIPS for PNG
            xres = image.get("xres")
        except pyvips.Error:
            xres = 25.4 / 96.0  # Fallback to 96 DPI

        try:
            yres = image.get("yres")
        except pyvips.Error:
            yres = 25.4 / 96.0  # Fallback to 96 DPI

        mm_width = image.width / xres if xres > 0 else None
        mm_height = image.height / yres if yres > 0 else None

        if mm_width is None or mm_height is None:
            return None
        return mm_width, mm_height

    def _render_to_vips_image(
        self, workpiece: "WorkPiece", width: int, height: int
    ) -> Optional[pyvips.Image]:
        if not workpiece.data:
            return None
        try:
            image = pyvips.Image.pngload_buffer(
                workpiece.data, access=pyvips.Access.RANDOM
            )
            h_scale = width / image.width
            v_scale = height / image.height
            return image.resize(h_scale, vscale=v_scale)
        except pyvips.Error:
            return None

    def render_to_pixels(
        self, workpiece: "WorkPiece", width: int, height: int
    ) -> Optional[cairo.ImageSurface]:
        # FIX: Use the cache-aware helper from the base class.
        final_image = self.get_or_create_vips_image(workpiece, width, height)
        if not final_image:
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


PNG_RENDERER = PngRenderer()
