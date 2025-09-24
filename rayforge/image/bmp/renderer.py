import cairo
import warnings
from typing import Optional, Tuple
from .parser import parse_bmp

with warnings.catch_warnings():
    warnings.simplefilter("ignore", DeprecationWarning)
    try:
        import pyvips
    except ImportError:
        raise ImportError("The BMP renderer requires the pyvips library.")

from ...core.workpiece import WorkPiece
from ..base_renderer import Renderer


class BmpRenderer(Renderer):
    """Renders BMP data from a WorkPiece."""

    def get_natural_size(
        self, workpiece: "WorkPiece"
    ) -> Optional[Tuple[float, float]]:
        if not workpiece.data:
            return None

        parsed_data = parse_bmp(workpiece.data)
        if not parsed_data:
            return None

        _, width, height, dpi_x, dpi_y = parsed_data
        mm_width = width * (25.4 / dpi_x)
        mm_height = height * (25.4 / dpi_y)
        return mm_width, mm_height

    def _render_to_vips_image(
        self, workpiece: "WorkPiece", width: int, height: int
    ) -> Optional[pyvips.Image]:
        if not workpiece.data:
            return None

        parsed_data = parse_bmp(workpiece.data)
        if not parsed_data:
            return None

        rgba_bytes, img_width, img_height, _, _ = parsed_data

        try:
            image = pyvips.Image.new_from_memory(
                rgba_bytes, img_width, img_height, 4, "uchar"
            )
            h_scale = width / image.width
            v_scale = height / image.height
            return image.resize(h_scale, vscale=v_scale)
        except pyvips.Error:
            return None

    def render_to_pixels(
        self, workpiece: "WorkPiece", width: int, height: int
    ) -> Optional[cairo.ImageSurface]:
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
            bgra_image.width,
            bgra_image.height,
            bgra_image.width * 4,
        )


BMP_RENDERER = BmpRenderer()
