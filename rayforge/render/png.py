from typing import Optional, Tuple
import cairo
import logging
import warnings
with warnings.catch_warnings():
    warnings.simplefilter("ignore", DeprecationWarning)
    import pyvips
from .renderer import Renderer

logger = logging.getLogger(__name__)


class PNGRenderer(Renderer):
    label = 'PNG files'
    mime_types = ('image/png',)
    extensions = ('.png',)

    def __init__(self, data: bytes):
        image = pyvips.Image.pngload_buffer(
            data, access=pyvips.Access.RANDOM
        )
        if image.bands == 3:
            image = image.bandjoin(255)
        self.vips_image = image

    def get_natural_size(
        self, px_factor: float = 0.0
    ) -> Tuple[Optional[float], Optional[float]]:
        try:
            xres = self.vips_image.get('xres')
        except pyvips.error.Error:
            xres = 5.0

        try:
            yres = self.vips_image.get('yres')
        except pyvips.error.Error:
            yres = 5.0

        mm_width = self.vips_image.width / xres if xres > 0 else None
        mm_height = self.vips_image.height / yres if yres > 0 else None
        return mm_width, mm_height

    def get_aspect_ratio(self) -> float:
        if self.vips_image.height == 0:
            return 1.0
        return self.vips_image.width / self.vips_image.height

    def _render_to_vips_image(
        self, width: int, height: int
    ) -> pyvips.Image:
        h_scale = width / self.vips_image.width
        v_scale = height / self.vips_image.height
        return self.vips_image.resize(h_scale, vscale=v_scale)

    def render_to_pixels(
        self, width: int, height: int
    ) -> Optional[cairo.ImageSurface]:
        final_image = self._render_to_vips_image(width, height)
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
            final_image.width * 4
        )
