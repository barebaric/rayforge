import cairo
import warnings
from typing import Optional, Tuple

with warnings.catch_warnings():
    warnings.simplefilter("ignore", DeprecationWarning)
    import pyvips

from ...core.workpiece import WorkPiece
from ..base_renderer import Renderer
from .. import image_util


class PngRenderer(Renderer):
    """Renders PNG data from a WorkPiece."""

    def get_natural_size(
        self, workpiece: "WorkPiece"
    ) -> Optional[Tuple[float, float]]:
        if not workpiece.data:
            return None
        image = image_util.load_vips_image_from_data(workpiece.data)
        return image_util.get_physical_size_mm(image) if image else None

    def _render_to_vips_image(
        self, workpiece: "WorkPiece", width: int, height: int
    ) -> Optional[pyvips.Image]:
        if not workpiece.data:
            return None

        image = image_util.load_vips_image_from_data(workpiece.data)
        if not image:
            return None

        if image.width == 0 or image.height == 0:
            return image

        h_scale = width / image.width
        v_scale = height / image.height
        return image.resize(h_scale, vscale=v_scale)

    def render_to_pixels(
        self, workpiece: "WorkPiece", width: int, height: int
    ) -> Optional[cairo.ImageSurface]:
        resized_image = self._render_to_vips_image(workpiece, width, height)
        if not resized_image:
            return None

        normalized_image = image_util.normalize_to_rgba(resized_image)
        if not normalized_image:
            return None

        return image_util.vips_rgba_to_cairo_surface(normalized_image)


PNG_RENDERER = PngRenderer()
