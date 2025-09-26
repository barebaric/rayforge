import cairo
import warnings
from typing import Optional, Tuple

with warnings.catch_warnings():
    warnings.simplefilter("ignore", DeprecationWarning)
    import pyvips

from ...core.workpiece import WorkPiece
from ..base_renderer import Renderer
from .. import image_util


class JpgRenderer(Renderer):
    """Renders JPEG data from a WorkPiece."""

    def get_natural_size(
        self, workpiece: "WorkPiece"
    ) -> Optional[Tuple[float, float]]:
        if not workpiece.data:
            return None
        # This utility function is format-agnostic
        try:
            image = pyvips.Image.jpegload_buffer(workpiece.data)
        except pyvips.Error:
            return None
        return image_util.get_physical_size_mm(image) if image else None

    def _render_to_vips_image(
        self, workpiece: "WorkPiece", width: int, height: int
    ) -> Optional[pyvips.Image]:
        if not workpiece.data:
            return None

        # This utility function is format-agnostic
        try:
            image = pyvips.Image.jpegload_buffer(workpiece.data)
        except pyvips.Error:
            return None
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

        # The rest of the rendering pipeline is also format-agnostic
        normalized_image = image_util.normalize_to_rgba(resized_image)
        if not normalized_image:
            return None

        return image_util.vips_rgba_to_cairo_surface(normalized_image)


# Create an instance for the importer to use
JPG_RENDERER = JpgRenderer()
