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
        if config := workpiece.source_segment:
            w = config.cropped_width_mm
            h = config.cropped_height_mm
            if w is not None and h is not None:
                return w, h
        if not workpiece.data:
            return None
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
        try:
            full_image = pyvips.Image.jpegload_buffer(workpiece.data)
        except pyvips.Error:
            return None
        if not full_image:
            return None

        image_to_process = full_image
        if config := workpiece.source_segment:
            if crop := config.crop_window_px:
                x, y, w, h = map(int, crop)
                image_to_process = image_util.safe_crop(full_image, x, y, w, h)
                if image_to_process is None:
                    return pyvips.Image.black(width, height, bands=4)

            mask_geo = config.segment_mask_geometry
            masked_image = image_util.apply_mask_to_vips_image(
                image_to_process, mask_geo
            )
            if masked_image:
                image_to_process = masked_image

        if image_to_process.width == 0 or image_to_process.height == 0:
            return image_to_process

        h_scale = width / image_to_process.width
        v_scale = height / image_to_process.height
        return image_to_process.resize(h_scale, vscale=v_scale)

    def render_to_pixels(
        self, workpiece: "WorkPiece", width: int, height: int
    ) -> Optional[cairo.ImageSurface]:
        resized_image = self.get_or_create_vips_image(workpiece, width, height)
        if not resized_image:
            return None
        normalized_image = image_util.normalize_to_rgba(resized_image)
        if not normalized_image:
            return None
        return image_util.vips_rgba_to_cairo_surface(normalized_image)


JPG_RENDERER = JpgRenderer()
