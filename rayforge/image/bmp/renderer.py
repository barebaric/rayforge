import cairo
import warnings
from typing import Optional, Tuple, TYPE_CHECKING, Dict, Any

from .parser import parse_bmp

with warnings.catch_warnings():
    warnings.simplefilter("ignore", DeprecationWarning)
    try:
        import pyvips
    except ImportError:
        raise ImportError("The BMP renderer requires the pyvips library.")

from ...core.workpiece import WorkPiece
from ..base_renderer import Renderer
from .. import image_util

if TYPE_CHECKING:
    from ...core.source_asset_segment import SourceAssetSegment
    from ...core.geo import Geometry
    from ...core.matrix import Matrix


class BmpRenderer(Renderer):
    """Renders BMP data from a WorkPiece."""

    def _get_natural_size_internal(
        self,
        *,
        render_data: Optional[bytes],
        source_segment: Optional["SourceAssetSegment"],
    ) -> Optional[Tuple[float, float]]:
        if source_segment:
            w = source_segment.cropped_width_mm
            h = source_segment.cropped_height_mm
            if w is not None and h is not None:
                return w, h
        if not render_data:
            return None
        parsed_data = parse_bmp(render_data)
        if not parsed_data:
            return None
        _, width, height, dpi_x, dpi_y = parsed_data
        dpi_x = dpi_x or 96.0
        dpi_y = dpi_y or 96.0
        mm_width = width * (25.4 / dpi_x)
        mm_height = height * (25.4 / dpi_y)
        return mm_width, mm_height

    def get_natural_size(
        self, workpiece: "WorkPiece"
    ) -> Optional[Tuple[float, float]]:
        return self._get_natural_size_internal(
            render_data=workpiece.data,
            source_segment=workpiece.source_segment,
        )

    def _render_to_vips_internal(
        self,
        *,
        render_data: Optional[bytes],
        source_segment: Optional["SourceAssetSegment"],
        width: int,
        height: int,
    ) -> Optional[pyvips.Image]:
        if not render_data:
            return None
        parsed_data = parse_bmp(render_data)
        if not parsed_data:
            return None
        rgba_bytes, img_width, img_height, _, _ = parsed_data
        try:
            full_image = pyvips.Image.new_from_memory(
                rgba_bytes, img_width, img_height, 4, "uchar"
            )
        except pyvips.Error:
            return None

        image_to_process = full_image
        if source_segment:
            if crop := source_segment.crop_window_px:
                x, y, w, h = map(int, crop)
                image_to_process = image_util.safe_crop(full_image, x, y, w, h)
                if image_to_process is None:
                    return pyvips.Image.black(width, height, bands=4)

            mask_geo = source_segment.segment_mask_geometry
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

    def _render_to_vips_image(
        self, workpiece: "WorkPiece", width: int, height: int
    ) -> Optional[pyvips.Image]:
        return self._render_to_vips_internal(
            render_data=workpiece.data,
            source_segment=workpiece.source_segment,
            width=width,
            height=height,
        )

    def render_to_pixels(
        self, workpiece: "WorkPiece", width: int, height: int
    ) -> Optional[cairo.ImageSurface]:
        final_image = self.get_or_create_vips_image(workpiece, width, height)
        if not final_image:
            return None
        normalized_image = image_util.normalize_to_rgba(final_image)
        if not normalized_image:
            return None
        return image_util.vips_rgba_to_cairo_surface(normalized_image)

    def get_natural_size_from_data(
        self,
        *,
        render_data: Optional[bytes],
        source_segment: Optional["SourceAssetSegment"],
        source_metadata: Optional[Dict[str, Any]],
        boundaries: Optional["Geometry"] = None,
        current_size: Optional[Tuple[float, float]] = None,
    ) -> Optional[Tuple[float, float]]:
        return self._get_natural_size_internal(
            render_data=render_data, source_segment=source_segment
        )

    def render_from_data(
        self,
        *,
        render_data: Optional[bytes],
        original_data: Optional[bytes] = None,
        source_segment: Optional["SourceAssetSegment"] = None,
        source_px_dims: Optional[Tuple[int, int]] = None,
        source_metadata: Optional[Dict[str, Any]] = None,
        boundaries: Optional["Geometry"] = None,
        workpiece_matrix: Optional["Matrix"] = None,
        width: int,
        height: int,
    ) -> Optional[pyvips.Image]:
        return self._render_to_vips_internal(
            render_data=render_data,
            source_segment=source_segment,
            width=width,
            height=height,
        )


BMP_RENDERER = BmpRenderer()
