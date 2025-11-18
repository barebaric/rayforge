import cairo
import io
import warnings
import logging
from typing import Optional, Tuple, TYPE_CHECKING, Dict, Any

from pypdf import PdfReader

with warnings.catch_warnings():
    warnings.simplefilter("ignore", DeprecationWarning)
    import pyvips

from ...core.workpiece import WorkPiece
from ..base_renderer import Renderer
from .. import image_util
from ..util import to_mm

if TYPE_CHECKING:
    from ...core.source_asset_segment import SourceAssetSegment
    from ...core.geo import Geometry
    from ...core.matrix import Matrix

logger = logging.getLogger(__name__)


class PdfRenderer(Renderer):
    """Renders PDF data from a WorkPiece."""

    def render_data_to_vips_image(
        self, data: bytes, dpi: int
    ) -> Optional[pyvips.Image]:
        """Renders raw PDF data to a pyvips Image at a specific DPI."""
        if not data:
            return None
        try:
            image = pyvips.Image.pdfload_buffer(data, dpi=dpi)
            if not isinstance(image, pyvips.Image) or image.width == 0:
                return None
            return image
        except Exception:
            logger.warning(
                "Failed to render PDF data to vips image.", exc_info=True
            )
            return None

    def _render_to_vips_internal(
        self,
        *,
        data_to_render: Optional[bytes],
        source_w_px: Optional[int],
        source_segment: Optional["SourceAssetSegment"],
        natural_size_mm: Optional[Tuple[float, float]],
        width: int,
        height: int,
    ) -> Optional[pyvips.Image]:
        if not data_to_render:
            return None

        # Determine DPI for the initial full-page render.
        if source_w_px and natural_size_mm and natural_size_mm[0] > 0:
            dpi = int((source_w_px / natural_size_mm[0]) * 25.4)
        else:
            dpi = 300

        full_image = self.render_data_to_vips_image(data_to_render, dpi=dpi)
        if not full_image:
            return None

        image_to_process = full_image
        if source_segment:
            if crop := source_segment.crop_window_px:
                x, y, w, h = map(int, crop)
                image_to_process = image_util.safe_crop(full_image, x, y, w, h)
                if image_to_process is None:
                    # If there's no intersection, the result is an empty image
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
        # Prioritize workpiece.data for transient objects (previewer),
        # fall back to original_data for fully loaded workpieces.
        data_to_render = workpiece.data or workpiece.original_data
        source = workpiece.source
        source_w_px = source.width_px if source else None
        natural_size_mm = self.get_natural_size(workpiece)

        return self._render_to_vips_internal(
            data_to_render=data_to_render,
            source_w_px=source_w_px,
            source_segment=workpiece.source_segment,
            natural_size_mm=natural_size_mm,
            width=width,
            height=height,
        )

    def _get_natural_size_internal(
        self, *, data_to_read: Optional[bytes]
    ) -> Optional[Tuple[float, float]]:
        if not data_to_read:
            return None
        try:
            reader = PdfReader(io.BytesIO(data_to_read))
            media_box = reader.pages[0].mediabox
            return (
                to_mm(float(media_box.width), "pt"),
                to_mm(float(media_box.height), "pt"),
            )
        except Exception:
            logger.warning(
                "Failed to get natural size from PDF data.", exc_info=True
            )
            return None

    def get_natural_size(
        self, workpiece: "WorkPiece"
    ) -> Optional[Tuple[float, float]]:
        """
        Calculates the natural size of a PDF from its data. The "natural
        size" for a PDF source is always the full page size, which is needed
        to calculate the correct DPI for rendering.
        """
        # Prioritize workpiece.data, as it's set for transient objects like
        # in the import previewer. Fall back to original_data for fully
        # loaded workpieces.
        data_to_read = workpiece.data or workpiece.original_data
        return self._get_natural_size_internal(data_to_read=data_to_read)

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
        return self._get_natural_size_internal(data_to_read=render_data)

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
        data_to_render = render_data or original_data
        source_w_px = source_px_dims[0] if source_px_dims else None
        natural_size_mm = self._get_natural_size_internal(
            data_to_read=data_to_render
        )

        return self._render_to_vips_internal(
            data_to_render=data_to_render,
            source_w_px=source_w_px,
            source_segment=source_segment,
            natural_size_mm=natural_size_mm,
            width=width,
            height=height,
        )


PDF_RENDERER = PdfRenderer()
