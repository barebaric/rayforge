import io
import warnings
import logging
from typing import Optional, Tuple, TYPE_CHECKING
from pypdf import PdfReader
from pypdf.errors import PdfReadError
from ..base_renderer import RasterRenderer, RenderSpecification

with warnings.catch_warnings():
    warnings.simplefilter("ignore", DeprecationWarning)
    import pyvips

if TYPE_CHECKING:
    from ...core.source_asset_segment import SourceAssetSegment
    from ...core.workpiece import RenderContext
    from ...image.structures import ImportResult

logger = logging.getLogger(__name__)


class PdfRenderer(RasterRenderer):
    """Renders PDF data."""

    def compute_render_spec(
        self,
        segment: Optional["SourceAssetSegment"],
        target_size: Tuple[int, int],
        source_context: "RenderContext",
    ) -> "RenderSpecification":
        target_width, target_height = target_size
        original_data = source_context.original_data
        source_px_dims = source_context.source_pixel_dims

        if (
            segment
            and segment.crop_window_px is not None
            and original_data
            and source_px_dims
        ):
            source_w, source_h = source_px_dims
            crop_x_f, crop_y_f, crop_w_f, crop_h_f = segment.crop_window_px
            crop_w, crop_h = float(crop_w_f), float(crop_h_f)

            if crop_w > 0 and crop_h > 0:
                scale_x = target_width / crop_w
                scale_y = target_height / crop_h
                render_width = max(1, int(source_w * scale_x))
                render_height = max(1, int(source_h * scale_y))

                scaled_x = int(crop_x_f * scale_x)
                scaled_y = int(crop_y_f * scale_y)
                scaled_w = int(crop_w * scale_x)
                scaled_h = int(crop_h * scale_y)
                crop_rect = (scaled_x, scaled_y, scaled_w, scaled_h)

                return RenderSpecification(
                    width=render_width,
                    height=render_height,
                    data=original_data,
                    crop_rect=crop_rect,
                    apply_mask=False,
                )

        return RenderSpecification(
            width=target_width,
            height=target_height,
            data=source_context.data,
            apply_mask=False,
        )

    def render_preview_image(
        self,
        import_result: "ImportResult",
        target_width: int,
        target_height: int,
    ) -> Optional[pyvips.Image]:
        """
        Generates a preview image from a PDF import.

        This method has special handling for PDFs. The PdfImporter pre-renders
        the PDF to a PNG and stores it in `base_render_data` as an
        optimization. This method will use that cached PNG if available,
        bypassing the expensive PDF rendering.
        """
        if not import_result.payload:
            return None

        source = import_result.payload.source
        if source.base_render_data:
            try:
                # The importer has cached a pre-rendered PNG. Load it directly.
                image = pyvips.Image.new_from_buffer(
                    source.base_render_data, ""
                )
                # Scale it to the final preview size.
                return image.thumbnail_image(
                    target_width, height=target_height, size="both"
                )
            except pyvips.Error as e:
                logger.warning(
                    f"Failed to load cached preview image from "
                    f"base_render_data: {e}"
                )
                # Fall through to rendering the original PDF.

        # Fallback: If no cached data, render the original PDF from scratch.
        return super().render_preview_image(
            import_result, target_width, target_height
        )

    def _get_page_points(self, data: bytes) -> Optional[Tuple[float, float]]:
        try:
            reader = PdfReader(io.BytesIO(data))
            mb = reader.pages[0].mediabox
            w, h = float(mb.width), float(mb.height)
            if w > 0 and h > 0:
                return w, h
        except (PdfReadError, IndexError, AttributeError, ValueError) as e:
            logger.warning(
                "Failed to read PDF page dimensions via pypdf: %s", e
            )

        try:
            probe = pyvips.Image.pdfload_buffer(data, dpi=72)
            w, h = float(probe.width), float(probe.height)
            if w > 0 and h > 0:
                return w, h
        except pyvips.Error as e:
            logger.warning(
                "Failed to read PDF page dimensions via pyvips: %s", e
            )

        return None

    def render_base_image(
        self,
        data: bytes,
        width: int,
        height: int,
        **kwargs,
    ) -> Optional[pyvips.Image]:
        if not data:
            return None

        # Check if data is pre-rendered PNG (from trace import)
        if data[:4] == b"\x89PNG":
            try:
                image = pyvips.Image.new_from_buffer(data, "")
                return image.thumbnail_image(width, height=height, size="both")
            except pyvips.Error as e:
                logger.warning(f"Failed to load PNG data: {e}")
                return None

        # For PDFs, we must determine a DPI to request from the loader
        # to achieve the target pixel dimensions.
        page_pts = self._get_page_points(data)
        if page_pts and width > 0 and height > 0:
            w_pt, h_pt = page_pts
            dpi = max((width / w_pt) * 72.0, (height / h_pt) * 72.0)
        else:
            dpi = 300.0

        try:
            image = pyvips.Image.pdfload_buffer(data, dpi=dpi)
            if not isinstance(image, pyvips.Image) or image.width == 0:
                return None
            return image
        except pyvips.Error:
            logger.warning(
                "Failed to render PDF data to vips image.", exc_info=True
            )
            return None


PDF_RENDERER = PdfRenderer()
