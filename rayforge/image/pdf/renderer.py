import io
import warnings
import logging
from typing import Optional, Tuple, TYPE_CHECKING
from pypdf import PdfReader
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
        return RenderSpecification(
            width=target_size[0],
            height=target_size[1],
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
        try:
            reader = PdfReader(io.BytesIO(data))
            media_box = reader.pages[0].mediabox
            w_pt = float(media_box.width)
            h_pt = float(media_box.height)

            if w_pt > 0 and h_pt > 0 and width > 0 and height > 0:
                # Target DPI = (pixels / points) * 72
                dpi_x = (width / w_pt) * 72.0
                dpi_y = (height / h_pt) * 72.0
                dpi = max(dpi_x, dpi_y)
            else:
                dpi = 300.0
        except Exception:
            dpi = 300.0

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


PDF_RENDERER = PdfRenderer()
