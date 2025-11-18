import io
import warnings
import logging
from typing import Optional, Tuple, TYPE_CHECKING, Dict, Any
from pypdf import PdfReader
from ..base_renderer import Renderer
from ..util import to_mm

with warnings.catch_warnings():
    warnings.simplefilter("ignore", DeprecationWarning)
    import pyvips

if TYPE_CHECKING:
    from ...core.geo import Geometry
    from ...core.source_asset_segment import SourceAssetSegment

logger = logging.getLogger(__name__)


class PdfRenderer(Renderer):
    """Renders PDF data."""

    def get_natural_size_from_data(
        self,
        *,
        render_data: Optional[bytes],
        source_segment: Optional["SourceAssetSegment"],
        source_metadata: Optional[Dict[str, Any]],
        boundaries: Optional["Geometry"] = None,
        current_size: Optional[Tuple[float, float]] = None,
    ) -> Optional[Tuple[float, float]]:
        if source_segment:
            w = source_segment.cropped_width_mm
            h = source_segment.cropped_height_mm
            if w is not None and h is not None:
                return w, h
        if not render_data:
            return None
        try:
            reader = PdfReader(io.BytesIO(render_data))
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

    def render_base_image(
        self,
        data: bytes,
        width: int,
        height: int,
        **kwargs,
    ) -> Optional[pyvips.Image]:
        if not data:
            return None

        # Calculate DPI needed to achieve the requested pixel dimensions
        # given the PDF's natural size.
        natural_size_mm = self.get_natural_size_from_data(
            render_data=data,
            source_segment=None,
            source_metadata=None,
        )

        dpi = 300  # default fallback
        if (
            natural_size_mm
            and natural_size_mm[0] > 0
            and natural_size_mm[1] > 0
        ):
            # dpi = pixels / inches
            # inches = mm / 25.4
            dpi_x = (width / natural_size_mm[0]) * 25.4
            dpi_y = (height / natural_size_mm[1]) * 25.4

            # Use the maximum required DPI to ensure sufficient resolution
            # for non-uniform scaling (e.g. squashing one dimension).
            dpi = max(dpi_x, dpi_y)

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
