import warnings
from typing import Optional, Tuple, TYPE_CHECKING, Dict, Any
from ..base_renderer import Renderer
from .parser import parse_bmp

with warnings.catch_warnings():
    warnings.simplefilter("ignore", DeprecationWarning)
    import pyvips
if TYPE_CHECKING:
    from ...core.source_asset_segment import SourceAssetSegment
    from ...core.geo import Geometry


class BmpRenderer(Renderer):
    """Renders BMP data."""

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
        parsed_data = parse_bmp(render_data)
        if not parsed_data:
            return None
        _, width, height, dpi_x, dpi_y = parsed_data
        dpi_x = dpi_x or 96.0
        dpi_y = dpi_y or 96.0
        mm_width = width * (25.4 / dpi_x)
        mm_height = height * (25.4 / dpi_y)
        return mm_width, mm_height

    def render_base_image(
        self,
        data: bytes,
        width: int,
        height: int,
        **kwargs,
    ) -> Optional[pyvips.Image]:
        if not data:
            return None
        parsed_data = parse_bmp(data)
        if not parsed_data:
            return None
        rgba_bytes, img_width, img_height, _, _ = parsed_data
        try:
            return pyvips.Image.new_from_memory(
                rgba_bytes, img_width, img_height, 4, "uchar"
            )
        except pyvips.Error:
            return None


BMP_RENDERER = BmpRenderer()
