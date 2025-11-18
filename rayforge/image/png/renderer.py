import warnings
from typing import Optional, Tuple, TYPE_CHECKING, Dict, Any
from ..base_renderer import Renderer
from .. import image_util

with warnings.catch_warnings():
    warnings.simplefilter("ignore", DeprecationWarning)
    import pyvips

if TYPE_CHECKING:
    from ...core.geo import Geometry
    from ...core.source_asset_segment import SourceAssetSegment


class PngRenderer(Renderer):
    """Renders PNG data."""

    def get_natural_size_from_data(
        self,
        *,
        render_data: Optional[bytes],
        source_segment: Optional["SourceAssetSegment"],
        source_metadata: Optional[Dict[str, Any]],
        boundaries: Optional["Geometry"] = None,
        current_size: Optional[Tuple[float, float]] = None,
    ) -> Optional[Tuple[float, float]]:
        if not render_data:
            return None
        try:
            image = pyvips.Image.pngload_buffer(
                render_data, access=pyvips.Access.RANDOM
            )
        except pyvips.Error:
            return None
        return image_util.get_physical_size_mm(image) if image else None

    def render_base_image(
        self,
        data: bytes,
        width: int,
        height: int,
        **kwargs,
    ) -> Optional[pyvips.Image]:
        if not data:
            return None
        try:
            return pyvips.Image.pngload_buffer(
                data, access=pyvips.Access.RANDOM
            )
        except pyvips.Error:
            return None


PNG_RENDERER = PngRenderer()
