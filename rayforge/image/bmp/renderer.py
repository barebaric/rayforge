import warnings
from typing import Optional, TYPE_CHECKING
from ..base_renderer import Renderer
from .parser import parse_bmp

with warnings.catch_warnings():
    warnings.simplefilter("ignore", DeprecationWarning)
    import pyvips
if TYPE_CHECKING:
    pass


class BmpRenderer(Renderer):
    """Renders BMP data."""

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
