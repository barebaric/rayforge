import warnings
from typing import Optional, TYPE_CHECKING
from ..base_renderer import Renderer

with warnings.catch_warnings():
    warnings.simplefilter("ignore", DeprecationWarning)
    import pyvips

if TYPE_CHECKING:
    pass


class JpgRenderer(Renderer):
    """Renders JPEG data."""

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
            return pyvips.Image.jpegload_buffer(data)
        except pyvips.Error:
            return None


JPG_RENDERER = JpgRenderer()
