import warnings

from ..base_renderer import RasterRenderer

with warnings.catch_warnings():
    warnings.simplefilter("ignore", DeprecationWarning)
    import pyvips


class JpgRenderer(RasterRenderer):
    """Renders JPEG data."""

    def render_base_image(
        self,
        data: bytes,
        width: int,
        height: int,
        **kwargs,
    ) -> pyvips.Image | None:
        if not data:
            return None
        try:
            return pyvips.Image.jpegload_buffer(data)
        except pyvips.Error:
            return None


JPG_RENDERER = JpgRenderer()
