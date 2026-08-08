import warnings

from ..base_renderer import RasterRenderer

with warnings.catch_warnings():
    warnings.simplefilter("ignore", DeprecationWarning)
    import pyvips


class PngRenderer(RasterRenderer):
    """Renders PNG data."""

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
            return pyvips.Image.pngload_buffer(
                data, access=pyvips.Access.RANDOM
            )
        except pyvips.Error:
            return None


PNG_RENDERER = PngRenderer()
