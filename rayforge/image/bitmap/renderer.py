import warnings

from ..base_renderer import RasterRenderer

with warnings.catch_warnings():
    warnings.simplefilter("ignore", DeprecationWarning)
    import pyvips


def load_first_page(data: bytes) -> pyvips.Image:
    """
    Loads a bitmap of any format libvips detects by content. Animated
    and multi-page files yield their first page.

    Raises:
        pyvips.Error: If the data is not a readable image.
    """
    return pyvips.Image.new_from_buffer(data, "", access=pyvips.Access.RANDOM)


class BitmapRenderer(RasterRenderer):
    """Renders GIF, TIFF, WebP and other libvips-readable bitmaps."""

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
            return load_first_page(data)
        except pyvips.Error:
            return None


BITMAP_RENDERER = BitmapRenderer()
