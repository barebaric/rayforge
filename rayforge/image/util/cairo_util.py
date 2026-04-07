import cairo
import numpy as np


def rgba_to_cairo_surface(rgba: np.ndarray) -> cairo.ImageSurface:
    """
    Convert an RGBA uint8 array to a premultiplied Cairo ARGB32 ImageSurface.

    Performs premultiplication using uint16 arithmetic and reorders
    channels from RGBA to BGRA (Cairo's native memory layout on
    little-endian systems).

    Args:
        rgba: A (h, w, 4) uint8 array in RGBA order with straight alpha.

    Returns:
        A cairo.ImageSurface in FORMAT_ARGB32 with premultiplied alpha.
    """
    h, w = rgba.shape[:2]
    a = rgba[..., 3].astype(np.uint16)
    bgra = np.empty((h, w, 4), dtype=np.uint8)
    bgra[..., 0] = (rgba[..., 2].astype(np.uint16) * a // 255).astype(np.uint8)
    bgra[..., 1] = (rgba[..., 1].astype(np.uint16) * a // 255).astype(np.uint8)
    bgra[..., 2] = (rgba[..., 0].astype(np.uint16) * a // 255).astype(np.uint8)
    bgra[..., 3] = rgba[..., 3]
    return cairo.ImageSurface.create_for_data(
        memoryview(bgra), cairo.FORMAT_ARGB32, w, h
    )
