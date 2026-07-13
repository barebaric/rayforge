"""
Grayscale and binary image conversion utilities for Cairo surfaces.
"""

import cairo
import numpy as np
from raygeo.image.convert import (
    rgba_to_binary,
    rgba_to_grayscale,
    rgba_to_grayscale_inplace,
)


def _extract_rgba(surface: cairo.ImageSurface) -> tuple:
    width = surface.get_width()
    height = surface.get_height()
    stride_px = surface.get_stride() // 4
    buf = np.frombuffer(surface.get_data(), dtype=np.uint8).copy()
    return buf, width, height, stride_px


def surface_to_grayscale(
    surface: cairo.ImageSurface,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Convert a Cairo ARGB32 surface to a grayscale array with alpha handling.

    Args:
        surface: Cairo ImageSurface in FORMAT_ARGB32 format.

    Returns:
        Tuple of (grayscale_array, alpha_array) as numpy arrays.
        grayscale_array: uint8 array with values 0-255.
        alpha_array: float32 array with values 0.0-1.0.
    """
    buf, width, height, stride = _extract_rgba(surface)
    return rgba_to_grayscale(buf, width, height, stride)


def surface_to_binary(
    surface: cairo.ImageSurface,
    threshold: int = 128,
    invert: bool = False,
) -> np.ndarray:
    """
    Convert a Cairo ARGB32 surface to a binary array using thresholding.

    Transparent pixels are always treated as white (0).

    Args:
        surface: Cairo ImageSurface in FORMAT_ARGB32 format.
        threshold: Brightness value (0-255) for binarization.
        invert: If True, pixels above threshold become black (1).

    Returns:
        2D numpy array with values 0 (white/transparent) or 1 (black).

    Raises:
        ValueError: If the surface format is not ARGB32.
    """
    if surface.get_format() != cairo.FORMAT_ARGB32:
        raise ValueError("Unsupported Cairo surface format")
    buf, width, height, stride = _extract_rgba(surface)
    return rgba_to_binary(buf, width, height, stride, threshold, invert)


def convert_surface_to_grayscale_inplace(
    surface: cairo.ImageSurface,
) -> None:
    """
    Convert a Cairo ARGB32 surface to grayscale in place.

    Args:
        surface: Cairo ImageSurface in FORMAT_ARGB32 format.

    Raises:
        ValueError: If the surface format is not ARGB32.
    """
    if surface.get_format() != cairo.FORMAT_ARGB32:
        raise ValueError("Unsupported Cairo surface format")
    width = surface.get_width()
    height = surface.get_height()
    stride_px = surface.get_stride() // 4
    buf = np.frombuffer(surface.get_data(), dtype=np.uint8)
    rgba_to_grayscale_inplace(buf, width, height, stride_px)


def get_visible_grayscale_values(
    surface: cairo.ImageSurface,
    invert: bool = False,
) -> np.ndarray:
    """
    Extract grayscale values for visible pixels from a Cairo ARGB32 surface.

    Args:
        surface: Cairo ImageSurface in FORMAT_ARGB32 format.
        invert: If True, invert grayscale values for visible pixels.

    Returns:
        1D uint8 numpy array of grayscale values for pixels with alpha > 0.
    """
    gray_image, alpha = surface_to_grayscale(surface)
    if invert:
        alpha_mask = alpha > 0
        gray_image[alpha_mask] = 255 - gray_image[alpha_mask]
    return gray_image[alpha > 0]
