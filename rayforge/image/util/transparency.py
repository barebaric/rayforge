"""
Transparency manipulation utilities for Cairo surfaces.
"""

import cairo
import numpy


def make_surface_transparent(
    surface: cairo.ImageSurface, threshold: int = 250
) -> None:
    """
    Make "almost white" pixels transparent in a Cairo ARGB32 surface.

    Modifies the surface in place. Pixels with average brightness above
    the threshold have their alpha channel set to 0.

    Args:
        surface: Cairo ImageSurface in FORMAT_ARGB32 format.
        threshold: Brightness threshold (0-255). Pixels with average
            RGB value >= threshold become transparent.

    Raises:
        ValueError: If the surface format is not ARGB32.
    """
    if surface.get_format() != cairo.FORMAT_ARGB32:
        raise ValueError("Surface must be in ARGB32 format.")

    width, height = surface.get_width(), surface.get_height()
    stride = surface.get_stride()

    data = surface.get_data()
    buf = numpy.frombuffer(data, dtype=numpy.uint8).reshape((height, stride))

    argb = buf.view(dtype=numpy.uint32)[:, :width]

    r = (argb >> 16) & 0xFF
    g = (argb >> 8) & 0xFF
    b = argb & 0xFF

    brightness = (
        r.astype(numpy.uint16)
        + g.astype(numpy.uint16)
        + b.astype(numpy.uint16)
    ) // 3
    mask = brightness >= threshold

    argb[mask] = (0x00 << 24) | (r[mask] << 16) | (g[mask] << 8) | b[mask]


def make_transparent_except_color(
    surface: cairo.ImageSurface,
    target_r: int,
    target_g: int,
    target_b: int,
) -> None:
    """
    Make all pixels transparent except those matching a target RGB color.

    Modifies the surface in place. Pixels that do not match the target
    color have their alpha channel set to 0.

    Args:
        surface: Cairo ImageSurface in FORMAT_ARGB32 format.
        target_r: Target red channel value (0-255).
        target_g: Target green channel value (0-255).
        target_b: Target blue channel value (0-255).

    Raises:
        ValueError: If the surface format is not ARGB32.
    """
    if surface.get_format() != cairo.FORMAT_ARGB32:
        raise ValueError("Surface must be in ARGB32 format.")

    width, height = surface.get_width(), surface.get_height()
    stride = surface.get_stride()

    data = surface.get_data()
    buf = numpy.frombuffer(data, dtype=numpy.uint8).reshape((height, stride))

    argb = buf.view(dtype=numpy.uint32)[:, :width]

    r = (argb >> 16) & 0xFF
    g = (argb >> 8) & 0xFF
    b = argb & 0xFF

    mask = ~((r == target_r) & (g == target_g) & (b == target_b))

    argb[mask] = (0x00 << 24) | (r[mask] << 16) | (g[mask] << 8) | b[mask]
