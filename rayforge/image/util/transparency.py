"""
Transparency manipulation utilities for Cairo surfaces.
"""

import cairo
import numpy
from raygeo.image.transparency import (
    make_transparent_by_brightness,
    make_transparent_except_color,
)


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
    stride_px = surface.get_stride() // 4

    data = surface.get_data()
    buf = numpy.frombuffer(data, dtype=numpy.uint8).copy()
    make_transparent_by_brightness(buf, width, height, stride_px, threshold)

    dst = numpy.frombuffer(data, dtype=numpy.uint8)
    dst[:] = buf
    surface.mark_dirty()


def make_transparent_except(
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
    stride_px = surface.get_stride() // 4

    data = surface.get_data()
    buf = numpy.frombuffer(data, dtype=numpy.uint8).copy()
    make_transparent_except_color(
        buf, width, height, stride_px, target_r, target_g, target_b
    )

    dst = numpy.frombuffer(data, dtype=numpy.uint8)
    dst[:] = buf
    surface.mark_dirty()
