"""
Fallback SVG rendering using Cairo/Rsvg when libvips lacks SVG support.

This module provides capability detection for libvips' svgload_buffer and
a Cairo-based fallback renderer using PyGObject's Rsvg binding.
"""

import logging
from typing import Optional, TYPE_CHECKING
import warnings

with warnings.catch_warnings():
    warnings.simplefilter("ignore", DeprecationWarning)
    import pyvips

if TYPE_CHECKING:
    import cairo

logger = logging.getLogger(__name__)

_SVG_LOAD_AVAILABLE: Optional[bool] = None


def _check_svg_load_capability() -> bool:
    """
    Tests whether pyvips.Image.svgload_buffer is available.

    Some libvips installations are compiled without librsvg support,
    causing svgload_buffer to fail. This function tests with a minimal
    SVG to determine availability.

    Returns:
        True if svgload_buffer works, False otherwise.
    """
    minimal_svg = b"""<?xml version="1.0" encoding="UTF-8"?>
<svg xmlns="http://www.w3.org/2000/svg" width="1" height="1">
  <rect width="1" height="1" fill="black"/>
</svg>"""

    try:
        img = pyvips.Image.svgload_buffer(minimal_svg)
        if img.width == 1 and img.height == 1:
            return True
    except pyvips.Error:
        pass
    except Exception:
        pass

    return False


SVG_LOAD_AVAILABLE = _check_svg_load_capability()
"""Boolean indicating whether pyvips.Image.svgload_buffer is available."""


def render_svg_to_cairo(
    svg_data: bytes, width: int, height: int
) -> Optional["cairo.ImageSurface"]:
    """
    Renders SVG data to a Cairo ImageSurface using PyGObject's Rsvg.

    Args:
        svg_data: Raw SVG bytes.
        width: Target width in pixels.
        height: Target height in pixels.

    Returns:
        A cairo.ImageSurface with the rendered SVG, or None on failure.
    """
    import cairo
    import gi

    gi.require_version("Rsvg", "2.0")
    from gi.repository import Rsvg

    if not svg_data:
        return None

    try:
        handle = Rsvg.Handle.new_from_data(svg_data)
        if handle is None:
            logger.error("Failed to create Rsvg.Handle from SVG data")
            return None

        surface = cairo.ImageSurface(cairo.FORMAT_ARGB32, width, height)
        ctx = cairo.Context(surface)

        dimensions = handle.get_dimensions()
        doc_width = dimensions.width
        doc_height = dimensions.height

        if doc_width > 0 and doc_height > 0:
            scale_x = width / doc_width
            scale_y = height / doc_height
            ctx.scale(scale_x, scale_y)

        viewport = Rsvg.Rectangle()
        viewport.x = 0
        viewport.y = 0
        viewport.width = width
        viewport.height = height
        handle.render_document(ctx, viewport)
        return surface

    except Exception as e:
        logger.error(f"Failed to render SVG with Cairo/Rsvg: {e}")
        return None


def cairo_surface_to_vips(
    surface: "cairo.ImageSurface",
) -> Optional[pyvips.Image]:
    """
    Converts a Cairo ImageSurface to a pyvips Image.

    This function handles the BGRA to RGBA conversion that Cairo uses
    internally.

    Args:
        surface: A cairo.ImageSurface in ARGB32 format.

    Returns:
        A pyvips.Image in RGBA format, or None on failure.
    """
    if not surface:
        return None

    try:
        h = surface.get_height()
        w = surface.get_width()

        vips_image = pyvips.Image.new_from_memory(
            surface.get_data(), w, h, 4, "uchar"
        )

        b = vips_image[0]
        g = vips_image[1]
        r = vips_image[2]
        a = vips_image[3]

        return r.bandjoin([g, b, a])

    except Exception as e:
        logger.error(f"Failed to convert Cairo surface to pyvips: {e}")
        return None


def load_svg_with_fallback(
    svg_data: bytes, width: Optional[int] = None, height: Optional[int] = None
) -> Optional[pyvips.Image]:
    """
    Loads SVG data using either libvips or Cairo fallback.

    This is a convenience function that automatically selects the appropriate
    rendering path based on available capabilities.

    Args:
        svg_data: Raw SVG bytes.
        width: Target width (required for Cairo fallback).
        height: Target height (required for Cairo fallback).

    Returns:
        A pyvips.Image, or None on failure.
    """
    if SVG_LOAD_AVAILABLE:
        try:
            return pyvips.Image.svgload_buffer(svg_data)
        except pyvips.Error as e:
            logger.error(f"Failed to load SVG with pyvips: {e}")
            return None
    else:
        if width is None or height is None:
            logger.error("Cairo fallback requires width and height parameters")
            return None

        surface = render_svg_to_cairo(svg_data, width, height)
        if surface:
            return cairo_surface_to_vips(surface)
        return None


if not SVG_LOAD_AVAILABLE:
    logger.warning(
        "pyvips svgload_buffer not available. "
        "Using Cairo/Rsvg fallback for SVG rendering."
    )
