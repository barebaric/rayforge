"""
Image utility functions split into logical submodules.

This module provides utilities for:
- sRGB <-> linear light conversion (srgb module)
- Grayscale and binary image conversion (grayscale module)
- Transparency manipulation (transparency module)
- PyVips image operations (vips module)
- Unit conversion and layout (unit module)
"""

from .cairo_util import rgba_to_cairo_surface
from .grayscale import (
    compute_auto_levels,
    convert_surface_to_grayscale_inplace,
    get_visible_grayscale_values,
    normalize_grayscale,
    surface_to_binary,
    surface_to_grayscale,
)
from .srgb import linear_to_srgb, resize_linear_nd, srgb_to_linear

from .transparency import (
    make_surface_transparent,
    make_transparent_except_color,
)
from .unit import (
    CAIRO_MAX_DIMENSION,
    calculate_chunk_layout,
    parse_length,
    to_mm,
)
from .vips import (
    apply_mask_to_vips_image,
    extract_vips_metadata,
    get_mm_per_pixel,
    get_physical_size_mm,
    normalize_to_rgba,
    resize_and_crop_from_full_image,
    resize_linear,
    safe_crop,
    vips_rgba_to_cairo_surface,
)

__all__ = [
    "CAIRO_MAX_DIMENSION",
    "apply_mask_to_vips_image",
    "calculate_chunk_layout",
    "compute_auto_levels",
    "convert_surface_to_grayscale_inplace",
    "extract_vips_metadata",
    "get_mm_per_pixel",
    "get_physical_size_mm",
    "get_visible_grayscale_values",
    "linear_to_srgb",
    "make_surface_transparent",
    "make_transparent_except_color",
    "normalize_grayscale",
    "normalize_to_rgba",
    "parse_length",
    "resize_and_crop_from_full_image",
    "resize_linear",
    "resize_linear_nd",
    "rgba_to_cairo_surface",
    "safe_crop",
    "srgb_to_linear",
    "surface_to_binary",
    "surface_to_grayscale",
    "to_mm",
    "vips_rgba_to_cairo_surface",
]
