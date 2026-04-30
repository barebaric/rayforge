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
from .srgb import srgb_to_linear, linear_to_srgb
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
    safe_crop,
    vips_rgba_to_cairo_surface,
)

__all__ = [
    "normalize_grayscale",
    "surface_to_grayscale",
    "surface_to_binary",
    "convert_surface_to_grayscale_inplace",
    "compute_auto_levels",
    "get_visible_grayscale_values",
    "srgb_to_linear",
    "linear_to_srgb",
    "make_surface_transparent",
    "make_transparent_except_color",
    "rgba_to_cairo_surface",
    "resize_and_crop_from_full_image",
    "safe_crop",
    "extract_vips_metadata",
    "get_mm_per_pixel",
    "get_physical_size_mm",
    "normalize_to_rgba",
    "vips_rgba_to_cairo_surface",
    "apply_mask_to_vips_image",
    "CAIRO_MAX_DIMENSION",
    "to_mm",
    "parse_length",
    "calculate_chunk_layout",
]
