"""
Image utility functions split into logical submodules.

This module provides utilities for:
- Grayscale and binary image conversion (grayscale module)
- Transparency manipulation (transparency module)
- PyVips image operations (vips module)
- Unit conversion and layout (unit module)
"""

from .grayscale import (
    normalize_grayscale,
    surface_to_grayscale,
    surface_to_binary,
    convert_surface_to_grayscale_inplace,
)

from .transparency import (
    make_surface_transparent,
    make_transparent_except_color,
)

from .vips import (
    resize_and_crop_from_full_image,
    safe_crop,
    extract_vips_metadata,
    get_mm_per_pixel,
    get_physical_size_mm,
    normalize_to_rgba,
    vips_rgba_to_cairo_surface,
    apply_mask_to_vips_image,
)

from .unit import (
    CAIRO_MAX_DIMENSION,
    to_mm,
    parse_length,
    calculate_chunk_layout,
)

__all__ = [
    "normalize_grayscale",
    "surface_to_grayscale",
    "surface_to_binary",
    "convert_surface_to_grayscale_inplace",
    "make_surface_transparent",
    "make_transparent_except_color",
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
