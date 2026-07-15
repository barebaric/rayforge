"""
Helper functions and enums for the material test grid.

Moved from the deleted material_test_grid_producer.py (Phase 6: eliminate
producer infrastructure). These utilities are used by the material test
widget and command for preview rendering and size calculation.
"""

from enum import Enum
from typing import Any, Dict, Tuple

import cairo
import numpy as np
from raygeo.ops.assembly.material_test_grid import (
    generate_material_test_grid_preview,
)

_MM_PER_INCH = 25.4


class MaterialTestGridType(Enum):
    """Material test types."""

    CUT = "Cut"
    ENGRAVE = "Engrave"


class GridMode(Enum):
    """Defines which parameters vary on grid axes."""

    POWER_VS_SPEED = "Power vs Speed"
    POWER_VS_PASSES = "Power vs Passes"
    SPEED_VS_PASSES = "Speed vs Passes"


def get_material_test_proportional_size(
    params: Dict[str, Any],
) -> Tuple[float, float]:
    """
    Calculates the natural size in mm for a material test grid.

    Args:
        params: A dictionary of geometric parameters for the grid, including
          'grid_dimensions', 'shape_size', 'spacing', and 'include_labels'.

    Returns:
        A tuple of (width, height) in millimeters.
    """
    cols, rows = map(int, params.get("grid_dimensions", (5, 5)))
    shape_size = params.get("shape_size", 10.0)
    spacing = params.get("spacing", 2.0)
    include_labels = params.get("include_labels", True)

    base_margin_left = min(shape_size * 1.5, 15.0)
    base_margin_top = min(shape_size * 1.5, 15.0)
    width = (cols * shape_size) + ((cols - 1) * spacing)
    height = (rows * shape_size) + ((rows - 1) * spacing)
    if include_labels:
        width += base_margin_left
        height += base_margin_top
    return width, height


def draw_preview(
    ctx: cairo.Context,
    width_px: float,
    height_px: float,
    params: Dict[str, Any],
):
    """
    Draws a visual-only preview of the material test grid.

    Renders the Ops via ``generate_material_test_grid_preview`` (raygeo),
    then blits the resulting RGBA buffer to the Cairo context.
    """
    size_mm = get_material_test_proportional_size(params)
    dpi_x = width_px / size_mm[0] * _MM_PER_INCH
    dpi_y = height_px / size_mm[1] * _MM_PER_INCH
    dpi = (dpi_x + dpi_y) / 2.0
    img = generate_material_test_grid_preview(
        size_mm=size_mm,
        dpi=dpi,
        cols=params.get("grid_dimensions", (5, 5))[0],
        rows=params.get("grid_dimensions", (5, 5))[1],
        min_speed=params.get("speed_range", (100.0, 500.0))[0],
        max_speed=params.get("speed_range", (100.0, 500.0))[1],
        min_power=params.get("power_range", (10.0, 100.0))[0],
        max_power=params.get("power_range", (10.0, 100.0))[1],
        min_passes=params.get("passes_range", (1, 5))[0],
        max_passes=params.get("passes_range", (1, 5))[1],
        fixed_speed=params.get("fixed_speed", 1000.0),
        fixed_power=params.get("fixed_power", 50.0),
        shape_size=params.get("shape_size", 10.0),
        spacing=params.get("spacing", 2.0),
        mode=(
            "cut" if params.get("test_type", "Cut") == "Cut" else "engrave"
        ),
        grid_mode=params.get("grid_mode", "Power vs Speed"),
        include_labels=params.get("include_labels", True),
    )

    h, w = img.shape[:2]
    bgra = np.empty_like(img)
    bgra[:, :, 0] = img[:, :, 2]  # B = R
    bgra[:, :, 1] = img[:, :, 1]  # G = G
    bgra[:, :, 2] = img[:, :, 0]  # R = B
    bgra[:, :, 3] = img[:, :, 3]  # A = A
    surface = cairo.ImageSurface.create_for_data(
        np.ascontiguousarray(bgra),
        cairo.FORMAT_ARGB32,
        w,
        h,
    )
    ctx.set_source_surface(surface, 0, 0)
    ctx.paint()
    surface.finish()


def draw_material_test_preview(
    ctx: cairo.Context, width: float, height: float, params: Dict[str, Any]
):
    """Stable entry point for the generic procedural renderer."""
    draw_preview(ctx, width, height, params)
