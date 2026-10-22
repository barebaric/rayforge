"""
Helper functions and enums for the material test grid.

Moved from the deleted material_test_grid_producer.py (Phase 6: eliminate
producer infrastructure). These utilities are used by the material test
widget and command for preview rendering and size calculation.
"""

from enum import Enum
from gettext import gettext as _
from typing import Any

import cairo
import numpy as np
from raygeo.ops.assembly.material_test_grid import (
    generate_material_test_grid_preview,
)

from rayforge.shared.units.formatter import get_display_unit_settings

_MM_PER_INCH = 25.4


class MaterialTestGridType(Enum):
    """Material test types."""

    CUT = "Cut"
    ENGRAVE = "Engrave"

    def label(self) -> str:
        labels = {
            self.CUT: _("Cut"),
            self.ENGRAVE: _("Engrave"),
        }
        return labels[self]


class GridMode(Enum):
    """Defines which parameters vary on grid axes.

    The enum value is the stable identifier expected by raygeo's
    material test assembler and by serialized documents. The label is
    translated for display only.
    """

    POWER_VS_SPEED = "Power vs Speed"
    POWER_VS_PASSES = "Power vs Passes"
    SPEED_VS_PASSES = "Speed vs Passes"
    SPEED_VS_OFFSET = "Speed vs Offset"

    @classmethod
    def coerce(cls, value: "GridMode | str | None") -> "GridMode":
        """Resolve a grid mode from a member, value, or name."""
        if isinstance(value, cls):
            return value
        if isinstance(value, str):
            try:
                return cls[value]
            except KeyError:
                pass
            try:
                return cls(value)
            except ValueError:
                pass
        return cls.POWER_VS_SPEED

    @property
    def label(self) -> str:
        labels = {
            self.POWER_VS_SPEED: _("Power vs Speed"),
            self.POWER_VS_PASSES: _("Power vs Passes"),
            self.SPEED_VS_PASSES: _("Speed vs Passes"),
            self.SPEED_VS_OFFSET: _("Speed vs Offset"),
        }
        return labels[self]

    @property
    def uses_power_range(self) -> bool:
        return self in (
            GridMode.POWER_VS_SPEED,
            GridMode.POWER_VS_PASSES,
        )

    @property
    def uses_speed_range(self) -> bool:
        return self in (
            GridMode.POWER_VS_SPEED,
            GridMode.SPEED_VS_PASSES,
            GridMode.SPEED_VS_OFFSET,
        )

    @property
    def uses_passes_range(self) -> bool:
        return self in (
            GridMode.POWER_VS_PASSES,
            GridMode.SPEED_VS_PASSES,
        )

    @property
    def uses_offset_range(self) -> bool:
        return self is GridMode.SPEED_VS_OFFSET

    @property
    def uses_fixed_speed(self) -> bool:
        return self is GridMode.POWER_VS_PASSES

    @property
    def uses_fixed_power(self) -> bool:
        return self in (
            GridMode.SPEED_VS_PASSES,
            GridMode.SPEED_VS_OFFSET,
        )

    @property
    def dimension_labels(
        self,
    ) -> tuple[tuple[str, str], tuple[str, str]]:
        """Titles and subtitles for the dimensions rows."""
        labels = {
            self.POWER_VS_SPEED: (
                (_("Columns (Power Steps)"), _("Rows (Speed Steps)")),
                (
                    _("Number of power variations"),
                    _("Number of speed variations"),
                ),
            ),
            self.POWER_VS_PASSES: (
                (_("Columns (Power Steps)"), _("Rows (Passes Steps)")),
                (
                    _("Number of power variations"),
                    _("Number of passes variations"),
                ),
            ),
            self.SPEED_VS_PASSES: (
                (_("Columns (Speed Steps)"), _("Rows (Passes Steps)")),
                (
                    _("Number of speed variations"),
                    _("Number of passes variations"),
                ),
            ),
            self.SPEED_VS_OFFSET: (
                (_("Columns (Speed Steps)"), _("Rows (Offset Steps)")),
                (
                    _("Number of speed variations"),
                    _("Number of offset variations"),
                ),
            ),
        }
        return labels[self]


def get_material_test_proportional_size(
    params: dict[str, Any],
) -> tuple[float, float]:
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
    params: dict[str, Any],
    size_mm: tuple[float, float] | None = None,
):
    """
    Draws a visual-only preview of the material test grid.

    Renders the Ops via ``generate_material_test_grid_preview`` (raygeo),
    then blits the resulting RGBA buffer to the Cairo context.

    ``size_mm`` is the current workpiece size; the grid is rendered at
    that size (falling back to the recipe's natural size) so the preview
    matches the material-test ops, which scale to fill ``workpiece.size``.
    """
    if size_mm is None:
        size_mm = get_material_test_proportional_size(params)
    dpi_x = width_px / size_mm[0] * _MM_PER_INCH
    dpi_y = height_px / size_mm[1] * _MM_PER_INCH
    dpi = max(dpi_x, dpi_y)
    unit_label, factor, precision = get_display_unit_settings("speed")
    img = generate_material_test_grid_preview(
        size_mm=size_mm,
        dpi=dpi,
        cols=int(params.get("grid_dimensions", (5, 5))[0]),
        rows=int(params.get("grid_dimensions", (5, 5))[1]),
        min_speed=params.get("speed_range", (100.0, 500.0))[0],
        max_speed=params.get("speed_range", (100.0, 500.0))[1],
        min_power=params.get("power_range", (10.0, 100.0))[0],
        max_power=params.get("power_range", (10.0, 100.0))[1],
        min_passes=params.get("passes_range", (1, 5))[0],
        max_passes=params.get("passes_range", (1, 5))[1],
        min_offset=params.get("offset_range", (-0.5, 0.5))[0],
        max_offset=params.get("offset_range", (-0.5, 0.5))[1],
        fixed_speed=params.get("fixed_speed", 1000.0),
        fixed_power=params.get("fixed_power", 50.0),
        shape_size=params.get("shape_size", 10.0),
        spacing=params.get("spacing", 2.0),
        mode=("cut" if params.get("test_type", "Cut") == "Cut" else "engrave"),
        grid_mode=GridMode.coerce(params.get("grid_mode")).value,
        include_labels=params.get("include_labels", True),
        label_power_percent=params.get("label_power_percent", 10.0),
        label_speed=params.get("label_speed", 1000.0),
        speed_unit_label=unit_label,
        speed_label_factor=factor,
        speed_label_precision=precision,
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
    # Scale the grid to fill the whole target surface. The raster
    # pipeline maps the rendered image back to millimeters using the
    # per-axis pixels-per-mm, so stretching the square grid here (the
    # surface aspect ratio is set by the laser spot geometry) yields
    # correctly-proportioned cells that cover the full workpiece.
    ctx.scale(width_px / w, height_px / h)
    ctx.set_source_surface(surface, 0, 0)
    ctx.paint()
    surface.finish()


def draw_material_test_preview(
    ctx: cairo.Context,
    width: float,
    height: float,
    params: dict[str, Any],
    size_mm: tuple[float, float] | None = None,
):
    """Stable entry point for the generic procedural renderer."""
    draw_preview(ctx, width, height, params, size_mm=size_mm)
