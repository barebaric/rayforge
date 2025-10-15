"""
Material Test Renderer

Renders a preview visualization of a material test grid for display on the
canvas. The actual ops generation is handled by MaterialTestGridProducer.
"""

from __future__ import annotations
import cairo
import json
import logging
from typing import (
    Dict,
    Any,
    Tuple,
    TYPE_CHECKING,
    Optional,
)
from .base_renderer import Renderer

if TYPE_CHECKING:
    from ..core.workpiece import WorkPiece

logger = logging.getLogger(__name__)


class MaterialTestRenderer(Renderer):
    """
    Renders a visual preview of a material test grid.

    The preview shows:
    - Grid cells with gradient shading (darker = more intense)
    - Speed and power labels
    - Axis labels
    """

    def __init__(self):
        """Initializes the renderer."""
        super().__init__()

    def render_to_pixels(
        self,
        workpiece: WorkPiece,
        width: int,
        height: int,
    ) -> Optional[cairo.ImageSurface]:
        """
        Renders the material test grid preview.

        Args:
            workpiece: WorkPiece to render (must have material_test
                ImportSource)
            width: Target width in pixels.
            height: Target height in pixels.

        Returns:
            Cairo ImageSurface with the rendered preview.
        """
        # Extract parameters from the workpiece's ImportSource data
        params = self._get_params_from_workpiece(workpiece)
        if not params:
            logger.warning("Could not extract material test parameters")
            return None

        surface = cairo.ImageSurface(cairo.FORMAT_ARGB32, width, height)
        ctx = cairo.Context(surface)

        # White background
        ctx.set_source_rgb(1, 1, 1)
        ctx.paint()

        # Calculate dimensions
        cols, rows = (
            int(params["grid_dimensions"][0]),
            int(params["grid_dimensions"][1]),
        )
        shape_size = params["shape_size"]
        spacing = params["spacing"]

        grid_width = cols * (shape_size + spacing) - spacing
        grid_height = rows * (shape_size + spacing) - spacing

        # Get margins if labels are enabled
        include_labels = params.get("include_labels", True)
        if include_labels:
            # Scale margins with font size to accommodate larger labels
            font_scale = 4.375 / 2.5
            label_margin = 15.0 * font_scale
            # Total content area includes negative label space
            total_width = grid_width + label_margin
            total_height = grid_height + label_margin
            # Grid starts at (0,0), but we need to account for negative
            # label space
            offset_x = label_margin
            offset_y = label_margin
        else:
            total_width = grid_width
            total_height = grid_height
            offset_x = 0
            offset_y = 0

        # Scale context to fit total content area
        scale_x = width / total_width if total_width > 0 else 1
        scale_y = height / total_height if total_height > 0 else 1

        # Flip Y-axis by using negative scale, then translate to correct
        # position
        ctx.scale(scale_x, -scale_y)

        # After Y-flip, translate to account for negative label
        # coordinates and flip
        # Y is now flipped, so we need to translate by -total_height to
        # position correctly
        ctx.translate(offset_x, -total_height + offset_y)

        # Draw the grid
        self._draw_grid(ctx, params)

        return surface

    def _get_params_from_workpiece(
        self, workpiece: WorkPiece
    ) -> Optional[Dict[str, Any]]:
        """Extracts producer parameters from workpiece data."""
        if not workpiece.data:
            return None

        try:
            # Data is JSON-encoded parameters
            params = json.loads(workpiece.data.decode("utf-8"))
            return params
        except (json.JSONDecodeError, UnicodeDecodeError) as e:
            logger.error(f"Failed to decode material test parameters: {e}")
            return None

    def _draw_grid(self, ctx: cairo.Context, params: Dict[str, Any]):
        cols, rows = (
            int(params["grid_dimensions"][0]),
            int(params["grid_dimensions"][1]),
        )
        shape_size = params["shape_size"]
        spacing = params["spacing"]
        speed_range = params["speed_range"]
        power_range = params["power_range"]
        test_type = params.get("test_type", "Cut")

        min_speed, max_speed = speed_range
        min_power, max_power = power_range

        # Rows vary speed (Y-axis), columns vary power (X-axis)
        speed_step = (max_speed - min_speed) / (rows - 1) if rows > 1 else 0
        power_step = (max_power - min_power) / (cols - 1) if cols > 1 else 0

        for r in range(rows):
            for c in range(cols):
                current_speed = min_speed + r * speed_step
                current_power = min_power + c * power_step

                x = c * (shape_size + spacing)
                y = r * (shape_size + spacing)

                # Calculate intensity (darker = more aggressive)
                speed_factor = (
                    1.0 - (current_speed - min_speed) / (max_speed - min_speed)
                    if max_speed > min_speed
                    else 0
                )
                power_factor = (
                    (current_power - min_power) / (max_power - min_power)
                    if max_power > min_power
                    else 0
                )
                intensity = (speed_factor + power_factor) / 2.0

                # Gradient from light gray (0.9) to dark gray (0.3)
                gray = 0.9 - (intensity * 0.6)

                if test_type == "Engrave":
                    # Fill cell with horizontal lines for engrave mode
                    ctx.set_source_rgb(gray, gray, gray)
                    ctx.rectangle(x, y, shape_size, shape_size)
                    ctx.fill()

                    # Draw horizontal raster lines
                    ctx.set_source_rgb(0.5, 0.5, 0.5)
                    ctx.set_line_width(0.1)
                    line_spacing = shape_size / 10  # ~10 lines per box
                    for i in range(11):
                        y_line = y + (i * line_spacing)
                        ctx.move_to(x, y_line)
                        ctx.line_to(x + shape_size, y_line)
                        ctx.stroke()
                else:
                    # Cut mode: just a light fill
                    ctx.set_source_rgb(0.95, 0.95, 0.95)
                    ctx.rectangle(x, y, shape_size, shape_size)
                    ctx.fill()

                # Border for both modes
                ctx.set_source_rgb(0.3, 0.3, 0.3)
                ctx.set_line_width(0.2)
                ctx.rectangle(x, y, shape_size, shape_size)
                ctx.stroke()

    def get_natural_size(
        self, workpiece: WorkPiece
    ) -> Optional[Tuple[float, float]]:
        """Returns the natural size of the test grid in mm."""
        params = self._get_params_from_workpiece(workpiece)
        if not params:
            return None

        cols, rows = (
            int(params["grid_dimensions"][0]),
            int(params["grid_dimensions"][1]),
        )
        shape_size = params["shape_size"]
        spacing = params["spacing"]

        grid_width = cols * (shape_size + spacing) - spacing
        grid_height = rows * (shape_size + spacing) - spacing

        # Add margins for labels if enabled (using shared layout)
        include_labels = params.get("include_labels", True)
        if include_labels:
            font_scale = 4.375 / 2.5
            label_margin = 15.0 * font_scale
            width = grid_width + label_margin
            height = grid_height + label_margin
        else:
            width = grid_width
            height = grid_height

        return width, height
