"""
Material Test Renderer

Renders a preview visualization of a material test grid for display on the
canvas. The actual ops generation is handled by MaterialTestGridProducer.
"""

from __future__ import annotations
import cairo
import json
import logging
from typing import Dict, Any, Optional, TYPE_CHECKING
from .base_renderer import Renderer
import warnings

with warnings.catch_warnings():
    warnings.simplefilter("ignore", DeprecationWarning)
    import pyvips

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


class MaterialTestRenderer(Renderer):
    """Renders material test grid previews."""

    def _get_params_from_data(
        self, data: Optional[bytes]
    ) -> Optional[Dict[str, Any]]:
        if not data:
            return None
        try:
            return json.loads(data.decode("utf-8"))
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
        grid_mode = params.get("grid_mode", "Power vs Speed")
        passes_range = params.get("passes_range", (1, 5))

        min_speed, max_speed = speed_range
        min_power, max_power = power_range
        min_passes, max_passes = int(passes_range[0]), int(passes_range[1])

        if grid_mode == "Power vs Passes":
            col_range = (min_power, max_power)
            row_range = (float(min_passes), float(max_passes))
        elif grid_mode == "Speed vs Passes":
            col_range = (min_speed, max_speed)
            row_range = (float(min_passes), float(max_passes))
        else:
            col_range = (min_power, max_power)
            row_range = (min_speed, max_speed)

        col_step = (
            (col_range[1] - col_range[0]) / (cols - 1) if cols > 1 else 0
        )
        row_step = (
            (row_range[1] - row_range[0]) / (rows - 1) if rows > 1 else 0
        )

        col_span = col_range[1] - col_range[0]
        row_span = row_range[1] - row_range[0]

        for r in range(rows):
            for c in range(cols):
                col_val = col_range[0] + c * col_step
                row_val = row_range[0] + r * row_step

                if grid_mode == "Power vs Passes":
                    power_factor = (
                        (col_val - col_range[0]) / col_span
                        if col_span > 0
                        else 0
                    )
                    passes_factor = (
                        (row_val - row_range[0]) / row_span
                        if row_span > 0
                        else 0
                    )
                    intensity = (power_factor + passes_factor) / 2.0
                elif grid_mode == "Speed vs Passes":
                    speed_factor = (
                        1.0 - (col_val - col_range[0]) / col_span
                        if col_span > 0
                        else 0
                    )
                    passes_factor = (
                        (row_val - row_range[0]) / row_span
                        if row_span > 0
                        else 0
                    )
                    intensity = (speed_factor + passes_factor) / 2.0
                else:
                    speed_factor = (
                        1.0 - (row_val - row_range[0]) / row_span
                        if row_span > 0
                        else 0
                    )
                    power_factor = (
                        (col_val - col_range[0]) / col_span
                        if col_span > 0
                        else 0
                    )
                    intensity = (speed_factor + power_factor) / 2.0

                # Gradient from light gray (0.9) to dark gray (0.3)
                gray = 0.9 - (intensity * 0.6)

                x = c * (shape_size + spacing)
                y = r * (shape_size + spacing)

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

    def render_base_image(
        self,
        data: bytes,
        width: int,
        height: int,
        **kwargs,
    ) -> Optional[pyvips.Image]:
        params = self._get_params_from_data(data)
        if not params:
            return None

        surface = cairo.ImageSurface(cairo.FORMAT_ARGB32, width, height)
        ctx = cairo.Context(surface)
        ctx.set_source_rgb(1, 1, 1)
        ctx.paint()

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
        offset_x = 0
        offset_y = 0
        total_width = grid_width
        total_height = grid_height

        if include_labels:
            base_margin = min(shape_size * 1.5, 15.0)
            total_width = grid_width + base_margin
            total_height = grid_height + base_margin
            offset_x = base_margin
            offset_y = base_margin

        scale_x = width / total_width if total_width > 0 else 1
        scale_y = height / total_height if total_height > 0 else 1

        ctx.scale(scale_x, -scale_y)
        ctx.translate(offset_x, -total_height + offset_y)

        self._draw_grid(ctx, params)

        h, w = surface.get_height(), surface.get_width()
        vips_image = pyvips.Image.new_from_memory(
            surface.get_data(), w, h, 4, "uchar"
        )
        b, g, r, a = (
            vips_image[0],
            vips_image[1],
            vips_image[2],
            vips_image[3],
        )
        return r.bandjoin([g, b, a])
