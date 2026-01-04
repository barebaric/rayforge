import cairo
import numpy as np
import math
from typing import TYPE_CHECKING
from ...core.geo.constants import (
    CMD_TYPE_MOVE,
    CMD_TYPE_LINE,
    CMD_TYPE_ARC,
    COL_TYPE,
    COL_X,
    COL_Y,
    COL_I,
    COL_J,
    COL_CW,
)

if TYPE_CHECKING:
    from ...core.geo.geometry import Geometry


def convert_surface_to_grayscale(surface):
    # Determine the number of channels based on the format
    surface_format = surface.get_format()
    if surface_format != cairo.FORMAT_ARGB32:
        raise ValueError("Unsupported Cairo surface format")

    width, height = surface.get_width(), surface.get_height()
    data = surface.get_data()
    data = np.frombuffer(data, dtype=np.uint8).reshape((height, width, 4))

    # Convert RGB to grayscale using luminosity method
    gray = (
        0.299 * data[:, :, 2] + 0.587 * data[:, :, 1] + 0.114 * data[:, :, 0]
    ).astype(np.uint8)

    # Set RGB channels to gray, keep alpha unchanged
    data[:, :, :3] = gray[:, :, None]

    return surface


def make_transparent(surface, threshold=250):
    if surface.get_format() != cairo.FORMAT_ARGB32:
        raise ValueError("Surface must be in ARGB32 format.")

    width, height = surface.get_width(), surface.get_height()
    stride = surface.get_stride()

    # Get pixel data as a NumPy array
    data = surface.get_data()
    buf = np.frombuffer(data, dtype=np.uint8).reshape((height, stride))

    # Convert to 32-bit ARGB view
    argb = buf.view(dtype=np.uint32)[:, :width]

    # Extract channels
    r = (argb >> 16) & 0xFF  # Red
    g = (argb >> 8) & 0xFF  # Green
    b = argb & 0xFF  # Blue

    # Find "almost white" pixels
    brightness = (
        r.astype(np.uint16) + g.astype(np.uint16) + b.astype(np.uint16)
    ) // 3
    mask = brightness >= threshold

    # Set these pixels to transparent
    argb[mask] = (0x00 << 24) | (r[mask] << 16) | (g[mask] << 8) | b[mask]

    # No need to return anything as the surface is modified in place


def draw_geometry_to_cairo_context(geometry: "Geometry", ctx: cairo.Context):
    """
    Draws a Geometry object's path to a Cairo context.

    This function iterates through the geometry's commands and translates
    them into the corresponding Cairo drawing operations.

    Args:
        geometry: The Geometry object to draw.
        ctx: The Cairo context to draw on.
    """
    last_point = (0.0, 0.0)
    data = geometry.data
    if data is None:
        return

    for i in range(len(data)):
        row = data[i]
        cmd_type = row[COL_TYPE]
        end = (row[COL_X], row[COL_Y])

        if cmd_type == CMD_TYPE_MOVE:
            ctx.move_to(end[0], end[1])
        elif cmd_type == CMD_TYPE_LINE:
            ctx.line_to(end[0], end[1])
        elif cmd_type == CMD_TYPE_ARC:
            # Cairo's arc needs center, radius, and angles.
            center_x = last_point[0] + row[COL_I]
            center_y = last_point[1] + row[COL_J]
            radius = math.hypot(row[COL_I], row[COL_J])

            start_angle = math.atan2(
                -row[COL_J], -row[COL_I]
            )  # Vector from center to start
            end_angle = math.atan2(end[1] - center_y, end[0] - center_x)

            clockwise = bool(row[COL_CW])
            if clockwise:
                ctx.arc_negative(
                    center_x, center_y, radius, start_angle, end_angle
                )
            else:
                ctx.arc(center_x, center_y, radius, start_angle, end_angle)

        last_point = end
