import cairo
import numpy as np
import math
from typing import TYPE_CHECKING
from ...core.geo.geometry import (
    MoveToCommand,
    LineToCommand,
    ArcToCommand,
    MovingCommand,
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
    for cmd in geometry.commands:
        if isinstance(cmd, MoveToCommand):
            ctx.move_to(cmd.end[0], cmd.end[1])
        elif isinstance(cmd, LineToCommand):
            ctx.line_to(cmd.end[0], cmd.end[1])
        elif isinstance(cmd, ArcToCommand):
            # Cairo's arc needs center, radius, and angles.
            center_x = last_point[0] + cmd.center_offset[0]
            center_y = last_point[1] + cmd.center_offset[1]
            radius = math.hypot(cmd.center_offset[0], cmd.center_offset[1])

            start_angle = math.atan2(
                -cmd.center_offset[1], -cmd.center_offset[0]
            )
            end_angle = math.atan2(
                cmd.end[1] - center_y, cmd.end[0] - center_x
            )

            if cmd.clockwise:
                ctx.arc_negative(
                    center_x, center_y, radius, start_angle, end_angle
                )
            else:
                ctx.arc(center_x, center_y, radius, start_angle, end_angle)

        if isinstance(cmd, MovingCommand):
            last_point = (cmd.end[0], cmd.end[1])
