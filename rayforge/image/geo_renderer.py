"""Geometry rendering utilities using Cairo."""

import math
from typing import Optional, Tuple

import cairo

from raygeo import Geometry
from raygeo.path import PyCommand


def geometry_to_cairo(
    geometry: Geometry,
    ctx: cairo.Context,
) -> None:
    """
    Render a Geometry object to a Cairo context.

    Args:
        geometry: The geometry to render.
        ctx: The Cairo context to draw to.
    """
    geometry.sync_to_data()
    last_point = (0.0, 0.0)

    for cmd in geometry.iter_typed_commands():
        end = (cmd.end[0], cmd.end[1])

        if isinstance(cmd, PyCommand.Move):
            ctx.move_to(end[0], end[1])
        elif isinstance(cmd, PyCommand.Line):
            ctx.line_to(end[0], end[1])
        elif isinstance(cmd, PyCommand.Arc):
            cx = last_point[0] + cmd.center_offset[0]
            cy = last_point[1] + cmd.center_offset[1]
            radius = math.hypot(cmd.center_offset[0], cmd.center_offset[1])

            start_angle = math.atan2(
                -cmd.center_offset[1], -cmd.center_offset[0]
            )
            end_angle = math.atan2(end[1] - cy, end[0] - cx)

            clockwise = cmd.clockwise
            if radius > 1e-9 and abs(end_angle - start_angle) < 1e-9:
                mid = start_angle + math.pi
                if clockwise:
                    ctx.arc_negative(cx, cy, radius, start_angle, mid)
                    ctx.arc_negative(cx, cy, radius, mid, start_angle)
                else:
                    ctx.arc(cx, cy, radius, start_angle, mid)
                    ctx.arc(cx, cy, radius, mid, start_angle)
            elif clockwise:
                ctx.arc_negative(cx, cy, radius, start_angle, end_angle)
            else:
                ctx.arc(cx, cy, radius, start_angle, end_angle)
        elif isinstance(cmd, PyCommand.Bezier):
            ctx.curve_to(
                cmd.control1[0],
                cmd.control1[1],
                cmd.control2[0],
                cmd.control2[1],
                end[0],
                end[1],
            )

        last_point = end


def render_geometry_to_png(
    geometry: Geometry,
    size: int,
    line_width: Optional[float] = None,
    color: Optional[Tuple[float, float, float, float]] = None,
) -> Optional[bytes]:
    """
    Render a geometry to PNG bytes fitting within a square of ``size`` pixels.

    Returns None if the geometry is empty.

    Args:
        geometry: The geometry to render.
        size: The square size of the output image in pixels.
        line_width: Optional line width. Defaults to max(1.0/scale, 0.5).
        color: Optional RGBA color tuple. Defaults to (0.55, 0.55, 0.55, 1.0).

    Returns:
        PNG bytes or None if geometry is empty.
    """
    if geometry.is_empty():
        return None

    x1, y1, x2, y2 = geometry.rect()
    gw = x2 - x1
    gh = y2 - y1
    if gw < 1e-9 or gh < 1e-9:
        return None

    padding = 4
    available = size - 2 * padding
    scale = min(available / gw, available / gh)

    surface = cairo.ImageSurface(cairo.Format.ARGB32, size, size)
    ctx = cairo.Context(surface)

    ctx.set_source_rgba(0, 0, 0, 0)
    ctx.set_operator(cairo.Operator.SOURCE)
    ctx.paint()
    ctx.set_operator(cairo.Operator.OVER)

    ctx.translate(size / 2, size / 2)
    ctx.scale(scale, -scale)
    ctx.translate(-(x1 + gw / 2), -(y1 + gh / 2))

    rgba = color or (0.55, 0.55, 0.55, 1.0)
    ctx.set_source_rgba(*rgba)
    width = line_width if line_width is not None else max(1.0 / scale, 0.5)
    ctx.set_line_width(width)
    geometry_to_cairo(geometry, ctx)
    ctx.stroke()

    surface.flush()

    import io

    buf = io.BytesIO()
    surface.write_to_png(buf)
    return buf.getvalue()
