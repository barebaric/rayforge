import math
from typing import Tuple, Optional
import cairo
import logging
from ...core.ops import (
    Ops,
    MoveToCommand,
    LineToCommand,
    ArcToCommand,
    ScanLinePowerCommand,
)
from .base import OpsEncoder


logger = logging.getLogger(__name__)


class CairoEncoder(OpsEncoder):
    """
    Encodes a Ops onto a Cairo surface, respecting embedded state commands
    (color, geometry) and machine dimensions for coordinate adjustments.
    """

    def encode(
        self,
        ops: Ops,
        ctx: cairo.Context,
        scale: Tuple[float, float],
        cut_color: Tuple[float, float, float] = (1, 0, 1),
        travel_color: Tuple[float, float, float] = (0.85, 0.85, 0.85),
        show_travel_moves: bool = False,
        drawable_height_px: Optional[int] = None,
    ) -> None:
        # Calculate scaling factors from surface and machine dimensions
        # The Ops are in machine coordinates, i.e. zero point
        # at the bottom left, and units are mm.
        # Since Cairo coordinates put the zero point at the top left, we must
        # subtract Y from the machine's Y axis maximum.
        scale_x, scale_y = scale
        if scale_y == 0:
            return
        ctx.save()

        target_surface = ctx.get_target()
        if isinstance(target_surface, cairo.RecordingSurface):
            # For a RecordingSurface, extents are in user space units (mm).
            # The ymax for inversion is simply the height of the extents.
            extents = target_surface.get_extents()
            if extents:
                _x, _y, _w, h = extents
                ymax = h
            else:
                ymax = 0  # Should not happen if surface has extents.
        else:
            # For ImageSurface, use the explicitly provided drawable height
            # to prevent miscalculations when the context is pre-translated.
            height_px = (
                drawable_height_px
                if drawable_height_px is not None
                else target_surface.get_height()
            )
            ymax = height_px / scale_y

        # Apply coordinate scaling and line width
        ctx.scale(scale_x, scale_y)
        ctx.set_hairline(True)
        ctx.move_to(0, ymax)

        prev_point_2d = 0, ymax
        for segment in ops.segments():
            for cmd in segment:
                # Skip any command that is just a marker.
                if cmd.is_marker_command():
                    continue

                # Now it's safe to assume the command might have an .end
                # attribute.
                if cmd.end is None:
                    continue

                x, y, z = cmd.end
                adjusted_y = ymax - y

                match cmd:
                    case MoveToCommand():
                        # Paint the travel move. We do not have to worry that
                        # there may be any unpainted path before it, because
                        # Ops.segments() ensures that each travel move opens
                        # a new segment.
                        if show_travel_moves:
                            ctx.set_source_rgb(*travel_color)
                            ctx.move_to(*prev_point_2d)
                            ctx.line_to(x, adjusted_y)
                            ctx.stroke()

                        ctx.move_to(x, adjusted_y)
                        prev_point_2d = x, adjusted_y

                    case LineToCommand():
                        ctx.line_to(x, adjusted_y)
                        prev_point_2d = x, adjusted_y

                    case ScanLinePowerCommand():
                        # Highly optimized rendering using a Cairo gradient.
                        # This avoids linearizing the command and calling
                        # stroke() thousands of times for a single line.
                        if not cmd.power_values:
                            continue

                        start_x, start_y, _ = cmd.start_point
                        end_x, end_y, _ = cmd.end

                        # Use Cairo's coordinate system (Y-down)
                        cairo_start_y = ymax - start_y
                        cairo_end_y = ymax - end_y

                        grad = cairo.LinearGradient(
                            start_x, cairo_start_y, end_x, cairo_end_y
                        )

                        num_steps = len(cmd.power_values)
                        last_power = -1

                        for i, power in enumerate(cmd.power_values):
                            if power != last_power:
                                # To create a sharp transition, add a color
                                # stop for the previous color just before
                                # this one.
                                if i > 0:
                                    p_old = last_power / 255.0
                                    offset_old = (i / num_steps) - 1e-9
                                    grad.add_color_stop_rgb(
                                        offset_old, p_old, p_old, p_old
                                    )

                                p_new = power / 255.0
                                offset_new = i / num_steps
                                grad.add_color_stop_rgb(
                                    offset_new, p_new, p_new, p_new
                                )
                                last_power = power

                        # Add final color stop for the last segment
                        p_final = last_power / 255.0
                        grad.add_color_stop_rgb(1.0, p_final, p_final, p_final)

                        # Draw the entire scan line with the gradient
                        ctx.new_path()
                        ctx.move_to(start_x, cairo_start_y)
                        ctx.line_to(end_x, cairo_end_y)
                        ctx.set_source(grad)
                        ctx.stroke()

                        # The logical pen position moves to the end of the line
                        ctx.move_to(end_x, cairo_end_y)
                        prev_point_2d = x, adjusted_y

                    case ArcToCommand():
                        # Start point is the x, y of the previous operation.
                        start_x, start_y = ctx.get_current_point()
                        # Stroke any preceding line segments before drawing
                        # the arc
                        ctx.set_source_rgb(*cut_color)
                        ctx.stroke()

                        # Draw the arc in the correct direction
                        # x, y: absolute values
                        # i, j: relative pos of arc center from start point.
                        i, j = cmd.center_offset

                        # The center point must also be calculated in the
                        # Y-down system
                        center_x = start_x + i
                        center_y = start_y - j

                        radius = math.dist(
                            (start_x, start_y), (center_x, center_y)
                        )
                        angle1 = math.atan2(
                            start_y - center_y, start_x - center_x
                        )
                        angle2 = math.atan2(
                            adjusted_y - center_y, x - center_x
                        )

                        # To draw a CCW arc (clockwise=False) on a flipped
                        # canvas, we must use Cairo's CW function
                        # (arc_negative).
                        if cmd.clockwise:
                            # A CW arc in the source becomes CCW when Y-axis
                            # is flipped.
                            ctx.arc(center_x, center_y, radius, angle1, angle2)
                        else:
                            # A CCW arc (like from DXF) becomes CW when Y-axis
                            # is flipped.
                            ctx.arc_negative(
                                center_x, center_y, radius, angle1, angle2
                            )

                        ctx.stroke()
                        ctx.move_to(x, adjusted_y)
                        prev_point_2d = x, adjusted_y

                    case _:
                        pass  # ignore unsupported operations

            # Draw the segment.
            ctx.set_source_rgb(*cut_color)
            ctx.stroke()
        ctx.restore()
