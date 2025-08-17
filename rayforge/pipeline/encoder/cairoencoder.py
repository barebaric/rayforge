import math
from typing import Tuple
import cairo
import logging
from ...core.ops import (Ops, MoveToCommand, LineToCommand, ArcToCommand)
from .base import OpsEncoder


logger = logging.getLogger(__name__)


class CairoEncoder(OpsEncoder):
    """
    Encodes a Ops onto a Cairo surface, respecting embedded state commands
    (color, geometry) and machine dimensions for coordinate adjustments.
    """
    def encode(self,
               ops: Ops,
               ctx: cairo.Context,
               scale: Tuple[float, float],
               cut_color: Tuple[float, float, float] = (1, 0, 1),
               travel_color: Tuple[float, float, float] = (.85, .85, .85),
               show_travel_moves: bool = False) -> None:
        # Calculate scaling factors from surface and machine dimensions
        # The Ops are in machine coordinates, i.e. zero point
        # at the bottom left, and units are mm.
        # Since Cairo coordinates put the zero point at the top left, we must
        # subtract Y from the machine's Y axis maximum.
        ctx.save()
        scale_x, scale_y = scale
        ymax = ctx.get_target().get_height()/scale_y  # For Y-axis inversion

        # Apply coordinate scaling and line width
        ctx.scale(scale_x, scale_y)
        ctx.set_hairline(True)
        ctx.move_to(0, ymax)

        prev_point_2d = 0, ymax
        for segment in ops.segments():
            for cmd in segment:
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

                        radius = math.dist((start_x, start_y),
                                           (center_x, center_y))
                        angle1 = math.atan2(start_y - center_y,
                                            start_x - center_x)
                        angle2 = math.atan2(adjusted_y - center_y,
                                            x - center_x)

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
                                center_x,
                                center_y,
                                radius,
                                angle1,
                                angle2
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
