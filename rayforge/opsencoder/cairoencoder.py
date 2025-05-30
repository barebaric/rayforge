import math
import cairo
import logging
from ..models.ops import Ops, MoveToCommand, LineToCommand, ArcToCommand
from ..models.machine import Machine
from .encoder import OpsEncoder


logger = logging.getLogger(__name__)


class CairoEncoder(OpsEncoder):
    """
    Encodes a Ops onto a Cairo surface, respecting embedded state commands
    (color, geometry) and machine dimensions for coordinate adjustments.
    """
    def encode(self,
               ops: Ops,
               machine: Machine,
               surface: cairo.ImageSurface,
               scale: tuple[float, float],
               show_travel_moves: bool = False) -> None:
        # Set up Cairo context and scaling
        ctx = cairo.Context(surface)
        ctx.set_source_rgb(1, 0, 1)

        # Calculate scaling factors from surface and machine dimensions
        # The Ops are in machine coordinates, i.e. zero point
        # at the bottom left, and units are mm.
        # Since Cairo coordinates put the zero point at the top left, we must
        # subtract Y from the machine's Y axis maximum.
        scale_x, scale_y = scale
        ymax = surface.get_height()/scale_y  # For Y-axis inversion

        # Apply coordinate scaling and line width
        ctx.scale(scale_x, scale_y)
        ctx.set_hairline(True)
        ctx.move_to(0, ymax)

        prev_point = 0, ymax
        for segment in ops.segments():
            for cmd in segment:
                match cmd, cmd.end:
                    case MoveToCommand(), (x, y):
                        adjusted_y = ymax - y

                        # Paint the travel move. We do not have to worry that
                        # there may be any unpainted path before it, because
                        # Ops.segments() ensures that each travel move opens
                        # a new segment.
                        if show_travel_moves:
                            ctx.set_source_rgb(.8, .8, .8)
                            ctx.move_to(*prev_point)
                            ctx.line_to(x, adjusted_y)
                            ctx.stroke()

                        ctx.move_to(x, adjusted_y)

                    case LineToCommand(), (x, y):
                        adjusted_y = ymax-y
                        ctx.line_to(x, adjusted_y)
                        prev_point = x, adjusted_y

                    case ArcToCommand(), (x, y):
                        # Start point is the x, y of the previous operation.
                        start_x, start_y = ctx.get_current_point()
                        ctx.set_source_rgb(1, 0, 1)
                        ctx.stroke()

                        # Draw the arc in the correct direction
                        # x, y: absolute values
                        # i, j: relative pos of arc center from start point.
                        i, j = cmd.center_offset
                        center_x = start_x+i
                        center_y = start_y+j
                        adjusted_y = ymax-y
                        radius = math.dist((start_x, start_y),
                                           (center_x, center_y))
                        angle1 = math.atan2(start_y - center_y,
                                            start_x - center_x)
                        angle2 = math.atan2(adjusted_y - center_y,
                                            x - center_x)
                        if cmd.clockwise:
                            ctx.arc(center_x, center_y, radius, angle1, angle2)
                        else:
                            ctx.arc_negative(
                                center_x,
                                center_y,
                                radius,
                                angle1,
                                angle2
                            )
                        ctx.set_source_rgb(0, 0, 1)
                        ctx.stroke()
                        ctx.move_to(x, adjusted_y)
                        prev_point = x, adjusted_y

                    case _:
                        pass  # ignore unsupported operations

            # Draw the segment.
            ctx.set_source_rgb(1, 0, 1)
            ctx.stroke()
