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
    Command,
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
        travel_color: Tuple[float, float, float] = (1.0, 0.4, 0.0),
        zero_power_color: Tuple[float, float, float] = (1.0, 0.2, 0.5),
        show_travel_moves: bool = False,
        drawable_height: Optional[float] = None,
    ) -> None:
        """
        Main orchestration method to draw Ops onto a Cairo context.

        Args:
            ops: The operations to encode.
            ctx: The Cairo context to draw on.
            scale: The (x, y) scaling factors.
            cut_color: RGB color for cutting moves.
            travel_color: RGB color for travel moves.
            zero_power_color: RGB color for cutting moves with zero power.
            show_travel_moves: Whether to draw travel moves.
            drawable_height: Optional explicit height of the drawable area.
        """
        scale_x, scale_y = scale
        if scale_y == 0:
            return
        show_zero_power_moves = show_travel_moves

        ctx.save()
        try:
            ymax = self._setup_cairo_context(ctx, scale, drawable_height)
            prev_point_2d = (0.0, ymax)

            for segment in ops.segments():
                for cmd in segment:
                    if cmd.is_marker_command() or cmd.end is None:
                        continue

                    prev_point_2d = self._process_command(
                        ctx,
                        cmd,
                        ymax,
                        prev_point_2d,
                        cut_color,
                        travel_color,
                        zero_power_color,
                        show_travel_moves,
                        show_zero_power_moves,
                    )

                # Draw the accumulated path for the segment (LineTo, etc.)
                ctx.set_source_rgb(*cut_color)
                ctx.stroke()
        finally:
            ctx.restore()

    def _process_command(
        self,
        ctx: cairo.Context,
        cmd: Command,
        ymax: float,
        prev_point_2d: Tuple[float, float],
        cut_color: Tuple[float, float, float],
        travel_color: Tuple[float, float, float],
        zero_power_color: Tuple[float, float, float],
        show_travel_moves: bool,
        show_zero_power_moves: bool,
    ) -> Tuple[float, float]:
        """
        Dispatches a command to the appropriate handler.
        Returns the new logical pen position.
        """
        x, y, _ = cmd.end
        adjusted_y = ymax - y

        match cmd:
            case MoveToCommand():
                return self._handle_move_to(
                    ctx,
                    cmd,
                    (x, adjusted_y),
                    prev_point_2d,
                    show_travel_moves,
                    travel_color,
                )
            case LineToCommand():
                is_zero_power = cmd.state is not None and cmd.state.power == 0
                if is_zero_power:
                    # Stroke any preceding path with the standard cut color
                    ctx.set_source_rgb(*cut_color)
                    ctx.stroke()

                    if show_zero_power_moves:
                        ctx.set_source_rgb(*zero_power_color)
                        ctx.move_to(*prev_point_2d)
                        ctx.line_to(x, adjusted_y)
                        ctx.stroke()

                    ctx.move_to(x, adjusted_y)
                    return x, adjusted_y
                else:
                    ctx.line_to(x, adjusted_y)
                    return x, adjusted_y
            case ScanLinePowerCommand():
                return self._handle_scanline(
                    ctx,
                    cmd,
                    ymax,
                    prev_point_2d,
                    cut_color,
                    zero_power_color,
                    show_zero_power_moves,
                )
            case ArcToCommand():
                return self._handle_arc_to(
                    ctx,
                    cmd,
                    (x, adjusted_y),
                    cut_color,
                    zero_power_color,
                    show_zero_power_moves,
                )
            case _:
                # Ignore unsupported operations, return previous point
                return prev_point_2d

    def _setup_cairo_context(
        self,
        ctx: cairo.Context,
        scale: Tuple[float, float],
        drawable_height: Optional[float],
    ) -> float:
        """
        Calculates Y-axis inversion offset and configures the Cairo context.
        Returns the calculated `ymax` for coordinate inversion.
        """
        scale_x, scale_y = scale
        target_surface = ctx.get_target()

        if isinstance(target_surface, cairo.RecordingSurface):
            if drawable_height is not None:
                ymax = drawable_height
            else:
                extents = target_surface.get_extents()
                ymax = extents[3] if extents else 0
        else:
            height_px = (
                drawable_height
                if drawable_height is not None
                else target_surface.get_height()
            )
            ymax = height_px / scale_y

        # Apply coordinate scaling and line width
        ctx.scale(scale_x, scale_y)
        ctx.set_hairline(True)
        ctx.move_to(0, ymax)
        return ymax

    def _handle_move_to(
        self,
        ctx: cairo.Context,
        cmd: MoveToCommand,
        adjusted_end: Tuple[float, float],
        prev_point_2d: Tuple[float, float],
        show_travel_moves: bool,
        travel_color: Tuple[float, float, float],
    ) -> Tuple[float, float]:
        """Handles a MoveTo command, optionally drawing the travel path."""
        x, adjusted_y = adjusted_end
        if show_travel_moves:
            ctx.set_source_rgb(*travel_color)
            ctx.move_to(*prev_point_2d)
            ctx.line_to(x, adjusted_y)
            ctx.stroke()

        ctx.move_to(x, adjusted_y)
        return x, adjusted_y

    def _handle_scanline(
        self,
        ctx: cairo.Context,
        cmd: ScanLinePowerCommand,
        ymax: float,
        prev_point_2d: Tuple[float, float],
        cut_color: Tuple[float, float, float],
        zero_power_color: Tuple[float, float, float],
        show_zero_power_moves: bool,
    ) -> Tuple[float, float]:
        """
        Handles a ScanLinePowerCommand by splitting it into chunks of
        zero-power and non-zero-power segments and drawing them accordingly.
        """
        if not cmd.power_values:
            return prev_point_2d

        # A scanline is a distinct operation; stroke any preceding path first.
        ctx.set_source_rgb(*cut_color)
        ctx.stroke()

        start_x, start_y = prev_point_2d
        end_x, end_y, _ = cmd.end
        cairo_end_y = ymax - end_y

        # Optimization: Handle case where entire scanline is at zero power
        if all(p == 0 for p in cmd.power_values):
            if show_zero_power_moves:
                ctx.set_source_rgb(*zero_power_color)
                ctx.move_to(start_x, start_y)
                ctx.line_to(end_x, cairo_end_y)
                ctx.stroke()
            ctx.move_to(end_x, cairo_end_y)
            return end_x, cairo_end_y

        # Deconstruct scanline into zero and non-zero power chunks
        p_start_vec = (start_x, start_y)
        line_vec = (end_x - start_x, cairo_end_y - start_y)
        num_steps = len(cmd.power_values)

        if num_steps == 0:
            ctx.move_to(end_x, cairo_end_y)
            return end_x, cairo_end_y

        chunk_start_idx = 0
        is_zero_chunk = cmd.power_values[0] == 0

        for i in range(1, num_steps):
            is_current_zero = cmd.power_values[i] == 0
            if is_current_zero != is_zero_chunk:
                # End of a chunk. Process it.
                self._draw_scanline_chunk(
                    ctx,
                    p_start_vec,
                    line_vec,
                    num_steps,
                    chunk_start_idx,
                    i,
                    cmd.power_values[chunk_start_idx:i],
                    is_zero_chunk,
                    zero_power_color,
                    show_zero_power_moves,
                )
                # Start a new chunk
                chunk_start_idx = i
                is_zero_chunk = is_current_zero

        # Process the final chunk
        self._draw_scanline_chunk(
            ctx,
            p_start_vec,
            line_vec,
            num_steps,
            chunk_start_idx,
            num_steps,
            cmd.power_values[chunk_start_idx:num_steps],
            is_zero_chunk,
            zero_power_color,
            show_zero_power_moves,
        )

        ctx.move_to(end_x, cairo_end_y)
        return end_x, cairo_end_y

    def _draw_scanline_chunk(
        self,
        ctx: cairo.Context,
        p_start_vec: Tuple[float, float],
        line_vec: Tuple[float, float],
        total_steps: int,
        start_idx: int,
        end_idx: int,
        power_slice: bytes,
        is_zero_chunk: bool,
        zero_power_color: Tuple[float, float, float],
        show_zero_power_moves: bool,
    ):
        """Draws a single segment (chunk) of a scanline."""
        if start_idx >= end_idx:
            return

        # Calculate chunk geometry
        t_start = start_idx / total_steps
        t_end = end_idx / total_steps

        chunk_start_pt = (
            p_start_vec[0] + t_start * line_vec[0],
            p_start_vec[1] + t_start * line_vec[1],
        )
        chunk_end_pt = (
            p_start_vec[0] + t_end * line_vec[0],
            p_start_vec[1] + t_end * line_vec[1],
        )

        if is_zero_chunk:
            if show_zero_power_moves:
                ctx.new_path()
                ctx.move_to(*chunk_start_pt)
                ctx.line_to(*chunk_end_pt)
                ctx.set_source_rgb(*zero_power_color)
                ctx.stroke()
        else:  # is non-zero chunk
            grad = cairo.LinearGradient(
                chunk_start_pt[0],
                chunk_start_pt[1],
                chunk_end_pt[0],
                chunk_end_pt[1],
            )
            num_chunk_steps = len(power_slice)
            last_power = -1

            for i, power in enumerate(power_slice):
                if power == last_power:
                    continue

                if i > 0 and num_chunk_steps > 1:
                    p_old = 1.0 - (last_power / 100.0)
                    offset_old = (i / num_chunk_steps) - 1e-9
                    grad.add_color_stop_rgba(
                        offset_old, p_old, p_old, p_old, 1.0
                    )

                p_new = 1.0 - (power / 100.0)
                offset_new = (
                    i / num_chunk_steps if num_chunk_steps > 0 else 0.0
                )
                grad.add_color_stop_rgba(offset_new, p_new, p_new, p_new, 1.0)
                last_power = power

            if last_power != -1:
                p_final = 1.0 - (last_power / 100.0)
                grad.add_color_stop_rgba(1.0, p_final, p_final, p_final, 1.0)

            ctx.new_path()
            ctx.move_to(*chunk_start_pt)
            ctx.line_to(*chunk_end_pt)
            ctx.set_source(grad)
            ctx.stroke()

    def _handle_arc_to(
        self,
        ctx: cairo.Context,
        cmd: ArcToCommand,
        adjusted_end: Tuple[float, float],
        cut_color: Tuple[float, float, float],
        zero_power_color: Tuple[float, float, float],
        show_zero_power_moves: bool,
    ) -> Tuple[float, float]:
        """
        Handles an ArcTo command by calculating geometry and drawing the arc.
        """
        start_x, start_y = ctx.get_current_point()
        x, adjusted_y = adjusted_end
        is_zero_power = cmd.state is not None and cmd.state.power == 0

        # Stroke any preceding line segments before drawing the arc
        ctx.set_source_rgb(*cut_color)
        ctx.stroke()

        if is_zero_power and not show_zero_power_moves:
            ctx.move_to(x, adjusted_y)
            return x, adjusted_y

        arc_color = zero_power_color if is_zero_power else cut_color
        ctx.set_source_rgb(*arc_color)

        # Calculate arc geometry in the Y-down Cairo coordinate system
        i, j = cmd.center_offset
        center_x = start_x + i
        center_y = start_y - j  # Invert relative offset for Y-down

        radius = math.dist((start_x, start_y), (center_x, center_y))
        angle1 = math.atan2(start_y - center_y, start_x - center_x)
        angle2 = math.atan2(adjusted_y - center_y, x - center_x)

        # To draw a CCW arc (clockwise=False) on a Y-flipped canvas,
        # we must use Cairo's CW function (arc_negative), and vice-versa.
        if cmd.clockwise:
            ctx.arc(center_x, center_y, radius, angle1, angle2)
        else:
            ctx.arc_negative(center_x, center_y, radius, angle1, angle2)

        ctx.stroke()
        ctx.move_to(x, adjusted_y)
        return x, adjusted_y
