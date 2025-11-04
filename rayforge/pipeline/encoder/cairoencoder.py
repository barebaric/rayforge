import math
from typing import Tuple, Optional, Union
import numpy as np
import cairo
import logging
from ...core.ops import (
    Ops,
    MoveToCommand,
    LineToCommand,
    ArcToCommand,
    ScanLinePowerCommand,
    SetPowerCommand,
)
from .base import OpsEncoder
from ...shared.util.colors import ColorSet


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
        colors: ColorSet,
        # Visibility Toggles
        show_cut_moves: bool = True,
        show_engrave_moves: bool = True,
        show_travel_moves: bool = True,
        show_zero_power_moves: bool = True,
        # Other Options
        drawable_height: Optional[float] = None,
    ) -> None:
        """
        Main orchestration method to draw Ops onto a Cairo context.

        Args:
            ops: The operations to encode.
            ctx: The Cairo context to draw on.
            scale: The (x, y) scaling factors.
            colors: A resolved ColorSet with ready-to-use color data.
            show_cut_moves: Whether to draw cut/arc moves.
            show_engrave_moves: Whether to draw engrave/scanline moves.
            show_travel_moves: Whether to draw travel moves.
            show_zero_power_moves: Whether to draw zero-power cut/engrave
              moves.
            drawable_height: Optional explicit height of the drawable area.
        """
        if scale[1] == 0:
            return

        cut_lut = colors.get_lut("cut")
        engrave_lut = colors.get_lut("engrave")
        travel_rgba = colors.get_rgba("travel")
        zero_power_rgba = colors.get_rgba("zero_power")

        ctx.save()
        try:
            ymax = self._setup_cairo_context(ctx, scale, drawable_height)
            logger.debug(f"CairoEncoder started. ymax={ymax}, scale={scale}")

            prev_point_2d = (0.0, ymax)
            current_power = 0.0
            is_first_move = True

            # --- Performance Optimization State ---
            # Tracks the color of the path currently being built.
            # None means no path is active or the color is not set.
            current_path_color: Optional[Tuple] = None
            # Flag to avoid stroking an empty path.
            path_has_content = False

            def flush_path():
                """Strokes the currently buffered path if it has content."""
                nonlocal path_has_content
                if path_has_content:
                    ctx.stroke()
                    path_has_content = False

            for cmd in ops:
                # Handle state change commands first
                if isinstance(cmd, SetPowerCommand):
                    current_power = cmd.power
                    continue

                if cmd.is_marker() or cmd.end is None:
                    continue

                x, y, _ = cmd.end
                adjusted_end = (x, ymax - y)

                match cmd:
                    case MoveToCommand():
                        # A move command breaks any current path being built.
                        flush_path()
                        current_path_color = None

                        # Optionally draw the travel move. This is drawn
                        # immediately as a single segment and is not batched.
                        if show_travel_moves and not is_first_move:
                            ctx.set_source_rgba(*travel_rgba)
                            ctx.move_to(*prev_point_2d)
                            ctx.line_to(*adjusted_end)
                            ctx.stroke()

                        prev_point_2d = adjusted_end
                        is_first_move = False

                    case LineToCommand() | ArcToCommand():
                        is_zero_power = math.isclose(current_power, 0.0)
                        should_draw = (
                            show_zero_power_moves
                            if is_zero_power
                            else show_cut_moves
                        )

                        if not should_draw:
                            # If not drawing, this move still breaks the
                            # current path.
                            flush_path()
                            current_path_color = None
                            prev_point_2d = adjusted_end
                            continue

                        power_idx = min(255, int(current_power * 255.0))
                        required_color = (
                            tuple(cut_lut[power_idx])
                            if not is_zero_power
                            else zero_power_rgba
                        )

                        # If color changes, flush the old path and start a
                        # new one.
                        if required_color != current_path_color:
                            flush_path()
                            ctx.set_source_rgba(*required_color)
                            current_path_color = required_color
                            # Start new subpath at the previous point.
                            ctx.move_to(*prev_point_2d)

                        # Add the command geometry to the current path.
                        if isinstance(cmd, LineToCommand):
                            ctx.line_to(*adjusted_end)
                        else:  # ArcToCommand
                            start_x, start_y = prev_point_2d
                            i, j = cmd.center_offset
                            center_x = start_x + i
                            center_y = start_y - j
                            radius = math.dist(
                                (start_x, start_y), (center_x, center_y)
                            )
                            angle1 = math.atan2(
                                start_y - center_y, start_x - center_x
                            )
                            angle2 = math.atan2(
                                adjusted_end[1] - center_y,
                                adjusted_end[0] - center_x,
                            )
                            if cmd.clockwise:
                                ctx.arc(
                                    center_x, center_y, radius, angle1, angle2
                                )
                            else:
                                ctx.arc_negative(
                                    center_x, center_y, radius, angle1, angle2
                                )

                        path_has_content = True
                        prev_point_2d = adjusted_end

                    case ScanLinePowerCommand():
                        # Scanlines are complex and drawn immediately.
                        # Flush any pending path.
                        flush_path()
                        current_path_color = None

                        prev_point_2d = self._handle_scanline(
                            ctx,
                            cmd,
                            ymax,
                            prev_point_2d,
                            engrave_lut,
                            zero_power_rgba,
                            show_engrave_moves,
                            show_zero_power_moves,
                        )
                        is_first_move = False

                    case _:
                        # Ignore unsupported operations
                        pass

            # After the loop, flush any remaining path.
            flush_path()

        finally:
            ctx.restore()

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
        ctx.set_line_cap(cairo.LINE_CAP_SQUARE)
        return ymax

    def _handle_scanline(
        self,
        ctx: cairo.Context,
        cmd: ScanLinePowerCommand,
        ymax: float,
        prev_point_2d: Tuple[float, float],
        engrave_lut: np.ndarray,
        zero_power_rgba: Tuple[float, ...],
        show_engrave_moves: bool,
        show_zero_power_moves: bool,
    ) -> Tuple[float, float]:
        """
        Handles a ScanLinePowerCommand by splitting it into chunks of
        zero-power and non-zero-power segments and drawing them accordingly.
        """
        if cmd.end is None:
            return prev_point_2d

        end_x, end_y, _ = cmd.end
        adjusted_end = (end_x, ymax - end_y)

        if not cmd.power_values:
            return adjusted_end

        start_x, start_y = prev_point_2d

        # Optimization: Handle case where entire scanline is at zero power
        if all(p == 0 for p in cmd.power_values):
            if show_zero_power_moves:
                ctx.set_source_rgba(*zero_power_rgba)
                ctx.move_to(start_x, start_y)
                ctx.line_to(*adjusted_end)
                ctx.stroke()
            return adjusted_end

        # Deconstruct scanline into zero and non-zero power chunks
        p_start_vec = (start_x, start_y)
        line_vec = (adjusted_end[0] - start_x, adjusted_end[1] - start_y)
        num_steps = len(cmd.power_values)

        if num_steps == 0:
            return adjusted_end

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
                    engrave_lut,
                    zero_power_rgba,
                    show_zero_power_moves,
                    show_engrave_moves,
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
            engrave_lut,
            zero_power_rgba,
            show_zero_power_moves,
            show_engrave_moves,
        )

        return adjusted_end

    def _draw_scanline_chunk(
        self,
        ctx: cairo.Context,
        p_start_vec: Tuple[float, float],
        line_vec: Tuple[float, float],
        total_steps: int,
        start_idx: int,
        end_idx: int,
        power_slice: Union[bytes, bytearray],
        is_zero_chunk: bool,
        engrave_lut: np.ndarray,
        zero_power_rgba: Tuple[float, ...],
        show_zero_power_moves: bool,
        show_engrave_moves: bool,
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
                ctx.move_to(*chunk_start_pt)
                ctx.line_to(*chunk_end_pt)
                ctx.set_source_rgba(*zero_power_rgba)
                ctx.stroke()
        elif show_engrave_moves:  # is non-zero chunk and should be shown
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
                    power_idx_old = min(255, last_power)
                    color_old = engrave_lut[power_idx_old]
                    offset_old = (i / num_chunk_steps) - 1e-9
                    grad.add_color_stop_rgba(offset_old, *color_old)

                power_idx_new = min(255, power)
                color_new = engrave_lut[power_idx_new]
                offset_new = (
                    i / num_chunk_steps if num_chunk_steps > 0 else 0.0
                )
                grad.add_color_stop_rgba(offset_new, *color_new)
                last_power = power

            if last_power != -1:
                power_idx_final = min(255, last_power)
                color_final = engrave_lut[power_idx_final]
                grad.add_color_stop_rgba(1.0, *color_final)

            ctx.move_to(*chunk_start_pt)
            ctx.line_to(*chunk_end_pt)
            ctx.set_source(grad)
            ctx.stroke()
