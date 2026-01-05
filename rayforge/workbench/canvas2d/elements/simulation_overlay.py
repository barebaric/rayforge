"""Preview overlay element for rendering operations playback on the canvas."""

import cairo
import numpy as np
from typing import Dict, Optional, Tuple
from gi.repository import Gdk
from ....core.ops import Ops, State
from ....core.ops.commands import ArcToCommand, ScanLinePowerCommand
from ...canvas.element import CanvasElement
from ....core.geo.linearize import linearize_arc


def speed_to_heatmap_color(
    speed: float,
    min_speed: float,
    max_speed: float,
) -> Tuple[float, float, float]:
    """
    Converts speed to RGB heatmap color.
    Blue (slow) → Cyan → Green → Yellow → Red (fast)
    """
    if max_speed <= min_speed:
        return (0.0, 1.0, 0.0)  # Green as default

    # Normalize speed to [0, 1]
    normalized = (speed - min_speed) / (max_speed - min_speed)
    normalized = max(0.0, min(1.0, normalized))  # Clamp

    # Heatmap: Blue → Cyan → Green → Yellow → Red (5 zones)
    if normalized < 0.25:
        # Blue to Cyan
        t = normalized / 0.25
        return (0.0, t, 1.0)
    elif normalized < 0.5:
        # Cyan to Green
        t = (normalized - 0.25) / 0.25
        return (0.0, 1.0, 1.0 - t)
    elif normalized < 0.75:
        # Green to Yellow
        t = (normalized - 0.5) / 0.25
        return (t, 1.0, 0.0)
    else:
        # Yellow to Red
        t = (normalized - 0.75) / 0.25
        return (1.0, 1.0 - t, 0.0)


class OpsTimeline:
    """Converts Ops into a timeline of steps with state tracking."""

    def __init__(self, ops: Optional[Ops] = None):
        self.ops = ops
        self.steps = []
        self.speed_range = (0.0, 1000.0)  # (min, max) from all operations
        self.op_to_timeline_index: Dict[int, int] = {}
        if ops:
            self._rebuild_timeline()

    def set_ops(self, ops: Optional[Ops]):
        """Updates the operations and rebuilds the timeline."""
        self.ops = ops
        self._rebuild_timeline()

    def _rebuild_timeline(self):
        """Builds timeline and calculates speed range from ALL operations."""
        self.steps = []
        self.op_to_timeline_index = {}
        if not self.ops or self.ops.is_empty():
            self.speed_range = (0.0, 1000.0)
            return

        current_state = State(power=0.0)
        current_pos = (0.0, 0.0, 0.0)
        speeds = []
        timeline_idx = 0

        for op_idx, cmd in enumerate(self.ops):
            # Update state if this is a state command
            if cmd.is_state_command():
                cmd.apply_to_state(current_state)
            # Only add movement commands to the timeline steps
            elif not cmd.is_marker():
                # Store command with state and starting position
                self.steps.append(
                    (cmd, State(**current_state.__dict__), current_pos)
                )
                self.op_to_timeline_index[op_idx] = timeline_idx
                timeline_idx += 1

                # Update current_pos
                if cmd.end is not None:
                    current_pos = cmd.end

                # Track speeds for range calculation
                if current_state.cut_speed is not None:
                    speeds.append(current_state.cut_speed)

        # Calculate speed range from all operations ONCE
        if speeds:
            self.speed_range = (min(speeds), max(speeds))
        else:
            self.speed_range = (0.0, 1000.0)

    def get_step_count(self) -> int:
        """Returns the total number of steps in the timeline."""
        return len(self.steps)

    def get_steps_up_to(self, step_index: int):
        """Returns all steps from start up to and including step_index."""
        if step_index < 0:
            return []
        return self.steps[: step_index + 1]


class SimulationOverlay(CanvasElement):
    """
    Canvas element that renders the operations simulation with heatmap colors
    and power-based transparency.
    """

    def __init__(self, work_area_size: Tuple[float, float], **kwargs):
        width_mm, height_mm = work_area_size
        super().__init__(
            x=0.0,
            y=0.0,
            width=width_mm,
            height=height_mm,
            buffered=False,
            **kwargs,
        )
        self.selectable = False
        self.clip = False

        # Timeline and playback state
        self.timeline = OpsTimeline()
        self.current_step = 0

        # Work area boundary (drawn as reference)
        self.work_area_bounds = (0.0, 0.0, width_mm, height_mm)

    def set_ops(self, ops: Optional[Ops]):
        """Updates the operations to preview."""
        self.timeline.set_ops(ops)
        self.current_step = 0
        self.mark_dirty()

    def set_step(self, step: int):
        """Sets the current playback step."""
        self.current_step = max(
            0, min(step, self.timeline.get_step_count() - 1)
        )
        self.mark_dirty()

    def get_step_count(self) -> int:
        """Returns total number of steps."""
        return self.timeline.get_step_count()

    def get_speed_range(self) -> Tuple[float, float]:
        """Returns the (min, max) speed range for heatmap."""
        return self.timeline.speed_range

    def get_current_position(self) -> Optional[Tuple[float, float]]:
        """Returns the current laser head position (x, y) in mm."""
        if not self.timeline.steps or self.current_step < 0:
            return None

        if self.current_step >= len(self.timeline.steps):
            # Return the last position
            last_cmd, _, _ = self.timeline.steps[-1]
            if hasattr(last_cmd, "end") and last_cmd.end is not None:
                return (last_cmd.end[0], last_cmd.end[1])
            return None

        # Get position at current step
        cmd, _, _ = self.timeline.steps[self.current_step]
        if hasattr(cmd, "end") and cmd.end is not None:
            return (cmd.end[0], cmd.end[1])
        return None

    def get_current_state(self) -> Optional[State]:
        """Returns the state at the current playback step."""
        if not self.timeline.steps or self.current_step < 0:
            return None

        if self.current_step >= len(self.timeline.steps):
            # Return the last state
            _, last_state, _ = self.timeline.steps[-1]
            return last_state

        # Get state at current step
        _, state, _ = self.timeline.steps[self.current_step]
        return state

    def draw(self, ctx: cairo.Context):
        """Renders the preview operations up to current_step."""
        if not self.timeline.steps:
            return

        steps = self.timeline.get_steps_up_to(self.current_step)
        if not steps:
            return

        min_speed, max_speed = self.timeline.speed_range

        # Draw each operation with heatmap color and power transparency
        for cmd, state, start_pos in steps:
            ctx.set_line_width(0.2)
            ctx.set_dash([])

            if cmd.is_travel_command():
                ctx.set_source_rgba(0.5, 0.5, 0.5, 0.2)  # Faint gray
                if cmd.end:
                    ctx.move_to(start_pos[0], start_pos[1])
                    ctx.line_to(cmd.end[0], cmd.end[1])
                    ctx.stroke()
                continue

            # Get speed and power for cutting moves.
            speed = state.cut_speed if state.cut_speed is not None else 1000.0
            power = state.power if state.power is not None else 1.0

            # Speed is visualized as color
            r, g, b = speed_to_heatmap_color(speed, min_speed, max_speed)

            # Maps power [0.0-1.0] to alpha
            alpha = 0.1 + power * 0.9

            ctx.set_source_rgba(r, g, b, alpha)

            if isinstance(cmd, ArcToCommand):
                for seg_start, seg_end in linearize_arc(cmd, start_pos):
                    ctx.move_to(seg_start[0], seg_start[1])
                    ctx.line_to(seg_end[0], seg_end[1])
                    ctx.stroke()
            elif isinstance(cmd, ScanLinePowerCommand):
                self._draw_scanline(ctx, cmd, start_pos, state)
            elif cmd.end:  # This handles LineToCommand
                ctx.move_to(start_pos[0], start_pos[1])
                ctx.line_to(cmd.end[0], cmd.end[1])
                ctx.stroke()

        # Draw laser head position indicator
        current_pos = self.get_current_position()
        if current_pos:
            self._draw_laser_head(ctx, current_pos)

    def _draw_scanline(
        self,
        ctx: cairo.Context,
        cmd: ScanLinePowerCommand,
        start_pos: tuple,
        state: State,
    ):
        """Draws a ScanLinePowerCommand as a series of colored segments."""
        if cmd.end is None:
            return

        p_start = np.array(start_pos[:2], dtype=np.float32)
        p_end = np.array(cmd.end[:2], dtype=np.float32)
        line_vec = p_end - p_start
        num_steps = len(cmd.power_values)
        if num_steps == 0:
            return

        min_s, max_s = self.timeline.speed_range

        # Speed -> Color
        r, g, b = speed_to_heatmap_color(state.cut_speed or 0.0, min_s, max_s)

        for i, power_byte in enumerate(cmd.power_values):
            if power_byte == 0:
                continue

            # Power -> Transparency (Opacity)
            alpha = 0.1 + (power_byte / 255.0) * 0.9
            t_start, t_end = i / num_steps, (i + 1) / num_steps
            seg_start_pt = p_start + t_start * line_vec
            seg_end_pt = p_start + t_end * line_vec

            ctx.set_source_rgba(r, g, b, alpha)
            ctx.move_to(seg_start_pt[0], seg_start_pt[1])
            ctx.line_to(seg_end_pt[0], seg_end_pt[1])
            ctx.stroke()

    def _draw_laser_head(self, ctx: cairo.Context, pos: Tuple[float, float]):
        """Draws the laser head indicator at the given position in mm."""
        x, y = pos

        # Draw a crosshair with circle
        ctx.set_source_rgba(1.0, 0.0, 0.0, 0.8)  # Red with transparency
        ctx.set_line_width(0.2)

        # Circle (3mm radius)
        ctx.arc(x, y, 3.0, 0, 2 * 3.14159)
        ctx.stroke()

        # Crosshair lines (6mm each direction)
        ctx.move_to(x - 6.0, y)
        ctx.line_to(x + 6.0, y)
        ctx.stroke()

        ctx.move_to(x, y - 6.0)
        ctx.line_to(x, y + 6.0)
        ctx.stroke()

        # Center dot
        ctx.arc(x, y, 0.5, 0, 2 * 3.14159)
        ctx.fill()

    def draw_overlay(self, ctx: cairo.Context):
        """
        Draws overlay elements in pixel space (after view transform).
        This renders the legend and other UI elements.
        """
        if not self.canvas:
            return

        # Draw heatmap legend
        self._draw_legend(ctx)

    def _draw_legend(self, ctx: cairo.Context):
        """Draws the speed heatmap legend in pixel space."""
        if not self.canvas:
            return

        min_speed, max_speed = self.timeline.speed_range

        # Legend position and size (in pixels)
        legend_x = 20
        legend_y = 20
        legend_width = 30
        legend_height = 200
        num_stops = 50

        ctx.save()

        # Draw gradient bar
        for i in range(num_stops):
            fraction = i / num_stops
            speed = min_speed + fraction * (max_speed - min_speed)
            r, g, b = speed_to_heatmap_color(speed, min_speed, max_speed)

            ctx.set_source_rgb(r, g, b)
            segment_height = legend_height / num_stops
            y = legend_y + legend_height - (i + 1) * segment_height
            ctx.rectangle(legend_x, y, legend_width, segment_height)
            ctx.fill()

        # Get theme colors
        # Use GtkWidget.get_color() to retrieve the theme's foreground color.
        # This works for both text and high-contrast borders.
        fg_color = self.canvas.get_color()

        # Draw border around legend using the foreground color
        Gdk.cairo_set_source_rgba(ctx, fg_color)
        ctx.set_line_width(1)
        ctx.rectangle(legend_x, legend_y, legend_width, legend_height)
        ctx.stroke()

        # Draw labels using the foreground color
        Gdk.cairo_set_source_rgba(ctx, fg_color)
        ctx.select_font_face(
            "sans-serif",
            cairo.FONT_SLANT_NORMAL,
            cairo.FONT_WEIGHT_NORMAL,
        )
        ctx.set_font_size(12)

        # Max speed label (top)
        ctx.move_to(legend_x + legend_width + 5, legend_y + 12)
        ctx.show_text(f"{max_speed:.0f}")

        # Min speed label (bottom)
        ctx.move_to(legend_x + legend_width + 5, legend_y + legend_height)
        ctx.show_text(f"{min_speed:.0f}")

        # "Speed" label
        ctx.move_to(legend_x, legend_y - 5)
        ctx.show_text("Speed (mm/min)")

        ctx.restore()
