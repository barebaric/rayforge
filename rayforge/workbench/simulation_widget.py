"""
Operations Preview Widget

Provides a video-player-like interface for previewing laser operations
execution. Shows operations being drawn progressively with power
represented as transparency.
"""

from typing import Optional, List, Tuple
import cairo
from gi.repository import Gtk, GLib
from ..core.ops import Ops
from ..core.ops.commands import (
    Command,
    LineToCommand,
    MoveToCommand,
    State,
    SetPowerCommand,
    SetCutSpeedCommand,
)


def speed_to_heatmap_color(
    speed: float,
    min_speed: float,
    max_speed: float,
) -> Tuple[float, float, float]:
    """
    Converts speed to RGB heatmap color.

    Blue (slow) → Cyan → Green → Yellow → Red (fast)

    Args:
        speed: Current speed value
        min_speed: Minimum speed in range
        max_speed: Maximum speed in range

    Returns:
        (r, g, b) tuple with values in [0, 1]
    """
    if max_speed <= min_speed:
        return (0.0, 1.0, 0.0)  # Green as default

    # Normalize speed to [0, 1]
    normalized = (speed - min_speed) / (max_speed - min_speed)
    normalized = max(0.0, min(1.0, normalized))  # Clamp

    # Heatmap: Blue → Cyan → Green → Yellow → Red
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
    """
    Manages the timeline of operations for preview playback.

    Breaks down operations into discrete drawing steps with associated
    power/speed state for rendering.
    """

    def __init__(self, ops: Optional[Ops] = None):
        self.ops = ops
        self.steps: List[
            Tuple[Command, State, Tuple[float, float, float]]
        ] = []
        self.speed_range = (0.0, 1000.0)  # Calculated during timeline build
        self._rebuild_timeline()

    def set_ops(self, ops: Optional[Ops]):
        """Updates the ops and rebuilds the timeline."""
        self.ops = ops
        self._rebuild_timeline()

    def _rebuild_timeline(self):
        """
        Builds a list of (command, state, start_point) tuples
        representing each drawing step in the operations.
        Also calculates the speed range across all operations.
        """
        self.steps = []
        if not self.ops or not self.ops.commands:
            self.speed_range = (0.0, 1000.0)
            return

        current_state = State()
        current_pos = (0.0, 0.0, 0.0)
        speeds = []

        for cmd in self.ops.commands:
            # Update state if this is a state command
            if isinstance(cmd, SetPowerCommand):
                current_state.power = cmd.power
            elif isinstance(cmd, SetCutSpeedCommand):
                current_state.cut_speed = cmd.speed
            elif hasattr(cmd, 'apply_to_state'):
                cmd.apply_to_state(current_state)

            # For moving commands, add them to the timeline
            if hasattr(cmd, 'end') and cmd.end is not None:
                # Store command with state and starting position
                self.steps.append(
                    (cmd, State(**current_state.__dict__), current_pos)
                )
                current_pos = cmd.end

                # Track speeds for range calculation
                if current_state.cut_speed is not None:
                    speeds.append(current_state.cut_speed)

        # Calculate speed range from all operations
        if speeds:
            self.speed_range = (min(speeds), max(speeds))
        else:
            self.speed_range = (0.0, 1000.0)

    def get_step_count(self) -> int:
        """Returns the total number of steps in the timeline."""
        return len(self.steps)

    def get_steps_up_to(
        self, index: int
    ) -> List[Tuple[Command, State, Tuple[float, float, float]]]:
        """Returns all steps up to and including the given index."""
        if index < 0:
            return []
        return self.steps[:index + 1]


class PreviewRenderer:
    """
    Renders operations with speed as heatmap color and power as transparency.
    """

    def __init__(self, width: int, height: int):
        self.width = width
        self.height = height
        self.surface = cairo.ImageSurface(
            cairo.FORMAT_ARGB32,
            width,
            height,
        )
        self.bounds = (0.0, 0.0, 100.0, 100.0)  # (min_x, min_y, max_x, max_y)
        self.speed_range = (0.0, 1000.0)  # Will be calculated from steps
        self.work_area = None  # (width, height) in mm, if set

    def set_bounds(
        self, min_x: float, min_y: float, max_x: float, max_y: float
    ):
        """Sets the coordinate bounds for the operations."""
        self.bounds = (min_x, min_y, max_x, max_y)

    def set_work_area(self, width_mm: float, height_mm: float):
        """Sets the work area dimensions to display."""
        self.work_area = (width_mm, height_mm)

    def resize(self, width: int, height: int):
        """Resizes the rendering surface."""
        if width != self.width or height != self.height:
            self.width = width
            self.height = height
            self.surface = cairo.ImageSurface(
                cairo.FORMAT_ARGB32, width, height
            )

    def render(
        self, steps: List[Tuple[Command, State, Tuple[float, float, float]]]
    ) -> cairo.ImageSurface:
        """
        Renders the given steps onto the surface.

        Args:
            steps: List of (command, state, start_point) tuples

        Returns:
            Cairo ImageSurface with the rendered preview
        """
        ctx = cairo.Context(self.surface)

        # Clear surface with white background
        ctx.set_source_rgb(1, 1, 1)
        ctx.paint()

        # Reserve space for legend on right side
        legend_width = 80
        available_width = self.width - legend_width

        # Calculate transform to fit operations/work area
        min_x, min_y, max_x, max_y = self.bounds

        # If work area is set, expand bounds to include it
        if self.work_area:
            work_w, work_h = self.work_area
            max_x = max(max_x, work_w)
            max_y = max(max_y, work_h)
            min_x = min(min_x, 0.0)
            min_y = min(min_y, 0.0)

        width_mm = max_x - min_x
        height_mm = max_y - min_y

        if width_mm <= 0 or height_mm <= 0:
            return self.surface

        # Calculate scale to fit with padding
        padding = 20
        scale_x = (available_width - 2 * padding) / width_mm
        scale_y = (self.height - 2 * padding) / height_mm
        scale = min(scale_x, scale_y)

        # Center the content
        offset_x = (
            padding + (available_width - 2 * padding - width_mm * scale) / 2
        )
        offset_y = (
            padding + (self.height - 2 * padding - height_mm * scale) / 2
        )

        # Save context for later (legend drawing)
        ctx.save()

        # Apply transform for operations
        ctx.translate(offset_x, offset_y)
        ctx.scale(scale, -scale)  # Flip Y axis
        ctx.translate(-min_x, -max_y)  # Position origin

        ctx.set_line_width(0.1)  # Line width in mm

        # Speed range is pre-calculated and passed in, not recalculated
        # per frame

        # Draw each step
        for cmd, state, start_pos in steps:
            # Calculate transparency based on power (0-100%)
            # Full power (100) = fully opaque (alpha=1.0)
            # No power (0) = nearly transparent (alpha=0.1)
            alpha = 0.1 + (state.power / 100.0) * 0.9

            if isinstance(cmd, LineToCommand):
                # Cutting move - heatmap color by speed, alpha by power
                speed = state.cut_speed if state.cut_speed else 0.0
                r, g, b = speed_to_heatmap_color(
                    speed, self.speed_range[0], self.speed_range[1]
                )
                ctx.set_source_rgba(r, g, b, alpha)
                ctx.move_to(start_pos[0], start_pos[1])
                ctx.line_to(cmd.end[0], cmd.end[1])
                ctx.stroke()
            elif isinstance(cmd, MoveToCommand):
                # Travel move - gray, very transparent
                ctx.set_source_rgba(0.5, 0.5, 0.5, 0.2)
                ctx.move_to(start_pos[0], start_pos[1])
                ctx.line_to(cmd.end[0], cmd.end[1])
                ctx.stroke()

        # Draw work area boundary if set
        if self.work_area:
            work_w, work_h = self.work_area
            ctx.set_source_rgb(0.3, 0.3, 0.3)
            ctx.set_line_width(0.5)
            ctx.rectangle(0, 0, work_w, work_h)
            ctx.stroke()

        # Restore context for legend drawing in pixel space
        ctx.restore()

        # Draw heatmap legend
        self._draw_heatmap_legend(ctx, legend_width)

        return self.surface

    def _draw_heatmap_legend(self, ctx: cairo.Context, legend_width: int):
        """Draws a heatmap legend showing speed → color mapping."""
        legend_x = self.width - legend_width + 10
        legend_y = 40
        legend_height = self.height - 80
        bar_width = 30

        # Title
        ctx.set_source_rgb(0, 0, 0)
        ctx.select_font_face(
            "Sans", cairo.FONT_SLANT_NORMAL, cairo.FONT_WEIGHT_BOLD
        )
        ctx.set_font_size(12)
        ctx.move_to(legend_x, legend_y - 15)
        ctx.show_text("Speed")

        # Draw color gradient bar
        steps = 50
        for i in range(steps):
            y = legend_y + (i / steps) * legend_height
            height_segment = legend_height / steps

            # Calculate speed for this position (top = max, bottom = min)
            speed_fraction = 1.0 - (i / steps)
            speed = (
                self.speed_range[0] +
                speed_fraction * (self.speed_range[1] - self.speed_range[0])
            )

            # Get color
            r, g, b = speed_to_heatmap_color(
                speed, self.speed_range[0], self.speed_range[1]
            )

            ctx.set_source_rgb(r, g, b)
            ctx.rectangle(legend_x, y, bar_width, height_segment + 1)
            ctx.fill()

        # Draw border around bar
        ctx.set_source_rgb(0, 0, 0)
        ctx.set_line_width(1)
        ctx.rectangle(legend_x, legend_y, bar_width, legend_height)
        ctx.stroke()

        # Draw labels
        ctx.set_source_rgb(0, 0, 0)
        ctx.select_font_face(
            "Sans", cairo.FONT_SLANT_NORMAL, cairo.FONT_WEIGHT_NORMAL
        )
        ctx.set_font_size(10)

        # Max speed at top
        max_label = f"{int(self.speed_range[1])}"
        ctx.move_to(legend_x + bar_width + 5, legend_y + 5)
        ctx.show_text(max_label)

        # Min speed at bottom
        min_label = f"{int(self.speed_range[0])}"
        ctx.move_to(legend_x + bar_width + 5, legend_y + legend_height)
        ctx.show_text(min_label)

        # Unit label
        ctx.set_font_size(9)
        ctx.move_to(legend_x, legend_y + legend_height + 15)
        ctx.show_text("mm/min")


class PreviewWidget(Gtk.Box):
    """
    A widget that provides video-player-like controls for
    previewing laser operations execution.
    """

    def __init__(self):
        super().__init__(orientation=Gtk.Orientation.VERTICAL, spacing=6)

        self.timeline = OpsTimeline()
        self.current_step = 0
        self.is_playing = False
        self.playback_timer = None

        # Build UI
        self._build_ui()

    def _build_ui(self):
        """Builds the preview widget UI."""
        # Drawing area for preview
        self.drawing_area = Gtk.DrawingArea()
        self.drawing_area.set_size_request(400, 300)
        self.drawing_area.set_hexpand(True)
        self.drawing_area.set_vexpand(True)
        self.drawing_area.set_draw_func(self._on_draw)
        self.append(self.drawing_area)

        # Create renderer
        self.renderer = PreviewRenderer(400, 300)

        # Controls box
        controls_box = Gtk.Box(
            orientation=Gtk.Orientation.HORIZONTAL, spacing=6
        )
        controls_box.set_margin_start(6)
        controls_box.set_margin_end(6)
        controls_box.set_margin_bottom(6)
        self.append(controls_box)

        # Play/Pause button
        self.play_button = Gtk.Button()
        self.play_button.set_icon_name("media-playback-start-symbolic")
        self.play_button.connect("clicked", self._on_play_pause)
        controls_box.append(self.play_button)

        # Progress slider
        self.slider = Gtk.Scale.new_with_range(
            Gtk.Orientation.HORIZONTAL, 0, 100, 1
        )
        self.slider.set_hexpand(True)
        self.slider.set_draw_value(False)
        self.slider.connect("value-changed", self._on_slider_changed)
        controls_box.append(self.slider)

        # Progress label
        self.progress_label = Gtk.Label(label="0 / 0")
        controls_box.append(self.progress_label)

    def set_ops(
        self,
        ops: Optional[Ops],
        bounds: Optional[Tuple[float, float, float, float]] = None
    ):
        """
        Sets the operations to preview.

        Args:
            ops: The operations to preview
            bounds: (min_x, min_y, max_x, max_y) bounds
        """
        self.timeline.set_ops(ops)
        self.current_step = 0
        self._stop_playback()

        # Update slider range
        step_count = self.timeline.get_step_count()
        self.slider.set_range(0, max(0, step_count - 1))
        self.slider.set_value(0)

        # Set bounds if provided
        if bounds:
            self.renderer.set_bounds(*bounds)

        # Update renderer speed range from timeline
        self.renderer.speed_range = self.timeline.speed_range

        # Update display
        self._update_progress_label()
        self.drawing_area.queue_draw()

    def _on_draw(self, area, ctx, width, height):
        """Handles drawing the preview."""
        # Resize renderer if needed
        self.renderer.resize(width, height)

        # Get steps up to current position
        steps = self.timeline.get_steps_up_to(self.current_step)

        # Render and draw
        surface = self.renderer.render(steps)
        ctx.set_source_surface(surface, 0, 0)
        ctx.paint()

    def _on_play_pause(self, button):
        """Handles play/pause button click."""
        if self.is_playing:
            self._stop_playback()
        else:
            self._start_playback()

    def _on_slider_changed(self, slider):
        """Handles manual slider movement."""
        new_step = int(slider.get_value())
        if new_step != self.current_step:
            self.current_step = new_step
            self._update_progress_label()
            self.drawing_area.queue_draw()

    def _start_playback(self):
        """Starts automatic playback."""
        self.is_playing = True
        self.play_button.set_icon_name("media-playback-pause-symbolic")

        # Start timer for playback (30 FPS)
        self.playback_timer = GLib.timeout_add(33, self._on_playback_tick)

    def _stop_playback(self):
        """Stops automatic playback."""
        self.is_playing = False
        self.play_button.set_icon_name("media-playback-start-symbolic")

        if self.playback_timer:
            GLib.source_remove(self.playback_timer)
            self.playback_timer = None

    def _on_playback_tick(self) -> bool:
        """Called periodically during playback."""
        if not self.is_playing:
            return False

        # Advance one step
        max_step = self.timeline.get_step_count() - 1
        if self.current_step < max_step:
            self.current_step += 1
            self.slider.set_value(self.current_step)
            self._update_progress_label()
            self.drawing_area.queue_draw()
            return True
        else:
            # Reached the end
            self._stop_playback()
            return False

    def _update_progress_label(self):
        """Updates the progress label."""
        total = self.timeline.get_step_count()
        current = self.current_step + 1 if total > 0 else 0
        self.progress_label.set_text(f"{current} / {total}")
