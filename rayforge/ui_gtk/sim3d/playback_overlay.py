import logging
from gettext import gettext as _
from typing import Protocol

from blinker import Signal
from gi.repository import GLib, Gtk
from raygeo.ops import Ops

from ..icons import get_icon
from ..shared.gtk import apply_css

logger = logging.getLogger(__name__)


class OpMapLike(Protocol):
    """Surface of :class:`MachineCodeOpMap` used by the overlay.

    Implemented by ``MachineCodeOpMap``; backed by compact numpy
    arrays that already exist per job, so holding a reference here
    adds no per-job memory.
    """

    @property
    def op_count(self) -> int: ...

    @property
    def line_count(self) -> int: ...

    def span_for_op(self, op_index: int) -> tuple[int, int]: ...

    def op_for_line(self, line_idx: int) -> int | None: ...


class PlaybackPlayer(Protocol):
    """Minimal OpPlayer surface required by the playback overlay."""

    ops: Ops

    @property
    def current_index(self) -> int: ...

    @property
    def is_finished(self) -> bool: ...

    @property
    def sim_time(self) -> float: ...

    def seek(self, index: int) -> None: ...

    def seek_to_fraction(self, fraction: float) -> None: ...

    def find_index_at_sim_time(self, t: float) -> int: ...

    def get_cumulative_time(self, idx: int) -> float: ...

    def set_sim_time(self, t: float) -> None: ...

    def set_playhead(self, index: int) -> None: ...

    def set_progress_anchor(self, completed: int, t: float) -> None: ...

    def sync_state_to_playhead(self) -> None: ...

    def playback_progress(self) -> tuple[int, float]: ...


SPEED_OPTIONS = [1, 2, 4, 8, 16, 32, 64]

# Wall-clock interval between playback ticks (~60 fps, matching the
# display frame rate). The simulated clock advances by this amount per
# tick, scaled by the speed multiplier. Every tick queues a render so
# the interpolated playhead is redrawn continuously, not only when the
# slider value changes.
TICK_SECONDS = 1.0 / 60.0

# Wall-clock span of the step-button animation (~0.2 s). Each manual
# step plays out over this fixed number of ticks, regardless of the
# command's simulated length, so the playhead glides to the next
# command instead of jumping.
STEP_ANIMATION_TICKS = 12
STEP_ANIMATION_SECONDS = STEP_ANIMATION_TICKS * TICK_SECONDS

css = """
.playback-overlay {
    background-color: alpha(@theme_bg_color, 0.75);
    border-radius: 6px;
    padding: 3px 6px;
}
.playback-overlay scale {
    min-width: 250px;
}
.speed-button {
    min-width: 36px;
    padding: 2px 6px;
    font-size: small;
}
"""


class PlaybackOverlay(Gtk.Box):
    """
    Playback controls (play/pause button + slider + speed button)
    shown as a bar below the 3D canvas. The slider spans the command
    indices, matching the G-code viewer; the underlying animation
    advances by simulated machine time so motion follows feed rates.
    Play starts a ~60 fps timer that drives the playhead.
    """

    step_changed = Signal()

    def __init__(self, **kwargs):
        super().__init__(
            orientation=Gtk.Orientation.HORIZONTAL,
            spacing=6,
            **kwargs,
        )
        apply_css(css)
        self.add_css_class("playback-overlay")
        self.set_halign(Gtk.Align.FILL)
        self.set_hexpand(True)
        self.set_margin_top(6)
        self.set_margin_bottom(6)

        self._play_icon = get_icon("play-arrow-symbolic")
        self._pause_icon = get_icon("pause-symbolic")

        self._play_button = Gtk.Button()
        self._play_button.set_child(self._play_icon)
        self._play_button.set_tooltip_text(_("Play simulation"))
        self._play_button.set_sensitive(False)
        self._play_button.set_focus_on_click(False)
        self._play_button.connect("clicked", self._on_play_clicked)
        self.append(self._play_button)

        self._step_back_button = Gtk.Button()
        self._step_back_button.set_child(get_icon("skip-previous-symbolic"))
        self._step_back_button.set_tooltip_text(_("Step backward"))
        self._step_back_button.set_sensitive(False)
        self._step_back_button.set_focus_on_click(False)
        self._step_back_button.connect("clicked", self._on_step_back)
        self.append(self._step_back_button)

        self._step_fwd_button = Gtk.Button()
        self._step_fwd_button.set_child(get_icon("skip-forward-symbolic"))
        self._step_fwd_button.set_tooltip_text(_("Step forward"))
        self._step_fwd_button.set_sensitive(False)
        self._step_fwd_button.set_focus_on_click(False)
        self._step_fwd_button.connect("clicked", self._on_step_fwd)
        self.append(self._step_fwd_button)

        self._slider = Gtk.Scale.new_with_range(
            Gtk.Orientation.HORIZONTAL, 0, 1, 1
        )
        self._slider.set_draw_value(False)
        self._slider.set_hexpand(True)
        self._slider.set_size_request(300, -1)
        self._slider.set_sensitive(False)
        self._slider.set_focus_on_click(False)
        self._slider.connect("value-changed", self._on_slider_changed)
        self.append(self._slider)

        self._speed_index = 0
        self._speed_button = Gtk.Button(label=f"{SPEED_OPTIONS[0]}x")
        self._speed_button.add_css_class("speed-button")
        self._speed_button.set_tooltip_text(_("Playback speed"))
        self._speed_button.set_focus_on_click(False)
        self._speed_button.connect("clicked", self._on_speed_clicked)
        self.append(self._speed_button)

        self._playing = False
        self._timer_id: int | None = None
        self._canvas = None
        self._player: PlaybackPlayer | None = None
        self._op_map: OpMapLike | None = None
        self._is_syncing = False
        self._tick_driving_slider = False
        self._sim_time: float = 0.0
        self._emitted_index: int = -1
        self._step_timer_id: int | None = None
        self._step_animating = False
        self._step_ticks_remaining = 0
        self._step_start_time = 0.0
        self._step_end_time = 0.0
        self._step_target = -1
        self._step_consumed = 0
        self._pending_steps = 0

        self.connect("destroy", self._on_destroy)

    def _on_destroy(self, widget):
        self._stop_playback()
        self._canvas = None

    def set_canvas(self, canvas):
        """Connect this overlay to a Canvas3D instance."""
        self._canvas = canvas

    def set_op_map(self, op_map: OpMapLike | None):
        """Provide the ops-to-gcode-line map used while stepping.

        When set, stepping skips commands that produced no G-code
        output (markers, redundant state changes) so every step lands
        on a line that is visible in the G-code viewer.  The slider
        then spans G-code line numbers, keeping its position in sync
        with the highlighted viewer line.

        The map is the encoder-produced ``MachineCodeOpMap``; this only
        holds a reference to its compact numpy arrays (4 bytes per line
        plus 8 bytes per op, already allocated for the job), so no
        additional per-job memory is created here.
        """
        self._op_map = op_map
        self._refresh_slider_range()

    def _gcode_line_count(self, index: int) -> int:
        """G-code lines produced by command *index* (1 if unknown)."""
        if self._op_map is None:
            return 1
        try:
            _, count = self._op_map.span_for_op(index)
            return count
        except (IndexError, AttributeError):
            return 1

    def _slider_extent(self) -> int:
        """Largest slider value: last G-code line (or command)."""
        if self._op_map is not None:
            return max(self._op_map.line_count - 1, 0)
        return max(self.command_count - 1, 0)

    def _slider_value_for_op(self, op_index: int) -> int | None:
        """Slider value displaying command *op_index*, or ``None``.

        With an op map this is the op's action line - the same line the
        G-code viewer highlights for it - so the slider always matches
        the selection.  Commands without G-code output keep the slider
        where it is.
        """
        if self._op_map is None:
            if not 0 <= op_index <= max(self.command_count - 1, 0):
                return None
            return max(0, op_index)
        try:
            start, count = self._op_map.span_for_op(op_index)
        except (IndexError, AttributeError):
            return None
        if count <= 0:
            return None
        return start + count - 1

    def _line_to_op(self, line: int) -> int | None:
        """Command owning *line*, or the nearest owner after/before it.

        Every encoded G-code line is owned by exactly one command, so
        the forward scan resolves on the first iteration; the loops are
        effectively O(1) and only walk further if an encoder ever left
        a run of unowned lines.  Called once per user scrub event,
        never per playback tick.
        """
        if self._op_map is None:
            return line
        n = self._op_map.line_count
        line = max(0, min(line, n - 1))
        for k in range(line, n):
            op = self._op_map.op_for_line(k)
            if op is not None and op >= 0:
                return op
        for k in range(line, -1, -1):
            op = self._op_map.op_for_line(k)
            if op is not None and op >= 0:
                return op
        return None

    def _set_slider_for_op(self, op_index: int):
        """Move the slider to *op_index*'s position (guarded)."""
        value = self._slider_value_for_op(op_index)
        if value is None:
            return
        self._tick_driving_slider = True
        self._slider.set_value(value)
        self._tick_driving_slider = False

    def _refresh_slider_range(self):
        """Recompute the slider range after the op map changed."""
        extent = self._slider_extent()
        self._tick_driving_slider = True
        self._slider.set_range(0, max(extent, 1))
        self._tick_driving_slider = False
        # A pristine playhead (-1) keeps the slider at the beginning;
        # an established playhead follows to its action line.
        if self.current_index >= 0:
            self._set_slider_for_op(self.current_index)

    def _resolve_step_target(self, base: int, steps: int) -> int:
        """Resolve a step of *steps* commands from *base*.

        Skips commands without G-code output so the playhead always
        lands on something visible in the G-code viewer.

        Complexity is bounded by ``abs(steps)`` (the number of queued
        clicks, a handful at most) times the longest run of commands
        without G-code output - linear, never exponential.
        """
        max_idx = self.command_count - 1
        delta = 1 if steps > 0 else -1
        k = base
        for _step in range(abs(steps)):
            k += delta
            if 0 <= k <= max_idx:
                while 0 <= k <= max_idx and self._gcode_line_count(k) == 0:
                    k += delta
        return max(0, min(k, max_idx))

    def set_player(
        self,
        player: PlaybackPlayer | None,
        initial_index: int = 0,
    ):
        """Set the OpPlayer backing this overlay's slider and seek calls.

        ``initial_index`` positions the slider for a freshly built
        player (typically 0). The player itself may already be seeked
        to the first layer for rendering.
        """
        self._cancel_step_animation()
        self._player = player
        if player is not None:
            self.update_ops_range(len(player.ops), initial_index)
            # Sync the simulated clock even when the slider does not
            # move (initial_index 0 with the slider already at 0), so
            # that stepping and playback start from the real position.
            if not self._playing:
                self._sim_time = player.get_cumulative_time(initial_index)
                player.set_sim_time(self._sim_time)
        else:
            self.update_ops_range(0)

    @property
    def command_count(self) -> int:
        """Number of commands in the current playback, or 0."""
        if self._player:
            return len(self._player.ops)
        return 0

    @property
    def current_index(self) -> int:
        """Current OpPlayer index, or -1."""
        if self._player:
            return self._player.current_index
        return -1

    def _total_time(self) -> float:
        """Total simulated job time in seconds (never zero)."""
        if self._player and self.command_count > 0:
            total = self._player.get_cumulative_time(self.command_count - 1)
            return max(total, 1e-9)
        return 1e-9

    def _set_slider_index(self, index: int):
        """Move the slider to a command index (guarded, no op map)."""
        value = self._slider_value_for_op(index)
        if value is None:
            return
        self._tick_driving_slider = True
        self._slider.set_value(value)
        self._tick_driving_slider = False

    def _emit_step_changed(self, ops_index: int):
        """Notify listeners when the command under the playhead changes."""
        if ops_index != self._emitted_index and not self._is_syncing:
            self._emitted_index = ops_index
            self.step_changed.send(self, ops_index=ops_index)

    def seek(self, index: int):
        """Jump the playhead to the given command index.

        Works while playing and while paused; the simulated clock is
        resynced to the new position either way. Anchoring by index
        (via ``set_playhead``) keeps the position exact across
        clusters of zero-duration commands.
        """
        if not self._player or self.command_count == 0:
            return
        self._cancel_step_animation()
        index = max(0, min(index, self.command_count - 1))
        self._player.set_playhead(index)
        self._sim_time = self._player.sim_time
        self._set_slider_index(index)
        if self._canvas:
            self._canvas.queue_render()
        self._emit_step_changed(index)

    def seek_to_fraction(self, fraction: float):
        """Seek to *fraction* (0.0-1.0) of the slider range."""
        if not self._player or self.command_count == 0:
            return
        fraction = max(0.0, min(1.0, fraction))
        target = round(fraction * self._slider_extent())
        op = self._line_to_op(target)
        if op is not None:
            self.seek(op)

    def handle_space(self):
        """Toggle playback when the space key is pressed."""
        if self.can_play():
            self.toggle_playback()

    def update_ops_range(self, command_count: int, initial_index: int = 0):
        """Update slider range for the given number of commands.

        initial_index positions the slider at the first layer's
        position so the canvas displays the correct surface from the
        start.  With an op map the slider spans G-code line numbers;
        otherwise it spans raw command indices.
        """
        if command_count > 0:
            self._refresh_slider_range()
            if initial_index > 0:
                # Restored playhead: show its action line.
                self._set_slider_for_op(initial_index)
            else:
                # Pristine playhead: nothing has executed yet, so the
                # slider starts at the very beginning of the job.
                self._tick_driving_slider = True
                self._slider.set_value(0)
                self._tick_driving_slider = False
            self._slider.set_sensitive(True)
            self._play_button.set_sensitive(True)
            self._step_back_button.set_sensitive(True)
            self._step_fwd_button.set_sensitive(True)
        else:
            self._slider.set_range(0, 1)
            self._set_slider_for_op(0)
            self._slider.set_sensitive(False)
            self._play_button.set_sensitive(False)
            self._step_back_button.set_sensitive(False)
            self._step_fwd_button.set_sensitive(False)

    def set_playback_position(self, ops_index: int):
        """
        Set the playhead from an external source (e.g. a G-code
        viewer click) without triggering a feedback loop.
        """
        if not self._player or self.command_count == 0:
            return
        self._is_syncing = True
        try:
            self.seek(int(ops_index))
        finally:
            self._is_syncing = False

    def can_play(self) -> bool:
        """Returns True if the play button is currently sensitive."""
        return self._play_button.get_sensitive()

    def toggle_playback(self):
        """Toggles play/pause state, as if the play button was clicked."""
        self._on_play_clicked(self._play_button)

    def _on_slider_changed(self, slider):
        if self._tick_driving_slider:
            return
        if not self._player or self.command_count == 0:
            return
        self._cancel_step_animation()
        index = self._line_to_op(int(slider.get_value()))
        if index is None:
            return
        if not self._playing:
            # Paused: anchor the playhead explicitly so positions stay
            # exact across zero-duration command clusters.
            self._player.set_playhead(index)
            self._sim_time = self._player.sim_time
        else:
            # While playing the slider drag resyncs the simulated
            # clock to the new position so the next tick continues
            # from there instead of snapping back to the old playhead.
            self._sim_time = self._player.get_cumulative_time(index)
            self._player.set_sim_time(self._sim_time)
            self._player.sync_state_to_playhead()
        if self._canvas:
            self._canvas.queue_render()
        self._emit_step_changed(index)

    def _on_play_clicked(self, button):
        if self._playing:
            self._stop_playback()
        else:
            self._start_playback()

    def _start_playback(self):
        if not self._canvas or self.command_count == 0:
            return
        if self._reached_end():
            self._restart_from_beginning()
        if self._step_animating:
            # Keep the interpolated time so playback continues
            # seamlessly from the gliding playhead.
            self._cancel_step_animation()
        if self._player:
            self._player.set_sim_time(self._sim_time)
        else:
            self._sim_time = 0.0
        self._playing = True
        self._play_button.set_child(self._pause_icon)
        self._play_button.set_tooltip_text(_("Pause simulation"))
        if self._timer_id is not None:
            GLib.source_remove(self._timer_id)
        self._timer_id = GLib.timeout_add(
            int(TICK_SECONDS * 1000), self._on_tick
        )

    def _reached_end(self) -> bool:
        """True when the playhead sits at (or past) the job end."""
        if self._player and self._player.is_finished:
            return True
        total = self._total_time()
        return self._sim_time >= total - TICK_SECONDS

    def _restart_from_beginning(self):
        self._sim_time = 0.0
        if self._player:
            self._player.set_progress_anchor(-1, 0.0)
        self._tick_driving_slider = True
        self._slider.set_value(0)
        self._tick_driving_slider = False

    def _stop_playback(self):
        self._cancel_step_animation()
        self._playing = False
        self._play_button.set_child(self._play_icon)
        self._play_button.set_tooltip_text(_("Play simulation"))
        if self._timer_id is not None:
            GLib.source_remove(self._timer_id)
            self._timer_id = None

    def _on_tick(self) -> bool:
        if not self._playing:
            return False
        if not self._canvas or not self._canvas.get_realized():
            self._stop_playback()
            return False
        if not self._player or self.command_count == 0:
            self._stop_playback()
            return False

        # Advance the simulated clock by real time times the speed
        # multiplier, then land on the command in effect at that time.
        multiplier = SPEED_OPTIONS[self._speed_index]
        self._sim_time += TICK_SECONDS * multiplier
        self._player.set_sim_time(self._sim_time)
        self._player.sync_state_to_playhead()
        max_idx = self.command_count - 1
        target = self._player.find_index_at_sim_time(self._sim_time)

        if target >= max_idx:
            # Anchor the playhead on the final command so its state
            # (laser off after the trailing M5s, home position) is
            # actually applied - identical to stepping forward onto
            # the last command by hand.
            self._player.set_playhead(max_idx)
            self._sim_time = self._player.sim_time
            self._tick_driving_slider = True
            self._slider.set_value(self._slider_extent())
            self._tick_driving_slider = False
            self._emit_step_changed(max_idx)
            self._stop_playback()
            return False

        self._set_slider_for_op(target)
        self._emit_step_changed(target)
        # The slider value only changes at command boundaries; within
        # a command the playhead still moves, so always redraw.
        self._canvas.queue_render()
        return True

    def _on_speed_clicked(self, button):
        self._speed_index = (self._speed_index + 1) % len(SPEED_OPTIONS)
        button.set_label(f"{SPEED_OPTIONS[self._speed_index]}x")

    def _on_step_back(self, button):
        # The bounds check must look past the playhead: a backward
        # click may cancel a forward glide even while the playhead
        # still sits at the first command.
        if self._pending_steps > 0 or self.current_index > 0:
            self._queue_step(-1)

    def _on_step_fwd(self, button):
        max_idx = self.command_count - 1
        if self._pending_steps < 0 or self.current_index < max_idx:
            self._queue_step(1)

    def _queue_step(self, delta: int):
        """Queue one manual step and start (or extend) its glide.

        Clicks arriving while a glide is running accumulate into the
        same glide, so rapid clicks move the coalesced number of
        commands in the time of a single step. While playing, steps
        jump instantly as before.
        """
        if self._playing or not self._player:
            if not self._player or self.command_count == 0:
                return
            base = self._player.find_index_at_sim_time(self._sim_time)
            target = self._resolve_step_target(base, delta)
            if target != self.current_index:
                self.seek(target)
            return
        self._pending_steps += delta
        self._start_step_glide()

    def _start_step_glide(self):
        """Start a glide covering the coalesced queued steps.

        The batch takes the duration of a single step no matter how
        many commands it spans; clicks arriving mid-glide retarget it
        to the new coalesced total.
        """
        if self._playing or not self._player:
            return
        # A fresh player sits at -1 (before the first command), so the
        # first forward step must land on the first visible command.
        current = self.current_index
        target = self._resolve_step_target(current, self._pending_steps)
        consumed = target - current
        if self._step_animating:
            if consumed == 0:
                # The queued clicks cancel each other out: snap back
                # to the playhead and drop the batch.
                self._cancel_step_animation()
                self._jump_to_index(current)
                return
            self._step_consumed = consumed
            self._step_target = target
            self._step_end_time = self._player.get_cumulative_time(target)
            return
        if consumed == 0:
            self._pending_steps = 0
            return
        end_time = self._player.get_cumulative_time(target)
        self._step_animating = True
        self._step_ticks_remaining = STEP_ANIMATION_TICKS
        self._step_start_time = self._sim_time
        self._step_end_time = end_time
        self._step_target = target
        self._step_consumed = consumed
        self._step_timer_id = GLib.timeout_add(
            int(TICK_SECONDS * 1000), self._on_step_tick
        )

    def _on_step_tick(self) -> bool:
        """Advance an in-flight step glide by one tick."""
        if not self._step_animating or not self._player:
            return False
        self._step_ticks_remaining -= 1
        progress = 1.0 - self._step_ticks_remaining / STEP_ANIMATION_TICKS
        self._sim_time = (
            self._step_start_time
            + (self._step_end_time - self._step_start_time) * progress
        )
        self._player.set_sim_time(self._sim_time)
        self._player.sync_state_to_playhead()
        if self._canvas:
            self._canvas.queue_render()
        if self._step_ticks_remaining == 0:
            self._pending_steps -= self._step_consumed
            self._cancel_step_animation()
            self._jump_to_index(self._step_target)
            return False
        return True

    def _jump_to_index(self, index: int):
        """Instantly place the playhead on a command boundary.

        Anchoring by index (rather than deriving the position from the
        simulated time) keeps stepping exact across clusters of
        zero-duration commands that share cumulative times.
        """
        if not self._player or self.command_count == 0:
            return
        index = max(0, min(index, self.command_count - 1))
        self._player.set_playhead(index)
        self._sim_time = self._player.sim_time
        self._set_slider_index(index)
        if self._canvas:
            self._canvas.queue_render()
        self._emit_step_changed(index)

    def _cancel_step_animation(self):
        """Stop any in-flight step glide and drop queued steps."""
        if self._step_timer_id is not None:
            GLib.source_remove(self._step_timer_id)
            self._step_timer_id = None
        self._step_animating = False
        self._pending_steps = 0
