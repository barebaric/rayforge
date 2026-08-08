import logging
from gettext import gettext as _
from typing import Any, Protocol

from blinker import Signal
from gi.repository import GLib, Gtk

from ..icons import get_icon
from ..shared.gtk import apply_css

logger = logging.getLogger(__name__)


class PlaybackPlayer(Protocol):
    """Minimal OpPlayer surface required by the playback overlay."""

    ops: Any

    @property
    def current_index(self) -> int: ...

    def seek(self, index: int) -> None: ...

    def seek_to_fraction(self, fraction: float) -> None: ...

    def find_index_at_sim_time(self, t: float) -> int: ...

    def get_cumulative_time(self, idx: int) -> float: ...

    def set_sim_time(self, t: float) -> None: ...

    def playback_progress(self) -> tuple[int, float]: ...


SPEED_OPTIONS = [1, 2, 4, 8, 16]

# Wall-clock interval between playback ticks (~60 fps, matching the
# display frame rate). The simulated clock advances by this amount per
# tick, scaled by the speed multiplier. Every tick queues a render so
# the interpolated playhead is redrawn continuously, not only when the
# slider value (command index) changes.
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
    shown as a bar below the 3D canvas. Slider drives OpPlayer.seek();
    play button starts a ~24 fps timer that advances the playhead by
    simulated machine time (speed multiplier scales real machine speed).
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
        self._play_button.connect("clicked", self._on_play_clicked)
        self.append(self._play_button)

        self._step_back_button = Gtk.Button()
        self._step_back_button.set_child(get_icon("skip-previous-symbolic"))
        self._step_back_button.set_tooltip_text(_("Step backward"))
        self._step_back_button.set_sensitive(False)
        self._step_back_button.connect("clicked", self._on_step_back)
        self.append(self._step_back_button)

        self._step_fwd_button = Gtk.Button()
        self._step_fwd_button.set_child(get_icon("skip-forward-symbolic"))
        self._step_fwd_button.set_tooltip_text(_("Step forward"))
        self._step_fwd_button.set_sensitive(False)
        self._step_fwd_button.connect("clicked", self._on_step_fwd)
        self.append(self._step_fwd_button)

        self._slider = Gtk.Scale.new_with_range(
            Gtk.Orientation.HORIZONTAL, 0, 1, 1
        )
        self._slider.set_draw_value(False)
        self._slider.set_hexpand(True)
        self._slider.set_size_request(300, -1)
        self._slider.set_sensitive(False)
        self._slider.connect("value-changed", self._on_slider_changed)
        self.append(self._slider)

        self._speed_index = 0
        self._speed_button = Gtk.Button(label=f"{SPEED_OPTIONS[0]}x")
        self._speed_button.add_css_class("speed-button")
        self._speed_button.set_tooltip_text(_("Playback speed"))
        self._speed_button.connect("clicked", self._on_speed_clicked)
        self.append(self._speed_button)

        self._playing = False
        self._timer_id: int | None = None
        self._canvas = None
        self._player: PlaybackPlayer | None = None
        self._is_syncing = False
        self._suppress_seek = False
        self._tick_driving_slider = False
        self._sim_time: float = 0.0
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

    def set_player(
        self,
        player: PlaybackPlayer | None,
        initial_index: int = 0,
    ):
        """Set the OpPlayer backing this overlay's slider and seek calls.

        ``initial_index`` positions the slider for a freshly built player
        (typically 0). The player itself may already be seeked to the
        first layer for rendering.
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

    def seek(self, index: int):
        """Seek the OpPlayer to the given command index.

        While paused, the simulated clock is resynced to the new
        position so that resuming play continues from there.
        """
        self._cancel_step_animation()
        if self._player:
            self._player.seek(index)
            if not self._playing:
                self._sim_time = self._player.get_cumulative_time(index)
                self._player.set_sim_time(self._sim_time)
            if self._canvas:
                self._canvas.queue_render()

    def seek_to_fraction(self, fraction: float):
        """Seek the OpPlayer by fraction (0.0 to 1.0) and sync the slider."""
        if self._player:
            self._player.seek_to_fraction(fraction)
            self.update_ops_range(
                len(self._player.ops),
                self._player.current_index,
            )
            if not self._playing:
                self._sim_time = self._player.get_cumulative_time(
                    self._player.current_index
                )
                self._player.set_sim_time(self._sim_time)
            if self._canvas:
                self._canvas.queue_render()

    def handle_space(self):
        """Toggle playback when the space key is pressed."""
        if self.can_play():
            self.toggle_playback()

    def update_ops_range(self, command_count: int, initial_index: int = 0):
        """Update slider range for the given number of commands.

        initial_index sets the slider to the first layer's position
        so the canvas displays the correct surface from the start.
        """
        if command_count > 0:
            self._slider.set_range(0, command_count - 1)
            self._slider.set_value(initial_index)
            self._slider.set_sensitive(True)
            self._play_button.set_sensitive(True)
            self._step_back_button.set_sensitive(True)
            self._step_fwd_button.set_sensitive(True)
        else:
            self._slider.set_range(0, 1)
            self._slider.set_value(0)
            self._slider.set_sensitive(False)
            self._play_button.set_sensitive(False)
            self._step_back_button.set_sensitive(False)
            self._step_fwd_button.set_sensitive(False)

    def get_slider_index(self) -> int:
        return int(self._slider.get_value())

    def _on_slider_changed(self, slider):
        if self._canvas:
            index = int(slider.get_value())
            if self.current_index != index:
                self.seek(index)
        # A user-initiated scrub while playing resyncs the simulated
        # clock to the new position so the next tick continues from
        # there instead of snapping back to the pre-drag playhead.
        # Tick-driven slider moves set ``_tick_driving_slider`` so they
        # are not mistaken for user scrubs.
        if self._playing and self._player and not self._tick_driving_slider:
            self._sim_time = self._player.get_cumulative_time(
                int(slider.get_value())
            )
            self._player.set_sim_time(self._sim_time)
        if not self._is_syncing:
            self.step_changed.send(self, ops_index=int(slider.get_value()))

    def set_playback_position(self, ops_index: int):
        """
        Set the slider position from an external source (e.g. a G-code
        viewer click) without triggering a feedback loop.
        """
        self._cancel_step_animation()
        self._is_syncing = True
        self._slider.set_value(ops_index)
        self._is_syncing = False

    def can_play(self) -> bool:
        """Returns True if the play button is currently sensitive."""
        return self._play_button.get_sensitive()

    def toggle_playback(self):
        """Toggles play/pause state, as if the play button was clicked."""
        self._on_play_clicked(self._play_button)

    def _on_play_clicked(self, button):
        if self._playing:
            self._stop_playback()
        else:
            self._start_playback()

    def _start_playback(self):
        if not self._canvas or self.command_count == 0:
            return
        max_idx = self.command_count - 1
        current = int(self._slider.get_value())
        if max_idx >= 0 and current >= max_idx:
            self._slider.set_value(0)
            current = 0
        # Resync the simulated clock to the current playhead so that
        # playback continues from wherever the user left the slider.
        # An in-flight step animation keeps its interpolated time so
        # that playback continues seamlessly from the gliding playhead.
        if self._player:
            if self._step_animating:
                self._cancel_step_animation()
            else:
                self._sim_time = self._player.get_cumulative_time(current)
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
        max_idx = self.command_count - 1
        target = self._player.find_index_at_sim_time(self._sim_time)

        if target >= max_idx:
            self._tick_driving_slider = True
            self._slider.set_value(max_idx)
            self._tick_driving_slider = False
            self._stop_playback()
            return False

        self._tick_driving_slider = True
        self._slider.set_value(target)
        self._tick_driving_slider = False
        # The slider value only changes at command boundaries; within a
        # command the playhead still moves, so always redraw.
        self._canvas.queue_render()
        return True

    def _on_speed_clicked(self, button):
        self._speed_index = (self._speed_index + 1) % len(SPEED_OPTIONS)
        button.set_label(f"{SPEED_OPTIONS[self._speed_index]}x")

    def _on_step_back(self, button):
        # The bounds check must look past the playhead (slider + queued
        # steps): a backward click may cancel a forward glide even
        # while the slider still sits at 0.
        if self._pending_steps > 0 or int(self._slider.get_value()) > 0:
            self._queue_step(-1)

    def _on_step_fwd(self, button):
        max_idx = self._slider.get_adjustment().get_upper()
        if self._pending_steps < 0 or int(self._slider.get_value()) < max_idx:
            self._queue_step(1)

    def _queue_step(self, delta: int):
        """Queue one manual step and start (or extend) its glide.

        Clicks arriving while a glide is running accumulate into the
        same glide, so rapid clicks move the coalesced number of
        commands in the time of a single step. While playing, steps
        jump instantly as before.
        """
        if self._playing or not self._player:
            self._slider.set_value(int(self._slider.get_value()) + delta)
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
        current = int(self._slider.get_value())
        max_idx = int(self._slider.get_adjustment().get_upper())
        target = max(0, min(current + self._pending_steps, max_idx))
        consumed = target - current
        if self._step_animating:
            if consumed == 0:
                # The queued clicks cancel each other out: snap back to
                # the playhead and drop the batch.
                self._cancel_step_animation()
                self._sim_time = self._player.get_cumulative_time(current)
                self._player.set_sim_time(self._sim_time)
                if self._canvas:
                    self._canvas.queue_render()
                return
            self._step_consumed = consumed
            self._step_target = target
            self._step_end_time = self._player.get_cumulative_time(target)
            return
        if consumed == 0:
            self._pending_steps = 0
            return
        end_time = self._player.get_cumulative_time(target)
        while end_time == self._sim_time:
            # Zero-duration commands between the playhead and the
            # target: move through them instantly, then continue with
            # the rest of the batch.
            self._pending_steps -= consumed
            self._slider.set_value(target)
            current = target
            target = max(0, min(current + self._pending_steps, max_idx))
            consumed = target - current
            if consumed == 0:
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
        if self._canvas:
            self._canvas.queue_render()
        if self._step_ticks_remaining == 0:
            self._pending_steps -= self._step_consumed
            self._cancel_step_animation()
            self._slider.set_value(self._step_target)
            return False
        return True

    def _cancel_step_animation(self):
        """Stop any in-flight step glide and drop queued steps."""
        if self._step_timer_id is not None:
            GLib.source_remove(self._step_timer_id)
            self._step_timer_id = None
        self._step_animating = False
        self._pending_steps = 0
