# flake8: noqa: E402
"""UI tests for the PlaybackOverlay playback glue."""

import gi

gi.require_version("Gtk", "4.0")
gi.require_version("Adw", "1")

import pytest

from rayforge.ui_gtk.sim3d.playback_overlay import (
    STEP_ANIMATION_SECONDS,
    TICK_SECONDS,
    PlaybackOverlay,
)


class FakePlayer:
    """Minimal OpPlayer stand-in exposing the playback surface.

    Each command takes ``cmd_time`` simulated seconds: cumulative time
    at index *i* is ``cmd_time * (i + 1)`` and the playhead lands on the
    last command whose cumulative time is <= *t*.
    """

    def __init__(self, n_ops: int = 0, cmd_time: float = 1.0):
        class FakeOps:
            def __init__(self, n):
                self._n = n

            def len(self) -> int:
                return self._n

            def __len__(self) -> int:
                return self._n

        self.ops = FakeOps(n_ops)
        self._current_index = -1
        self._cmd_time = cmd_time
        self._sim_time = 0.0

    @property
    def current_index(self) -> int:
        return self._current_index

    def seek(self, index: int):
        self._current_index = index

    def seek_to_fraction(self, fraction: float):
        self._current_index = int(self.ops.len() * fraction)

    def find_index_at_sim_time(self, t: float) -> int:
        n = self.ops.len()
        if n == 0:
            return 0
        idx = int(t / self._cmd_time) - 1
        return max(0, min(idx, n - 1))

    def get_cumulative_time(self, idx: int) -> float:
        n = self.ops.len()
        if n == 0:
            return 0.0
        idx = max(0, min(idx, n - 1))
        return self._cmd_time * (idx + 1)

    def set_sim_time(self, t: float):
        self._sim_time = t

    def playback_progress(self):
        return (self._current_index + 1, 0.0)

    def render_state(self):
        return None


class FakeCanvas:
    """Minimal Canvas3D stand-in recording queue_render calls."""

    def __init__(self):
        self.render_queued = 0

    def queue_render(self):
        self.render_queued += 1

    def get_realized(self) -> bool:
        return True


@pytest.mark.ui
def test_controls_do_not_grab_focus_on_click(ui_context_initializer):
    overlay = PlaybackOverlay()
    for widget in (
        overlay._play_button,
        overlay._step_back_button,
        overlay._step_fwd_button,
        overlay._speed_button,
        overlay._slider,
    ):
        assert not widget.get_focus_on_click()


@pytest.mark.ui
def test_command_count_no_player(ui_context_initializer):
    overlay = PlaybackOverlay()
    assert overlay.command_count == 0
    assert overlay.current_index == -1


@pytest.mark.ui
def test_command_count_with_player(ui_context_initializer):
    overlay = PlaybackOverlay()
    overlay.set_player(FakePlayer(n_ops=10))
    assert overlay.command_count == 10


@pytest.mark.ui
def test_set_player_none_disables_range(ui_context_initializer):
    overlay = PlaybackOverlay()
    overlay.set_player(FakePlayer(n_ops=10))
    overlay.set_player(None)
    assert overlay.command_count == 0
    assert not overlay._slider.get_sensitive()


@pytest.mark.ui
def test_seek_seeks_player_and_queues_render(ui_context_initializer):
    overlay = PlaybackOverlay()
    canvas = FakeCanvas()
    overlay.set_canvas(canvas)
    player = FakePlayer(n_ops=10)
    overlay.set_player(player)
    canvas.render_queued = 0
    overlay.seek(4)
    assert player.current_index == 4
    assert canvas.render_queued == 1


@pytest.mark.ui
def test_seek_without_player_noop(ui_context_initializer):
    overlay = PlaybackOverlay()
    canvas = FakeCanvas()
    overlay.set_canvas(canvas)
    overlay.seek(4)
    assert canvas.render_queued == 0


@pytest.mark.ui
def test_set_player_enables_controls(ui_context_initializer):
    overlay = PlaybackOverlay()
    overlay.set_player(FakePlayer(n_ops=0))
    assert not overlay._play_button.get_sensitive()
    assert not overlay._step_back_button.get_sensitive()
    assert not overlay._step_fwd_button.get_sensitive()
    assert not overlay._slider.get_sensitive()

    overlay.set_player(FakePlayer(n_ops=10))
    assert overlay._play_button.get_sensitive()
    assert overlay._step_back_button.get_sensitive()
    assert overlay._step_fwd_button.get_sensitive()
    assert overlay._slider.get_sensitive()


@pytest.mark.ui
def test_set_player_starts_slider_at_zero(ui_context_initializer):
    overlay = PlaybackOverlay()
    player = FakePlayer(n_ops=10)
    player.seek(4)
    overlay.set_player(player)
    assert int(overlay._slider.get_value()) == 0


@pytest.mark.ui
def test_set_player_with_initial_index_positions_slider(
    ui_context_initializer,
):
    overlay = PlaybackOverlay()
    player = FakePlayer(n_ops=10)
    overlay.set_player(player, initial_index=4)
    assert int(overlay._slider.get_value()) == 4


@pytest.mark.ui
def test_seek_to_fraction_seeks_and_syncs_slider(ui_context_initializer):
    overlay = PlaybackOverlay()
    canvas = FakeCanvas()
    overlay.set_canvas(canvas)
    player = FakePlayer(n_ops=10)
    overlay.set_player(player)
    canvas.render_queued = 0
    overlay.seek_to_fraction(0.5)
    assert player.current_index == 5
    assert canvas.render_queued == 1


@pytest.mark.ui
def test_handle_space_toggles_only_when_playable(ui_context_initializer):
    overlay = PlaybackOverlay()
    overlay.set_canvas(FakeCanvas())
    overlay.set_player(FakePlayer(n_ops=0))
    assert not overlay._play_button.get_sensitive()
    overlay.handle_space()
    assert not overlay._playing

    overlay.set_player(FakePlayer(n_ops=10))
    assert overlay._play_button.get_sensitive()
    overlay.handle_space()
    assert overlay._playing


@pytest.mark.ui
def test_start_playback_resyncs_sim_time_from_playhead(ui_context_initializer):
    overlay = PlaybackOverlay()
    overlay.set_canvas(FakeCanvas())
    player = FakePlayer(n_ops=10, cmd_time=1.0)
    overlay.set_player(player)
    overlay._slider.set_value(3)
    overlay._start_playback()
    assert overlay._sim_time == pytest.approx(4.0)
    assert overlay._playing


@pytest.mark.ui
def test_start_playback_wraps_from_end(ui_context_initializer):
    overlay = PlaybackOverlay()
    overlay.set_canvas(FakeCanvas())
    player = FakePlayer(n_ops=10, cmd_time=1.0)
    overlay.set_player(player)
    overlay._slider.set_value(9)
    overlay._start_playback()
    assert int(overlay._slider.get_value()) == 0
    assert overlay._sim_time == pytest.approx(1.0)
    assert overlay._playing


@pytest.mark.ui
def test_tick_advances_sim_time_by_tick_interval(ui_context_initializer):
    overlay = PlaybackOverlay()
    overlay.set_canvas(FakeCanvas())
    player = FakePlayer(n_ops=100, cmd_time=1.0)
    overlay.set_player(player)
    overlay._start_playback()
    ticks = int(1.0 / TICK_SECONDS) + 1  # enough to cross cmd_time=1.0
    for _ in range(ticks):
        assert overlay._on_tick()
    # sim_time = 1.0 (start) + ticks * TICK_SECONDS -> command index 1.
    assert overlay._sim_time == pytest.approx(1.0 + ticks * TICK_SECONDS)
    assert int(overlay._slider.get_value()) == 1


@pytest.mark.ui
def test_speed_multiplier_scales_sim_time(ui_context_initializer):
    overlay = PlaybackOverlay()
    overlay.set_canvas(FakeCanvas())
    player = FakePlayer(n_ops=100, cmd_time=1.0)
    overlay.set_player(player)
    overlay._speed_index = 1  # 2x
    overlay._start_playback()
    ticks = int(0.5 / TICK_SECONDS) + 1  # half the sim time at 2x
    for _ in range(ticks):
        assert overlay._on_tick()
    # sim_time = 1.0 + ticks * 2 * TICK_SECONDS -> command index 1.
    assert overlay._sim_time == pytest.approx(1.0 + 2 * ticks * TICK_SECONDS)
    assert int(overlay._slider.get_value()) == 1


@pytest.mark.ui
def test_tick_stops_at_last_command(ui_context_initializer):
    overlay = PlaybackOverlay()
    overlay.set_canvas(FakeCanvas())
    player = FakePlayer(n_ops=3, cmd_time=1.0)
    overlay.set_player(player)
    overlay._start_playback()
    max_ticks = int(3.0 / TICK_SECONDS) + 10
    for _ in range(max_ticks):
        if not overlay._playing:
            break
        overlay._on_tick()
    assert not overlay._playing
    assert int(overlay._slider.get_value()) == 2


@pytest.mark.ui
def test_step_forward_animates_to_next_command(ui_context_initializer):
    overlay = PlaybackOverlay()
    overlay.set_canvas(FakeCanvas())
    player = FakePlayer(n_ops=10, cmd_time=1.0)
    overlay.set_player(player)
    overlay._on_step_fwd(None)
    # Stepping is animated, not instant: the slider keeps showing the
    # current command until the animation has played out.
    assert overlay._step_animating
    assert int(overlay._slider.get_value()) == 0
    ticks = 0
    while overlay._step_animating:
        overlay._on_step_tick()
        ticks += 1
    assert ticks == int(STEP_ANIMATION_SECONDS / TICK_SECONDS)
    assert player.current_index == 1
    assert overlay._sim_time == pytest.approx(2.0)
    assert int(overlay._slider.get_value()) == 1


@pytest.mark.ui
def test_step_back_animates_to_previous_command(ui_context_initializer):
    overlay = PlaybackOverlay()
    overlay.set_canvas(FakeCanvas())
    player = FakePlayer(n_ops=10, cmd_time=1.0)
    overlay.set_player(player)
    overlay._slider.set_value(5)
    overlay._on_step_back(None)
    assert overlay._step_animating
    while overlay._step_animating:
        overlay._on_step_tick()
    assert player.current_index == 4
    assert overlay._sim_time == pytest.approx(5.0)
    assert int(overlay._slider.get_value()) == 4


@pytest.mark.ui
def test_step_animation_interpolates_sim_time(ui_context_initializer):
    overlay = PlaybackOverlay()
    canvas = FakeCanvas()
    overlay.set_canvas(canvas)
    player = FakePlayer(n_ops=10, cmd_time=10.0)
    overlay.set_player(player)
    overlay._on_step_fwd(None)
    # Halfway through the animation of a 10 s command the playhead is
    # interpolated mid-command, not parked on a command edge.
    for _ in range(int(STEP_ANIMATION_SECONDS / TICK_SECONDS / 2)):
        overlay._on_step_tick()
    assert overlay._sim_time == pytest.approx(15.0)
    assert int(overlay._slider.get_value()) == 0
    while overlay._step_animating:
        overlay._on_step_tick()
    assert player.current_index == 1


@pytest.mark.ui
def test_step_forward_while_playing_jumps_instantly(ui_context_initializer):
    overlay = PlaybackOverlay()
    overlay.set_canvas(FakeCanvas())
    player = FakePlayer(n_ops=10, cmd_time=1.0)
    overlay.set_player(player)
    overlay._start_playback()
    overlay._on_step_fwd(None)
    assert not overlay._step_animating
    assert player.current_index == 1


@pytest.mark.ui
def test_play_during_step_animation_keeps_interpolated_time(
    ui_context_initializer,
):
    overlay = PlaybackOverlay()
    overlay.set_canvas(FakeCanvas())
    player = FakePlayer(n_ops=10, cmd_time=10.0)
    overlay.set_player(player)
    overlay._on_step_fwd(None)
    for _ in range(int(STEP_ANIMATION_SECONDS / TICK_SECONDS / 2)):
        overlay._on_step_tick()
    mid = overlay._sim_time
    overlay._start_playback()
    assert not overlay._step_animating
    assert overlay._playing
    assert overlay._sim_time == pytest.approx(mid)


@pytest.mark.ui
def test_seek_cancels_in_flight_step_animation(ui_context_initializer):
    overlay = PlaybackOverlay()
    overlay.set_canvas(FakeCanvas())
    player = FakePlayer(n_ops=10, cmd_time=10.0)
    overlay.set_player(player)
    overlay._on_step_fwd(None)
    assert overlay._step_animating
    overlay.seek(3)
    assert not overlay._step_animating
    assert player.current_index == 3


@pytest.mark.ui
def test_rapid_step_forward_clicks_coalesce(ui_context_initializer):
    overlay = PlaybackOverlay()
    overlay.set_canvas(FakeCanvas())
    player = FakePlayer(n_ops=10, cmd_time=1.0)
    overlay.set_player(player)
    for _ in range(5):
        overlay._on_step_fwd(None)
    # The batch glides as one: five commands in the time of one step.
    assert overlay._pending_steps == 5
    assert overlay._step_animating
    ticks = 0
    while overlay._pending_steps or overlay._step_animating:
        overlay._on_step_tick()
        ticks += 1
    assert ticks == int(STEP_ANIMATION_SECONDS / TICK_SECONDS)
    assert player.current_index == 5
    assert int(overlay._slider.get_value()) == 5
    assert overlay._sim_time == pytest.approx(6.0)


@pytest.mark.ui
def test_rapid_step_back_clicks_coalesce(ui_context_initializer):
    overlay = PlaybackOverlay()
    overlay.set_canvas(FakeCanvas())
    player = FakePlayer(n_ops=10, cmd_time=1.0)
    overlay.set_player(player)
    overlay._slider.set_value(6)
    for _ in range(4):
        overlay._on_step_back(None)
    while overlay._pending_steps or overlay._step_animating:
        overlay._on_step_tick()
    assert player.current_index == 2
    assert overlay._sim_time == pytest.approx(3.0)


@pytest.mark.ui
def test_mixed_step_clicks_coalesce_to_net(ui_context_initializer):
    overlay = PlaybackOverlay()
    overlay.set_canvas(FakeCanvas())
    player = FakePlayer(n_ops=10, cmd_time=1.0)
    overlay.set_player(player)
    overlay._slider.set_value(3)
    for _ in range(3):
        overlay._on_step_fwd(None)
    overlay._on_step_back(None)
    assert overlay._pending_steps == 2
    while overlay._pending_steps or overlay._step_animating:
        overlay._on_step_tick()
    assert player.current_index == 5
    assert overlay._sim_time == pytest.approx(6.0)


@pytest.mark.ui
def test_step_queue_clamped_at_end(ui_context_initializer):
    overlay = PlaybackOverlay()
    overlay.set_canvas(FakeCanvas())
    player = FakePlayer(n_ops=3, cmd_time=1.0)
    overlay.set_player(player)
    for _ in range(10):
        overlay._on_step_fwd(None)
    ticks = 0
    while overlay._pending_steps or overlay._step_animating:
        overlay._on_step_tick()
        ticks += 1
    # Ten clicks only reach the last command, in the time of one step.
    assert ticks == int(STEP_ANIMATION_SECONDS / TICK_SECONDS)
    assert int(overlay._slider.get_value()) == 2
    assert player.current_index == 2


@pytest.mark.ui
def test_clicks_during_glide_merge_into_it(ui_context_initializer):
    overlay = PlaybackOverlay()
    overlay.set_canvas(FakeCanvas())
    player = FakePlayer(n_ops=10, cmd_time=1.0)
    overlay.set_player(player)
    overlay._on_step_fwd(None)
    # Two more clicks while the first glide is running merge into it.
    overlay._on_step_fwd(None)
    overlay._on_step_fwd(None)
    assert overlay._pending_steps == 3
    ticks = 0
    while overlay._pending_steps or overlay._step_animating:
        overlay._on_step_tick()
        ticks += 1
    # All three commands glide over in the time of a single step.
    assert ticks == int(STEP_ANIMATION_SECONDS / TICK_SECONDS)
    assert player.current_index == 3
    assert int(overlay._slider.get_value()) == 3


@pytest.mark.ui
def test_opposing_clicks_during_glide_cancel_it(ui_context_initializer):
    overlay = PlaybackOverlay()
    canvas = FakeCanvas()
    overlay.set_canvas(canvas)
    player = FakePlayer(n_ops=10, cmd_time=1.0)
    overlay.set_player(player)
    overlay._on_step_fwd(None)
    for _ in range(6):
        overlay._on_step_tick()
    # A backward click cancels the queued forward step: the glide snaps
    # back to the playhead it started from.
    overlay._on_step_back(None)
    assert overlay._pending_steps == 0
    assert not overlay._step_animating
    assert overlay._sim_time == pytest.approx(1.0)
    assert int(overlay._slider.get_value()) == 0


@pytest.mark.ui
def test_seek_during_glide_cancels_pending_steps(ui_context_initializer):
    overlay = PlaybackOverlay()
    overlay.set_canvas(FakeCanvas())
    player = FakePlayer(n_ops=10, cmd_time=1.0)
    overlay.set_player(player)
    overlay._on_step_fwd(None)
    overlay._on_step_fwd(None)
    assert overlay._pending_steps == 2
    overlay.seek(4)
    assert not overlay._step_animating
    assert overlay._pending_steps == 0
    assert player.current_index == 4


@pytest.mark.ui
def test_slider_seek_while_paused_resyncs_sim_time(ui_context_initializer):
    overlay = PlaybackOverlay()
    overlay.set_canvas(FakeCanvas())
    player = FakePlayer(n_ops=10, cmd_time=1.0)
    overlay.set_player(player)
    overlay._slider.set_value(5)
    assert overlay._sim_time == pytest.approx(6.0)
    assert player.current_index == 5


@pytest.mark.ui
def test_playback_ticks_do_not_resync_sim_time(ui_context_initializer):
    overlay = PlaybackOverlay()
    overlay.set_canvas(FakeCanvas())
    player = FakePlayer(n_ops=100, cmd_time=1.0)
    overlay.set_player(player)
    overlay._start_playback()
    # Playback keeps advancing the simulated clock; slider-driven seeks
    # during playback must not clamp _sim_time back to a command edge.
    for _ in range(24):
        overlay._on_tick()
    assert overlay._sim_time == pytest.approx(1.0 + 24 * TICK_SECONDS)
    for _ in range(24):
        overlay._on_tick()
    assert overlay._sim_time == pytest.approx(1.0 + 48 * TICK_SECONDS)


@pytest.mark.ui
def test_tick_always_queues_render_within_command(ui_context_initializer):
    overlay = PlaybackOverlay()
    canvas = FakeCanvas()
    overlay.set_canvas(canvas)
    player = FakePlayer(n_ops=100, cmd_time=10.0)
    overlay.set_player(player)
    overlay._start_playback()
    canvas.render_queued = 0
    # All ticks stay inside the first long command: the slider value
    # never changes, but every tick must still redraw the playhead.
    for _ in range(5):
        assert overlay._on_tick()
    assert canvas.render_queued == 5


@pytest.mark.ui
def test_scrub_while_playing_resyncs_sim_time(ui_context_initializer):
    overlay = PlaybackOverlay()
    overlay.set_canvas(FakeCanvas())
    player = FakePlayer(n_ops=100, cmd_time=1.0)
    overlay.set_player(player)
    overlay._start_playback()
    for _ in range(60):
        overlay._on_tick()
    before = overlay._sim_time
    assert before > 1.0
    # User drags the slider to command 20 while playback is active.
    overlay._slider.set_value(20)
    assert overlay._sim_time == pytest.approx(21.0)
    assert overlay._playing
    # The next tick continues from the dragged position, not the old one.
    overlay._on_tick()
    assert overlay._sim_time > 21.0
    assert overlay._sim_time < 21.0 + 2 * TICK_SECONDS


@pytest.mark.ui
def test_scrub_while_playing_does_not_fight_tick(ui_context_initializer):
    overlay = PlaybackOverlay()
    overlay.set_canvas(FakeCanvas())
    player = FakePlayer(n_ops=100, cmd_time=1.0)
    overlay.set_player(player)
    overlay._start_playback()
    for _ in range(60):
        overlay._on_tick()
    overlay._slider.set_value(20)
    # Repeated ticks must keep advancing from the scrubbed position,
    # never snapping the slider back below it.
    slider_before = int(overlay._slider.get_value())
    for _ in range(10):
        overlay._on_tick()
    assert int(overlay._slider.get_value()) >= slider_before
