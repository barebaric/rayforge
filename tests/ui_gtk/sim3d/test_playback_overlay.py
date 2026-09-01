# flake8: noqa: E402
"""UI tests for the PlaybackOverlay playback glue."""

import gi

gi.require_version("Gtk", "4.0")
gi.require_version("Adw", "1")

from typing import Any

import pytest

from rayforge.ui_gtk.sim3d.playback_overlay import (
    STEP_ANIMATION_SECONDS,
    TICK_SECONDS,
    PlaybackOverlay,
)


class FakePlayer:
    """Minimal OpPlayer stand-in exposing the playback surface.

    Each command takes ``cmd_time`` simulated seconds (or the entries
    of ``cmd_times``): cumulative time at index *i* is the sum of the
    durations up to and including *i*, and the playhead lands on the
    last command whose cumulative time is <= *t*.

    The ``ops`` attribute is intentionally opaque to the type checker;
    the overlay only reads its length.
    """

    ops: Any

    def __init__(
        self,
        n_ops: int = 0,
        cmd_time: float = 1.0,
        cmd_times: list[float] | None = None,
    ) -> None:
        class FakeOps:
            def __init__(self, n):
                self._n = n

            def len(self) -> int:
                return self._n

            def __len__(self) -> int:
                return self._n

        self.ops = FakeOps(n_ops)
        self._current_index = -1
        if cmd_times is None:
            cmd_times = [cmd_time] * n_ops
        self._cmd_times = list(cmd_times)
        self._sim_time = 0.0
        self.seek_count = 0

    @property
    def current_index(self) -> int:
        return self._current_index

    @property
    def is_finished(self) -> bool:
        if not self._cmd_times:
            return False
        return self._sim_time >= sum(self._cmd_times) - 1e-9

    @property
    def sim_time(self) -> float:
        return self._sim_time

    def set_playhead(self, index: int) -> None:
        n = len(self._cmd_times)
        index = max(0, min(index, n - 1))
        self._current_index = index
        self._sim_time = self.get_cumulative_time(index)

    def set_progress_anchor(self, completed: int, t: float) -> None:
        n = len(self._cmd_times)
        completed = max(-1, min(completed, n - 1))
        self._current_index = completed
        self._sim_time = float(t)

    def seek(self, index: int) -> None:
        self.seek_count += 1
        self._current_index = index

    def seek_to_fraction(self, fraction: float) -> None:
        self._current_index = int(self.ops.len() * fraction)

    def find_index_at_sim_time(self, t: float) -> int:
        n = len(self._cmd_times)
        if n == 0:
            return 0
        idx = 0
        acc = 0.0
        for i, duration in enumerate(self._cmd_times):
            acc += duration
            if acc <= t:
                idx = i
        return idx

    def get_cumulative_time(self, idx: int) -> float:
        n = len(self._cmd_times)
        if n == 0:
            return 0.0
        idx = max(0, min(idx, n - 1))
        return sum(self._cmd_times[: idx + 1])

    def set_sim_time(self, t: float):
        self._sim_time = t

    def sync_state_to_playhead(self) -> None:
        # Mirrors OpPlayer: state advances only through the commands
        # COMPLETED before the playhead (p - 1), leaving trailing
        # commands unapplied until explicitly anchored.
        p, _frac = self.playback_progress()
        n = len(self._cmd_times)
        self._current_index = max(0, min(p - 1, n - 1))

    def playback_progress(self) -> tuple[int, float]:
        n = len(self._cmd_times)
        if n == 0:
            return (0, 0.0)
        idx = self.find_index_at_sim_time(self._sim_time)
        t_end = self.get_cumulative_time(idx)
        if idx + 1 >= n:
            return (idx, 1.0)
        span = self.get_cumulative_time(idx + 1) - t_end
        if span <= 0.0:
            return (idx + 1, 0.0)
        frac = (self._sim_time - t_end) / span
        return (idx + 1, max(0.0, min(1.0, frac)))

    def render_state(self):
        return None


class FakeOpMap:
    """Minimal MachineCodeOpMap stand-in: op -> (start_line, lines)."""

    def __init__(self, spans):
        self._spans = list(spans)
        self._line_to_op = {}
        for op, (start, count) in enumerate(self._spans):
            for ln in range(start, start + count):
                self._line_to_op[ln] = op

    @property
    def op_count(self) -> int:
        return len(self._spans)

    @property
    def line_count(self) -> int:
        return max((start + count for start, count in self._spans), default=0)

    def span_for_op(self, op_index):
        return self._spans[op_index]

    def op_for_line(self, line_idx: int) -> int | None:
        return self._line_to_op.get(line_idx)


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
    # Half the total simulated time (5 s of 10 s) lands on command 4.
    assert player.current_index == 4
    assert int(overlay._slider.get_value()) == 4
    assert canvas.render_queued >= 1


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
    # Drag the slider to command 3 (4 s cumulative) while paused.
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
    # Run to the very end, then press play again: it restarts from 0.
    overlay.seek_to_fraction(1.0)
    overlay._start_playback()
    assert int(overlay._slider.get_value()) == 0
    assert overlay._sim_time == pytest.approx(0.0)
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
    player.seek(0)
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
    assert player.current_index == 5
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
    player.seek(0)
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
    player.seek(0)
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
    player.seek(0)
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
    player.seek(0)
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
    # User drags the slider to command 24 (25 s cumulative) while
    # playback is active.
    overlay._slider.set_value(24)
    assert overlay._sim_time == pytest.approx(25.0)
    assert overlay._playing
    # The next tick continues from the dragged position, not the old one.
    overlay._on_tick()
    assert overlay._sim_time > 25.0
    assert overlay._sim_time < 25.0 + 2 * TICK_SECONDS


@pytest.mark.ui
def test_scrub_while_playing_does_not_fight_tick(ui_context_initializer):
    overlay = PlaybackOverlay()
    overlay.set_canvas(FakeCanvas())
    player = FakePlayer(n_ops=100, cmd_time=1.0)
    overlay.set_player(player)
    overlay._start_playback()
    for _ in range(60):
        overlay._on_tick()
    overlay._slider.set_value(24)
    # Repeated ticks must keep advancing from the scrubbed position,
    # never snapping the slider back below it.
    slider_before = int(overlay._slider.get_value())
    for _ in range(10):
        overlay._on_tick()
    assert int(overlay._slider.get_value()) >= slider_before


@pytest.mark.ui
def test_tick_driven_slider_moves_do_not_seek(ui_context_initializer):
    """Regression test for issue #370.

    Tick-driven slider moves are driven by the simulated clock; a seek
    per command boundary replays commands on the main thread and stalls
    playback on large jobs.
    """
    overlay = PlaybackOverlay()
    overlay.set_canvas(FakeCanvas())
    player = FakePlayer(n_ops=100, cmd_time=TICK_SECONDS)
    overlay.set_player(player)
    overlay._start_playback()
    player.seek_count = 0
    # Enough ticks to cross many command boundaries.
    for _ in range(int(1.0 / TICK_SECONDS) + 10):
        assert overlay._on_tick()
    assert overlay._slider.get_value() > 0
    assert player.seek_count == 0


@pytest.mark.ui
def test_scrub_while_playing_resyncs_clock_to_boundary(ui_context_initializer):
    overlay = PlaybackOverlay()
    overlay.set_canvas(FakeCanvas())
    player = FakePlayer(n_ops=100, cmd_time=1.0)
    overlay.set_player(player)
    overlay._start_playback()
    player.seek_count = 0
    # Dragging the slider while playing resyncs the clock to the
    # dragged command's completion time (command 24 -> 25 s).
    overlay._slider.set_value(24)
    assert player.current_index == 24


@pytest.mark.ui
def test_stepping_visits_every_command_exactly(ui_context_initializer):
    """Regression test for issue #370.

    Zero-duration commands share cumulative times; stepping must land
    on every command exactly once instead of skipping clusters or
    getting stuck.
    """
    overlay = PlaybackOverlay()
    overlay.set_canvas(FakeCanvas())
    durations = [0.0] * 4 + [60.0] + [0.0] * 5
    player = FakePlayer(n_ops=len(durations), cmd_times=durations)
    overlay.set_player(player)
    visited = []
    for _ in range(len(durations)):
        overlay._on_step_fwd(None)
        while overlay._step_animating:
            overlay._on_step_tick()
        visited.append(player.current_index)
    assert visited == list(range(len(durations)))
    for _ in range(len(durations)):
        overlay._on_step_back(None)
        while overlay._step_animating:
            overlay._on_step_tick()
    assert player.current_index == 0


@pytest.mark.ui
def test_stepping_skips_commands_without_gcode(ui_context_initializer):
    """Regression test for issue #370.

    Marker and state commands produce no G-code lines; stepping must
    skip them so every step selects a visible G-code line.
    """
    overlay = PlaybackOverlay()
    overlay.set_canvas(FakeCanvas())
    player = FakePlayer(n_ops=8, cmd_time=0.5)
    overlay.set_player(player)
    # Ops 1 and 3-5 produced no G-code output.
    overlay.set_op_map(
        FakeOpMap(
            [(0, 1), (1, 0), (2, 1), (3, 0), (4, 0), (5, 0), (6, 1), (7, 1)]
        )
    )
    overlay._on_step_fwd(None)
    while overlay._step_animating:
        overlay._on_step_tick()
    assert player.current_index == 0
    overlay._on_step_fwd(None)
    while overlay._step_animating:
        overlay._on_step_tick()
    # Op 1 produced no G-code and is skipped.
    assert player.current_index == 2
    overlay._on_step_back(None)
    while overlay._step_animating:
        overlay._on_step_tick()
    assert player.current_index == 0


@pytest.mark.ui
def test_stepping_without_op_map_visits_every_command(
    ui_context_initializer,
):
    overlay = PlaybackOverlay()
    overlay.set_canvas(FakeCanvas())
    player = FakePlayer(n_ops=6, cmd_time=1.0)
    player.seek(0)
    overlay.set_player(player)
    overlay._on_step_fwd(None)
    while overlay._step_animating:
        overlay._on_step_tick()
    assert player.current_index == 1


@pytest.mark.ui
def test_slider_spans_gcode_lines_when_mapped(ui_context_initializer):
    """Regression test for issue #370.

    With an op map the slider spans G-code line numbers, so stepping
    onto an early command must not push the slider past 50% just
    because the preamble contains many invisible marker commands.
    """
    overlay = PlaybackOverlay()
    overlay.set_canvas(FakeCanvas())
    # Ops 0-6 are zero-output preamble; op 7 owns "T0", op 8 "G0",
    # op 9 owns two lines.
    spans = [(0, 3)] + [(3, 0)] * 6 + [(3, 1), (4, 1), (5, 2)]
    player = FakePlayer(n_ops=len(spans), cmd_time=0.5)
    overlay.set_player(player)
    overlay.set_op_map(FakeOpMap(spans))
    # Slider extent is the last g-code line index (6), not the op count.
    assert overlay._slider.get_adjustment().get_upper() == 6
    # Pristine playhead: nothing executed yet, slider starts at zero.
    assert int(overlay._slider.get_value()) == 0
    # One step reaches op 0 ("G21"/"G90"/"G54" span); the second step
    # skips all six marker commands and lands on "T0" (op 7) - yet the
    # slider only moves from line 2 to line 3 instead of jumping ahead.
    overlay._on_step_fwd(None)
    while overlay._step_animating:
        overlay._on_step_tick()
    assert player.current_index == 0
    # Slider matches the viewer selection: op 0's action line.
    assert int(overlay._slider.get_value()) == 2
    overlay._on_step_fwd(None)
    while overlay._step_animating:
        overlay._on_step_tick()
    assert player.current_index == 7
    assert int(overlay._slider.get_value()) == 3


@pytest.mark.ui
def test_scrubbing_slider_maps_back_to_op(ui_context_initializer):
    overlay = PlaybackOverlay()
    overlay.set_canvas(FakeCanvas())
    spans = [(0, 3)] + [(3, 0)] * 2 + [(3, 1), (4, 1)]
    player = FakePlayer(n_ops=len(spans), cmd_time=1.0)
    overlay.set_player(player)
    overlay.set_op_map(FakeOpMap(spans))
    # Dragging the slider to line 4 ("G0") lands on op 4.
    overlay._slider.set_value(4)
    assert player.current_index == 4


@pytest.mark.ui
def test_stepping_reaches_slider_end(ui_context_initializer):
    """Regression test for issue #370.

    Stepping onto the last command must push the slider all the way
    to its end, matching the G-code viewer's last-line selection.
    """
    overlay = PlaybackOverlay()
    overlay.set_canvas(FakeCanvas())
    spans = [(0, 1), (1, 0), (1, 2)]
    player = FakePlayer(n_ops=len(spans), cmd_time=1.0)
    overlay.set_player(player)
    overlay.set_op_map(FakeOpMap(spans))
    assert int(overlay._slider.get_value()) == 0
    overlay._on_step_fwd(None)
    while overlay._step_animating:
        overlay._on_step_tick()
    assert player.current_index == 0
    overlay._on_step_fwd(None)
    while overlay._step_animating:
        overlay._on_step_tick()
    # Op 2 owns lines 1-2 including the last one; both the viewer
    # selection and the slider sit at the final line.
    assert player.current_index == 2
    assert int(overlay._slider.get_value()) == 2
    assert int(overlay._slider.get_value()) == overlay._slider_extent()


@pytest.mark.ui
def test_playback_completion_anchors_last_command(ui_context_initializer):
    """Regression test for issue #370 (laser stays on at job end).

    Trailing zero-duration commands (the M5 power-offs) were never
    applied when playback completed, because the state sync anchors
    at the command BEFORE the playhead. Completion must anchor the
    playhead on the final command, exactly like manual stepping.
    """
    overlay = PlaybackOverlay()
    overlay.set_canvas(FakeCanvas())
    durations = [2.0, 1.0, 0.0, 0.5, 0.0, 0.0]
    player = FakePlayer(n_ops=len(durations), cmd_times=durations)
    overlay.set_player(player)
    overlay._start_playback()
    for _ in range(int(4.0 / TICK_SECONDS) + 10):
        if not overlay._playing:
            break
        overlay._on_tick()
    assert not overlay._playing
    # The final command (a no-output M5 stand-in) is applied.
    assert player.current_index == len(durations) - 1
