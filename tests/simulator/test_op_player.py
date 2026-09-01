import math

import pytest
from raygeo.ops import Ops
from raygeo.ops.axis import Axis
from raygeo.ops.types import CommandType

from rayforge.context import RayforgeContext
from rayforge.core.doc import Doc
from rayforge.machine.kinematic_mapping import (
    KinematicMapping,
    build_layer_assembly,
)
from rayforge.machine.models.machine import Machine
from rayforge.machine.models.rotary_module import RotaryMode, RotaryModule
from rayforge.simulator.machine_state import MachineState
from rayforge.simulator.op_player import OpPlayer, build_snapshots


def _make_machine():
    ctx = RayforgeContext()
    return Machine(ctx)


def _make_ops():
    ops = Ops()
    ops.set_power(0.5)
    ops.set_feed_rate(800)
    ops.move_to(0.0, 0.0, 0.0)
    ops.line_to(10.0, 0.0, 0.0)
    ops.set_power(1.0)
    ops.line_to(10.0, 10.0, 0.0)
    ops.move_to(0.0, 0.0, 0.0)
    return ops


def test_seek_zero():
    ops = _make_ops()
    player = OpPlayer(ops, _make_machine(), Doc())
    player.seek(0)
    assert player.current_index == 0
    assert player.state.power == 0.5


def test_time_ops_drive_playback_time():
    state_ops = _make_ops()
    time_ops = Ops()
    time_ops.set_power(0.5)
    time_ops.set_feed_rate(800)
    time_ops.move_to(0.0, 0.0, 0.0)
    time_ops.line_to(1.0, 0.0, 0.0)
    player = OpPlayer(state_ops, _make_machine(), Doc(), time_ops=time_ops)
    # Cumulative time reflects the short move of the time ops, not the
    # long move of the state ops (command index 3 is the first cut).
    state_cum = state_ops.get_cumulative_time_at(3, 800.0, 3000.0, 1000.0)
    time_cum = time_ops.get_cumulative_time_at(3, 800.0, 3000.0, 1000.0)
    assert time_cum < state_cum
    assert player.get_cumulative_time(3) == pytest.approx(time_cum)
    # Command lookup at a simulated time also follows the time ops.
    mid = (time_cum + state_cum) / 2.0
    assert player.find_index_at_sim_time(mid) == 3
    # State still advances from the state ops.
    player.seek(3)
    assert player.state.axes[Axis.X] == 10.0


def test_advance_from_start():
    ops = _make_ops()
    player = OpPlayer(ops, _make_machine(), Doc())
    player.advance_to(2)
    player.advance_to(3)
    assert player.current_index == 3
    assert player.state.axes[Axis.X] == 10.0
    assert player.state.axes[Axis.Y] == 0.0


def test_seek_forward_then_backward_replays():
    ops = _make_ops()
    player = OpPlayer(ops, _make_machine(), Doc())
    player.seek(6)
    assert player.state.axes[Axis.X] == 0.0
    assert player.state.power == 1.0

    player.seek(5)
    assert player.state.axes[Axis.X] == 10.0
    assert player.state.axes[Axis.Y] == 10.0


def test_seek_then_advance():
    ops = _make_ops()
    player = OpPlayer(ops, _make_machine(), Doc())
    player.seek(3)
    player.advance_to(6)
    assert player.current_index == 6
    assert player.state.axes[Axis.X] == 0.0


def test_sync_state_to_playhead_advances_to_playhead():
    ops = _make_ops()
    player = OpPlayer(ops, _make_machine(), Doc())
    player.seek(0)
    t = player.get_cumulative_time(4)
    player.set_sim_time(t)
    player.sync_state_to_playhead()
    p, _frac = player.playback_progress()
    reference = OpPlayer(ops, _make_machine(), Doc())
    reference.seek(max(p - 1, 0))
    assert player.current_index == reference.current_index
    assert player.state.axes == reference.state.axes


def test_sync_state_to_playhead_scrubs_backwards():
    ops = _make_ops()
    player = OpPlayer(ops, _make_machine(), Doc())
    player.seek(6)
    t = player.get_cumulative_time(2)
    player.set_sim_time(t)
    player.sync_state_to_playhead()
    p, _frac = player.playback_progress()
    reference = OpPlayer(ops, _make_machine(), Doc())
    reference.seek(max(p - 1, 0))
    assert player.current_index == reference.current_index
    assert player.state.axes == reference.state.axes


def test_advance_backwards_raises():
    ops = _make_ops()
    player = OpPlayer(ops, _make_machine(), Doc())
    player.advance_to(3)
    with pytest.raises(ValueError, match="Cannot advance backwards"):
        player.advance_to(2)


def test_seek_out_of_range_raises():
    ops = _make_ops()
    player = OpPlayer(ops, _make_machine(), Doc())
    with pytest.raises(IndexError):
        player.seek(999)


def test_set_playhead_anchors_at_command_boundary():
    """Regression test for issue #370 stepping sync.

    Commands 0-2 of ``_make_ops`` take zero simulated time, so
    commands 0, 1 and 2 all share cumulative time 0. Anchoring by
    index must land exactly on the stepped-to command instead of
    letting the ambiguous time lookup skip ahead.
    """
    ops = _make_ops()
    player = OpPlayer(ops, _make_machine(), Doc())
    player.set_playhead(2)
    assert player.current_index == 2
    assert player.sim_time == pytest.approx(0.0)
    # Progress is anchored to command 3 (the first cut) at frac 0:
    # nothing of it is revealed yet.
    assert player.playback_progress() == (3, 0.0)
    assert player.state.axes[Axis.X] == 0.0
    assert player.state.power == 0.5

    # Stepping onto a zero-duration command right after a move: state
    # includes it, progress anchors to the next cut with no reveal.
    player.set_playhead(4)
    assert player.current_index == 4
    assert player.playback_progress() == (5, 0.0)

    # Stepping onto the final cut reveals everything up to it.
    player.set_playhead(5)
    assert player.current_index == 5
    assert player.state.axes[Axis.Y] == 10.0
    p, frac = player.playback_progress()
    assert (p, frac) == (6, 0.0)


def test_set_playhead_backwards_recovers_state():
    ops = _make_ops()
    player = OpPlayer(ops, _make_machine(), Doc())
    player.set_playhead(5)
    player.set_playhead(2)
    assert player.current_index == 2
    assert player.state.axes[Axis.X] == 0.0
    assert player.state.power == 0.5


def test_set_progress_anchor_keeps_fraction():
    ops = _make_ops()
    player = OpPlayer(ops, _make_machine(), Doc())
    t_start = player.get_cumulative_time(2)
    t_end = player.get_cumulative_time(3)
    mid = (t_start + t_end) / 2.0
    player.set_progress_anchor(2, mid)
    assert player.current_index == 2
    p, frac = player.playback_progress()
    assert p == 3
    assert frac == pytest.approx(0.5)
    # State reflects completed commands only.
    assert player.state.axes[Axis.X] == 0.0


def _make_step_ops():
    """Ops mirroring a simple two-pass cut job with state changes.

    Layout: preamble states, a rapid, a laser-on state, a cut, a
    zero-duration filler, another cut, two laser-off states, and a
    rapid home - like G-code with M3/M5 commands around moves.
    """
    ops = Ops()
    ops.set_power(0.0)
    ops.set_feed_rate(500)
    ops.move_to(83.877, 60.0)
    ops.set_power(0.8)
    ops.line_to(36.123, 60.0)
    ops.set_feed_rate(500)
    ops.line_to(83.877, 60.0)
    ops.set_power(0.0)
    ops.set_power(0.0)
    ops.move_to(0.0, 0.0)
    return ops


def test_set_playhead_steps_through_zero_duration_commands():
    """Stepping must land exactly one command at a time.

    Zero-duration state changes share cumulative times with their
    neighbours; index-anchored playhead placement keeps the reveal and
    the reported position in sync with the stepped-to command.
    """
    ops = _make_step_ops()
    n = ops.len()
    player = OpPlayer(ops, _make_machine(), Doc())

    seen = []
    for index in range(n):
        player.set_playhead(index)
        p, frac = player.playback_progress()
        seen.append((index, player.current_index, p))
        if index + 1 < n:
            # Progress anchored to the next command, not yet revealed.
            assert player.current_index == index
            assert p == index + 1
            assert frac == pytest.approx(0.0)
    assert seen[0] == (0, 0, 1)

    # On the laser-on state: neither cut may be revealed yet.
    player.set_playhead(3)
    p, _frac = player.playback_progress()
    assert p == 4

    # After the first cut: exactly the first cut is done, not both.
    player.set_playhead(5)
    p, _frac = player.playback_progress()
    assert p == 6

    # Stepping backwards lands exactly, restoring earlier state.
    player.set_playhead(7)
    assert player.current_index == 7
    assert player.state.power == 0.0
    player.set_playhead(4)
    assert player.current_index == 4
    assert player.state.power == 0.8


def test_advance_out_of_range_raises():
    ops = _make_ops()
    player = OpPlayer(ops, _make_machine(), Doc())
    with pytest.raises(IndexError):
        player.advance_to(999)


def test_empty_ops_raises():
    with pytest.raises(ValueError):
        OpPlayer(Ops(), _make_machine(), Doc())


def test_none_ops_raises():
    with pytest.raises(ValueError):
        OpPlayer(None, _make_machine(), Doc())  # type: ignore[arg-type]


def test_seek_last_movement():
    ops = _make_ops()
    player = OpPlayer(ops, _make_machine(), Doc())
    last = player.seek_last_movement()
    assert last == 6
    assert player.state.axes[Axis.X] == 0.0
    assert player.state.axes[Axis.Y] == 0.0


def test_skip_snapshot_build_then_set_snapshots():
    machine = _make_machine()
    doc = Doc()
    ops = Ops()
    ops.set_power(0.3)
    ops.move_to(5.0, 5.0, 0.0)
    ops.line_to(15.0, 25.0, 3.0)

    player = OpPlayer(ops, machine, doc, build_snapshots=False)
    assert player.snapshots == []

    reference = OpPlayer(ops, machine, doc)
    reference.seek(ops.len() - 1)
    player.set_snapshots(reference.snapshots)
    assert player.snapshots is reference.snapshots

    player.seek(ops.len() - 1)
    assert player.state.axes == reference.state.axes


def test_snapshots_replaced_after_build():
    machine = _make_machine()
    doc = Doc()
    ops = Ops()
    ops.set_power(0.3)
    ops.move_to(5.0, 5.0, 0.0)
    ops.line_to(15.0, 25.0, 3.0)

    player = OpPlayer(ops, machine, doc, build_snapshots=False)
    player.set_snapshots([])
    assert player.snapshots == []


def test_random_access_matches_sequential():
    machine = _make_machine()
    doc = Doc()
    ops = Ops()
    ops.set_power(0.3)
    ops.move_to(5.0, 5.0, 0.0)
    ops.set_power(0.7)
    ops.line_to(15.0, 25.0, 3.0)
    ops.set_feed_rate(1200)
    ops.line_to(50.0, 60.0, 0.0)

    sequential = OpPlayer(ops, machine, doc)
    sequential.seek(ops.len() - 1)

    player = OpPlayer(ops, machine, doc)
    player.seek(1)
    player.seek(ops.len() - 1)

    assert player.state.axes == sequential.state.axes
    assert player.state.power == sequential.state.power
    assert player.state.cut_speed == sequential.state.cut_speed


def test_scanline_tracked():
    ops = Ops()
    ops.move_to(0.0, 0.0, 0.0)
    ops.scan_to(10.0, 0.0, 0.0, bytearray([100, 200]))
    ops.line_to(20.0, 0.0, 0.0)

    player = OpPlayer(ops, _make_machine(), Doc())
    player.seek(2)

    assert 1 in player.state.reached_textures
    assert 0 not in player.state.reached_textures
    assert 2 not in player.state.reached_textures


def test_default_source_axis_is_y():
    ops = _make_ops()
    player = OpPlayer(ops, _make_machine(), Doc())
    assert player._source_axis == Axis.Y


def test_seek_resets_source_axis():
    ops = _make_ops()
    player = OpPlayer(ops, _make_machine(), Doc())
    player._source_axis = Axis.X
    player.seek(0)
    assert player._source_axis == Axis.Y


def test_replacement_mode_no_rotary_mapping():
    machine = _make_machine()
    rm = RotaryModule()
    rm.set_mode(RotaryMode.AXIS_REPLACEMENT)
    machine.add_rotary_module(rm)

    ops = Ops()
    ops.move_to(0, 0, 0)
    ops.layer_start("test")
    ops.line_to(10, 20, 0)

    doc = Doc()
    doc.active_layer.uid = "test"
    doc.active_layer.set_rotary_enabled(True)
    doc.active_layer.set_rotary_module_uid(rm.uid)

    player = OpPlayer(ops, machine, doc)
    player.seek(ops.len() - 1)

    assert player.state.axes.get(Axis.A, 0.0) == pytest.approx(0.0)


def test_true_4th_axis_copies_to_rotary():
    from rayforge.machine.kinematic_mapping import KinematicMapping

    machine = _make_machine()
    rm = RotaryModule()
    rm.set_mode(RotaryMode.TRUE_4TH_AXIS)
    rm.set_axis(Axis.A)
    machine.add_rotary_module(rm)

    ops = Ops()
    ops.move_to(0, 0, 0)
    ops.layer_start("test")
    ops.line_to(10, 20, 0)

    mapping = KinematicMapping(
        rotary_axis=Axis.A,
        diameter=25.0,
    )
    mapping.apply(ops)

    doc = Doc()
    doc.active_layer.uid = "test"
    doc.active_layer.set_rotary_enabled(True)
    doc.active_layer.set_rotary_module_uid(rm.uid)

    player = OpPlayer(ops, machine, doc)
    player.seek(ops.len() - 1)

    diameter = 25.0
    expected_deg = (20.0 / (diameter * math.pi)) * 360.0
    assert player.state.axes[Axis.A] == pytest.approx(expected_deg)


def _make_rotary_doc(uid, machine):
    rm = RotaryModule()
    rm.default_diameter = 40.0
    machine.add_rotary_module(rm)
    doc = Doc()
    doc.active_layer.uid = uid
    doc.active_layer.set_rotary_enabled(True)
    doc.active_layer.set_rotary_module_uid(rm.uid)
    return doc


def test_seek_into_layer_emits_layer_changed():
    machine = _make_machine()
    doc = _make_rotary_doc("test", machine)

    ops = Ops()
    ops.move_to(0, 0, 0)
    ops.layer_start("test")
    ops.line_to(10, 20, 0)

    player = OpPlayer(ops, machine, doc)
    received = []

    def _on_layer_changed(sender, **kw):
        received.append(kw["layer_uid"])

    player.layer_changed.connect(_on_layer_changed)
    player.seek(ops.len() - 1)
    assert "test" in received


def test_seek_preamble_effective_layer_is_first_layer():
    machine = _make_machine()
    doc = _make_rotary_doc("test", machine)

    ops = Ops()
    ops.move_to(5, 5, 0)
    ops.set_power(0.5)
    ops.layer_start("test")
    ops.line_to(10, 20, 0)

    player = OpPlayer(ops, machine, doc)
    player.seek(0)
    assert player.get_effective_layer(doc) is doc.layers[0]


def test_seek_flat_layer_builds_flat_assembly():
    machine = _make_machine()
    rm = RotaryModule()
    machine.add_rotary_module(rm)
    doc = Doc()
    doc.active_layer.uid = "test"

    ops = Ops()
    ops.move_to(0, 0, 0)
    ops.layer_start("test")
    ops.line_to(10, 20, 0)

    player = OpPlayer(ops, machine, doc)
    player.seek(ops.len() - 1)
    assembly = build_layer_assembly(machine, player.get_effective_layer(doc))
    assert not assembly.has_rotary
    assert not machine._layer_configured


def _make_long_ops(n=1000):
    ops = Ops()
    ops.set_power(0.5)
    ops.move_to(0.0, 0.0, 0.0)
    for i in range(n):
        ops.line_to(float(i % 20), float((i // 20) % 20), 0.0)
    return ops


def test_build_snapshots_empty_for_short_ops():
    ops = _make_long_ops(n=10)
    assert build_snapshots(ops, _make_machine(), Doc()) == []


def test_build_snapshots_targets_spaced_every_interval():
    ops = _make_long_ops(n=2500)
    snapshots = build_snapshots(ops, _make_machine(), Doc())
    assert [s[0] for s in snapshots] == [1000, 2000]
    for target, state, source, rotary, dpm in snapshots:
        assert isinstance(target, int)
        assert isinstance(state, MachineState)
        assert source == Axis.Y
        assert rotary is None


def test_build_snapshots_matches_sequential_playback():
    machine = _make_machine()
    doc = Doc()
    ops = _make_long_ops(n=1000)

    sequential = OpPlayer(ops, machine, doc)
    sequential.seek(ops.len() - 1)

    player = OpPlayer(ops, machine, doc, build_snapshots=False)
    player.set_snapshots(build_snapshots(ops, machine, doc))
    player.seek(ops.len() - 1)

    assert player.state.axes == sequential.state.axes
    assert player.state.power == sequential.state.power
    assert player.state.cut_speed == sequential.state.cut_speed


def test_build_snapshots_clears_reached_textures():
    ops = Ops()
    ops.move_to(0.0, 0.0, 0.0)
    for i in range(1000):
        ops.scan_to(float(i % 20), 0.0, 0.0, bytearray([i % 255]))
    ops.line_to(50.0, 0.0, 0.0)

    snapshots = build_snapshots(ops, _make_machine(), Doc())
    assert snapshots
    for _, state, _, _, _ in snapshots:
        assert len(state.reached_textures) == 0


# --- playback clock helpers ---


def test_playback_params_defaults_match_raygeo():
    ops = _make_ops()
    player = OpPlayer(ops, _make_machine(), Doc())
    assert player.get_cumulative_time(0) >= 0.0


def test_set_playback_params_changes_cumulative_time():
    ops = _make_ops()
    player = OpPlayer(ops, _make_machine(), Doc())
    player.set_playback_params(600.0, 1200.0, 0.0)
    # cmd3 line_to(10, 0) at 800mm/min feed (from ops) = 0.75s
    assert player.get_cumulative_time(3) == pytest.approx(0.75, abs=1e-9)
    assert player.get_cumulative_time(4) == pytest.approx(0.75, abs=1e-9)
    # cmd5 line_to(10, 10) at 800mm/min = 0.75s more
    assert player.get_cumulative_time(5) == pytest.approx(1.5, abs=1e-9)
    # cmd6 move_to(0, 0) at 1200mm/min rapid = 14.14mm / 20mm/s
    assert player.get_cumulative_time(6) == pytest.approx(
        1.5 + math.sqrt(200.0) / 20.0, abs=1e-9
    )


def test_get_cumulative_time_out_of_range_clamps():
    ops = _make_ops()
    player = OpPlayer(ops, _make_machine(), Doc())
    player.set_playback_params(600.0, 1200.0, 0.0)
    total = player.get_cumulative_time(6)
    assert player.get_cumulative_time(999) == pytest.approx(total, abs=1e-9)


def test_find_index_at_sim_time_matches_cumulative():
    ops = _make_ops()
    player = OpPlayer(ops, _make_machine(), Doc())
    player.set_playback_params(600.0, 1200.0, 0.0)
    # cum: [0, 0, 0, 0.75, 0.75, 1.5, 2.207]
    assert player.find_index_at_sim_time(0.0) == 2
    assert player.find_index_at_sim_time(0.5) == 2
    assert player.find_index_at_sim_time(0.75) == 4
    assert player.find_index_at_sim_time(1.0) == 4
    assert player.find_index_at_sim_time(1.5) == 5
    assert player.find_index_at_sim_time(100.0) == 6


def test_find_index_at_sim_time_before_first_completion():
    ops = Ops()
    ops.line_to(10.0, 0.0, 0.0)  # 10mm at 600mm/min = 1.0s
    player = OpPlayer(ops, _make_machine(), Doc())
    player.set_playback_params(600.0, 1200.0, 0.0)
    assert player.get_cumulative_time(0) == pytest.approx(1.0, abs=1e-9)
    assert player.find_index_at_sim_time(0.5) == 0
    assert player.find_index_at_sim_time(1.0) == 0
    assert player.find_index_at_sim_time(1.5) == 0


def test_find_index_at_sim_time_with_acceleration():
    ops = _make_ops()
    player = OpPlayer(ops, _make_machine(), Doc())
    player.set_playback_params(600.0, 1200.0, 1000.0)
    # Acceleration slows every move: cumulative time grows.
    no_accel = OpPlayer(ops, _make_machine(), Doc())
    no_accel.set_playback_params(600.0, 1200.0, 0.0)
    assert player.get_cumulative_time(6) > no_accel.get_cumulative_time(6)


# --- playback progress (fractional playhead) ---


def _make_progress_player():
    player = OpPlayer(_make_ops(), _make_machine(), Doc())
    player.set_playback_params(600.0, 1200.0, 0.0)
    return player


def test_playback_progress_mid_command():
    player = _make_progress_player()
    player.set_sim_time(0.375)  # halfway through the 0.75s cmd3 cut
    assert player.playback_progress() == (3, pytest.approx(0.5))


def test_playback_progress_at_command_boundary():
    player = _make_progress_player()
    player.set_sim_time(0.75)  # cmd3 just completed
    p, frac = player.playback_progress()
    assert p == 5
    assert frac == pytest.approx(0.0)


def test_playback_progress_after_end():
    player = _make_progress_player()
    player.set_sim_time(100.0)
    p, frac = player.playback_progress()
    assert p == 6
    assert frac == pytest.approx(1.0)


def test_playback_progress_before_first_completion():
    ops = Ops()
    ops.line_to(10.0, 0.0, 0.0)  # 10mm at 600mm/min = 1.0s
    player = OpPlayer(ops, _make_machine(), Doc())
    player.set_playback_params(600.0, 1200.0, 0.0)
    player.set_sim_time(0.5)
    p, frac = player.playback_progress()
    assert p == 0
    assert frac == pytest.approx(0.5)


def test_render_state_interpolates_within_command():
    player = _make_progress_player()
    player.seek(2)  # state at (0, 0) after move_to
    player.set_sim_time(0.375)  # halfway through cmd3 cut to (10, 0)
    state = player.render_state()
    assert state.axes[Axis.X] == pytest.approx(5.0)
    assert state.axes[Axis.Y] == pytest.approx(0.0)
    assert state.axes[Axis.Z] == pytest.approx(0.0)


def test_render_state_returns_state_at_boundary():
    player = _make_progress_player()
    player.seek(3)
    player.set_sim_time(0.75)  # exactly at a command boundary
    assert player.render_state() is player.state


def test_render_state_returns_state_for_non_moving_command():
    ops = Ops()
    ops.set_feed_rate(600)  # 10 mm/s
    ops.move_to(0.0, 0.0, 0.0)
    ops.line_to(10.0, 0.0, 0.0)  # 1s
    ops.dwell(1000.0)  # 1s dwell
    player = OpPlayer(ops, _make_machine(), Doc())
    player.set_playback_params(600.0, 1200.0, 0.0)
    player.seek(2)  # state at (10, 0) after the line
    # cum: [0, 0, 1.0, 2.0]; at t=1.5 the dwell (idx 3) is in progress.
    player.set_sim_time(1.5)
    p, frac = player.playback_progress()
    assert p == 3
    assert frac == pytest.approx(0.5)
    # The in-progress command does not move: the plain state is used.
    assert player.render_state() is player.state
    assert player.state.axes[Axis.X] == pytest.approx(10.0)


def test_render_state_from_home_before_first_completion():
    ops = Ops()
    ops.line_to(10.0, 0.0, 0.0)  # 10mm at 600mm/min = 1.0s
    player = OpPlayer(ops, _make_machine(), Doc())
    player.set_playback_params(600.0, 1200.0, 0.0)
    player.set_sim_time(0.5)
    state = player.render_state()
    home_x = player._home_axes[Axis.X]
    assert state.axes[Axis.X] == pytest.approx(home_x + 0.5 * (10.0 - home_x))


def test_render_state_paused_returns_plain_state():
    player = _make_progress_player()
    player.seek(3)
    player.set_sim_time(0.0)
    assert player.render_state() is player.state


def test_render_state_laser_on_during_cut():
    player = _make_progress_player()
    player.seek(2)  # state after move_to
    player.set_sim_time(0.375)  # halfway through the cmd3 cut
    state = player.render_state()
    assert state.laser_on is True


def test_render_state_laser_off_during_travel():
    player = _make_progress_player()
    player.seek(5)
    player.set_sim_time(1.75)  # midway through the final move_to travel
    state = player.render_state()
    assert state.laser_on is False


def _make_replacement_player():
    """Player over AXIS_REPLACEMENT mapped ops with a rotation move.

    The mapped ops keep endpoint Y at the cylinder position while the
    real rotation lives in extra_axes[Y]; the pure-X travel back to
    zero rotation afterwards exercises the extra-axis interpolation.
    """
    machine = _make_machine()
    rm = RotaryModule()
    rm.set_mode(RotaryMode.AXIS_REPLACEMENT)
    rm.set_axis(Axis.Y)
    machine.add_rotary_module(rm)

    ops = Ops()
    ops.set_feed_rate(600)  # 10 mm/s
    ops.move_to(0.0, 0.0, 0.0)
    ops.layer_start("test")
    ops.line_to(10.0, 10.0, 0.0)  # 10 mm of circumference -> 90 deg
    ops.move_to(20.0, 0.0, 0.0)  # pure-X travel back to 0 deg
    ops.layer_end("test")

    diameter = 40.0
    mapping = KinematicMapping.from_rotary_module(rm, diameter)
    assert mapping is not None
    mapping.apply(ops)

    doc = Doc()
    doc.active_layer.uid = "test"
    doc.active_layer.set_rotary_enabled(True)
    doc.active_layer.set_rotary_diameter(diameter)
    doc.active_layer.set_rotary_module_uid(rm.uid)

    player = OpPlayer(ops, machine, doc)
    player.set_playback_params(600.0, 1200.0, 0.0)
    return player, diameter


def _rotation_command_index(player):
    """Command index of the 90-degree rotation move (line_to)."""
    for i in range(player.ops.len()):
        if player.ops.command_type(i) == CommandType.LINE_TO:
            return i
    raise AssertionError("expected a line_to command")


def test_render_state_interpolates_rotary_axis_linearly():
    """The rotary axis must interpolate linearly between commands.

    Regression test: in AXIS_REPLACEMENT mode the rotary axis is Y, which
    also receives the endpoint-Y interpolation. Interpolating the extra
    axis from the partially-interpolated value produced a quadratic curve
    (the cylinder "arced" between correct endpoints). The rotation must
    instead be blended from the pre-command state.
    """
    player, diameter = _make_replacement_player()
    rot = _rotation_command_index(player)
    player.seek(rot)  # state now holds the 90 deg rotation
    expected_deg = (10.0 / (diameter * math.pi)) * 360.0
    assert player.state.axes[Axis.Y] == pytest.approx(expected_deg)

    travel = rot + 1
    t0 = player.get_cumulative_time(rot)
    t1 = player.get_cumulative_time(travel)
    player.set_sim_time(t0 + 0.5 * (t1 - t0))
    p, frac = player.playback_progress()
    assert p == travel
    assert frac == pytest.approx(0.5)
    state = player.render_state()
    assert state.axes[Axis.X] == pytest.approx(15.0)
    assert state.axes[Axis.Y] == pytest.approx(expected_deg * 0.5)


def test_render_state_keeps_rotary_axis_constant_for_pure_x():
    """A pure-X move must not spin the cylinder during the glide.

    Regression test: when the rotary axis (Y) collides with a non-zero
    endpoint Y (cylinder position), the old interpolation blended the two
    and made the cylinder rotate out-and-back during a pure-X line.
    """
    player, diameter = _make_replacement_player()
    rot = _rotation_command_index(player)
    player.seek(rot)
    travel = rot + 1
    t0 = player.get_cumulative_time(rot)
    t1 = player.get_cumulative_time(travel)
    player.set_sim_time(t0 + 0.25 * (t1 - t0))
    p, frac = player.playback_progress()
    assert p == travel
    assert frac == pytest.approx(0.25)
    state = player.render_state()
    expected_deg = (10.0 / (diameter * math.pi)) * 360.0
    assert state.axes[Axis.Y] == pytest.approx(expected_deg * 0.75)


def test_render_state_laser_on_for_whole_cut():
    player = _make_progress_player()
    player.seek(2)
    # The cut spans cum[2]=0 to cum[3]=0.75; laser must be on at the
    # very start of the cut, not only once the cut command completes.
    player.set_sim_time(0.01)
    assert player.render_state().laser_on is True
    player.set_sim_time(0.74)
    assert player.render_state().laser_on is True


def _make_arc_player():
    """Player with a full-circle arc command."""
    ops = Ops()
    ops.set_feed_rate(600)  # 10 mm/s
    ops.move_to(0.0, 0.0, 0.0)
    ops.arc_to(0.0, 0.0, 10.0, 0.0, True)  # full circle CW, r=10
    # Arc circumference = 2*pi*10 ~ 62.83mm at 10mm/s ~ 6.283s
    return ops


def _make_rotary_arc_player():
    """Player with an arc in AXIS_REPLACEMENT rotary mode.

    Creates a half-circle arc on a cylinder.  The arc's Y displacement
    in Cartesian space should cause the rotary axis to oscillate.
    """
    machine = _make_machine()
    rm = RotaryModule()
    rm.set_mode(RotaryMode.AXIS_REPLACEMENT)
    rm.set_axis(Axis.Y)
    machine.add_rotary_module(rm)

    diameter = 40.0

    ops = Ops()
    ops.set_feed_rate(600)
    ops.move_to(0.0, 5.0, 0.0)
    ops.layer_start("test")
    # Half-circle arc from (0,5) to (10,5), center (5,5), radius 5,
    # CCW.  In Cartesian space Y ranges from 5 to 10.
    ops.arc_to(10.0, 5.0, 5.0, 0.0, False)
    ops.layer_end("test")

    mapping = KinematicMapping.from_rotary_module(rm, diameter)
    assert mapping is not None
    mapping.apply(ops)

    doc = Doc()
    doc.active_layer.uid = "test"
    doc.active_layer.set_rotary_enabled(True)
    doc.active_layer.set_rotary_diameter(diameter)
    doc.active_layer.set_rotary_module_uid(rm.uid)

    player = OpPlayer(ops, machine, doc)
    player.set_playback_params(600.0, 3000.0, 0.0)
    return player, diameter


def _arc_cum(ops, idx):
    """Cumulative time at command *idx* for feed=600, rapid=3000."""
    return ops.get_cumulative_time_at(idx, 600.0, 3000.0, 0.0)


def test_render_state_full_circle_interpolates_along_arc():
    """Full circle (start == end) must trace the arc, not stay put."""
    ops = _make_arc_player()
    player = OpPlayer(ops, _make_machine(), Doc())
    player.set_playback_params(600.0, 3000.0, 0.0)
    player.seek(1)  # state at (0, 0) after move_to
    # The arc is command 2. Its duration is cum[2]-cum[1].
    arc_dur = _arc_cum(ops, 2) - _arc_cum(ops, 1)
    # Quarter way through the circle.
    player.set_sim_time(_arc_cum(ops, 1) + 0.25 * arc_dur)
    p, _frac = player.playback_progress()
    assert p == 2
    state = player.render_state()
    # Quarter CW circle from (0,0) around center (10,0):
    # start angle = atan2(0-0, 0-10) = pi
    # sweep = -2pi, frac=0.25 → angle = pi - pi/2 = pi/2
    # pos = (10 + 10*cos(pi/2), 0 + 10*sin(pi/2)) = (10, 10)
    assert state.axes[Axis.X] == pytest.approx(10.0, abs=1e-6)
    assert state.axes[Axis.Y] == pytest.approx(10.0, abs=1e-6)


def test_render_state_full_circle_halfway():
    """Halfway through a full circle the head is diametrically opposite."""
    ops = _make_arc_player()
    player = OpPlayer(ops, _make_machine(), Doc())
    player.set_playback_params(600.0, 3000.0, 0.0)
    player.seek(1)
    arc_dur = _arc_cum(ops, 2) - _arc_cum(ops, 1)
    player.set_sim_time(_arc_cum(ops, 1) + 0.5 * arc_dur)
    p, _frac = player.playback_progress()
    assert p == 2
    state = player.render_state()
    # Half CW circle from (0,0) around center (10,0):
    # start angle = pi, sweep = -2pi, frac=0.5 → angle = pi - pi = 0
    # pos = (10 + 10*cos(0), 0 + 10*sin(0)) = (20, 0)
    assert state.axes[Axis.X] == pytest.approx(20.0, abs=1e-6)
    assert state.axes[Axis.Y] == pytest.approx(0.0, abs=1e-6)


def test_render_state_partial_arc_interpolates_along_curve():
    """Full circle (start == end) must trace the arc, not stay put.

    This is a regression test for the linearization bug where arc
    commands were linearly interpolated between start and end points.
    """
    ops = _make_arc_player()
    player = OpPlayer(ops, _make_machine(), Doc())
    player.set_playback_params(600.0, 3000.0, 0.0)
    player.seek(1)
    arc_dur = _arc_cum(ops, 2) - _arc_cum(ops, 1)
    # At 75% through the full circle CW:
    # start angle = pi, sweep = -2pi, frac=0.75 → angle = pi - 3pi/2 = -pi/2
    # pos = (10 + 10*cos(-pi/2), 0 + 10*sin(-pi/2)) = (10, -10)
    player.set_sim_time(_arc_cum(ops, 1) + 0.75 * arc_dur)
    state = player.render_state()
    assert state.axes[Axis.X] == pytest.approx(10.0, abs=1e-6)
    assert state.axes[Axis.Y] == pytest.approx(-10.0, abs=1e-6)


def test_render_state_partial_arc_ccw():
    """CCW partial arc interpolates along the curve."""
    ops = Ops()
    ops.set_feed_rate(600)
    ops.move_to(0.0, -10.0, 0.0)
    # Quarter circle CCW from (0,-10) to (10,0), center offset (0,10) →
    # absolute center (0,0), radius 10.
    ops.arc_to(10.0, 0.0, 0.0, 10.0, False)
    player = OpPlayer(ops, _make_machine(), Doc())
    player.set_playback_params(600.0, 3000.0, 0.0)
    player.seek(1)
    arc_dur = _arc_cum(ops, 2) - _arc_cum(ops, 1)
    player.set_sim_time(_arc_cum(ops, 1) + 0.5 * arc_dur)
    state = player.render_state()
    # start angle: atan2(-10-0, 0-0) = -pi/2
    # end angle: atan2(0-0, 10-0) = 0
    # sweep_ccw = (0 - (-pi/2)) % 2pi = pi/2
    # CCW → sweep = pi/2
    # At frac=0.5: angle = -pi/2 + 0.5 * pi/2 = -pi/4
    # pos = (10*cos(-pi/4), 10*sin(-pi/4)) = (7.071, -7.071)
    assert state.axes[Axis.X] == pytest.approx(
        10.0 * math.cos(-math.pi / 4), abs=1e-6
    )
    assert state.axes[Axis.Y] == pytest.approx(
        10.0 * math.sin(-math.pi / 4), abs=1e-6
    )


def test_render_state_partial_arc_cw():
    """CW partial arc interpolates along the curve."""
    ops = Ops()
    ops.set_feed_rate(600)
    ops.move_to(0.0, 10.0, 0.0)
    # Quarter circle CW from (0,10) to (10,0), center offset (0,-10) →
    # absolute center (0,0), radius 10.
    ops.arc_to(10.0, 0.0, 0.0, -10.0, True)
    player = OpPlayer(ops, _make_machine(), Doc())
    player.set_playback_params(600.0, 3000.0, 0.0)
    player.seek(1)
    arc_dur = _arc_cum(ops, 2) - _arc_cum(ops, 1)
    player.set_sim_time(_arc_cum(ops, 1) + 0.5 * arc_dur)
    state = player.render_state()
    # start angle: atan2(10-0, 0-0) = pi/2
    # end angle: atan2(0-0, 10-0) = 0
    # sweep_ccw = (0 - pi/2) % 2pi = 3pi/2
    # CW → sweep = 3pi/2 - 2pi = -pi/2
    # At frac=0.5: angle = pi/2 + 0.5 * (-pi/2) = pi/4
    # pos = (10*cos(pi/4), 10*sin(pi/4)) = (7.071, 7.071)
    assert state.axes[Axis.X] == pytest.approx(
        10.0 * math.cos(math.pi / 4), abs=1e-6
    )
    assert state.axes[Axis.Y] == pytest.approx(
        10.0 * math.sin(math.pi / 4), abs=1e-6
    )


def test_render_state_rotary_arc_rotation_oscillates():
    """Rotary axis changes during an arc on a cylinder.

    Regression test: the rotary angle was constant during arcs because
    the arc geometry mixed Cartesian Y with the rotary angle.  The
    rotation should follow the arc's Cartesian Y displacement,
    converted via the degrees-per-mm factor.
    """
    player, _diameter = _make_rotary_arc_player()
    # Arc is command index 3 (after set_feed_rate, move_to, layer_start).
    cum2 = _arc_cum(player.ops, 2)
    cum3 = _arc_cum(player.ops, 3)
    arc_dur = cum3 - cum2
    # The arc is a half-circle CCW from (0,5) to (10,5) around
    # center (5,5).  At the midpoint (frac=0.5), the arc passes
    # through Y=0 (bottom of circle), so the rotary angle should
    # be LOWER than at the start.
    player.seek(2)
    t_mid = cum2 + 0.5 * arc_dur
    player.set_sim_time(t_mid)
    state_mid = player.render_state()
    rot_mid = state_mid.axes[Axis.Y]

    # At start (frac=0), rotation should be at the initial value.
    player.set_sim_time(cum2 + 0.01 * arc_dur)
    state_start = player.render_state()
    rot_start = state_start.axes[Axis.Y]

    # The arc dips to Y=0 at the midpoint, so rotary angle is lower.
    assert rot_mid < rot_start


def _make_true4th_broken_frame_player():
    """Player over TRUE_4TH_AXIS mapped ops whose arcs lost their frame.

    Mirrors a wrapped-cylinder cut with four consecutive arcs (I/J from
    the unwrapped geometry): mapping collapses endpoint Y onto the
    cylinder position, detaching each stored circle from its
    endpoints.
    """
    machine = _make_machine()
    rm = RotaryModule()
    rm.set_mode(RotaryMode.TRUE_4TH_AXIS)
    rm.set_axis(Axis.A)
    rm.default_diameter = 40.0
    machine.add_rotary_module(rm)

    arcs = [
        # (x_end, a_end_deg, i, j) like the reported gcode.
        (29.859, -110.648, 14.165, 15.021),
        (29.522, -92.459, 75.975, 24.224),
        (29.438, -53.874, 311.838, 26.596),
        (29.944, -19.149, 101.253, 1.595),
    ]
    circumference = math.pi * 40.0
    a_start = -128.0
    y = a_start / 360.0 * circumference

    def build():
        seq = Ops()
        seq.set_feed_rate(60)
        seq.move_to(29.0, y, 0.0)
        seq.layer_start("test")
        for x_end, a_end, i, j in arcs:
            seq.arc_to(x_end, a_end / 360.0 * circumference, i, j, True)
        seq.layer_end("test")
        return seq

    # Like the pipeline: playback walks the mapped ops while the time
    # model uses the raw unwrapped ones.
    time_ops = build()
    ops = build()
    KinematicMapping(rotary_axis=Axis.A, diameter=40.0).apply(ops)

    doc = Doc()
    doc.active_layer.uid = "test"
    doc.active_layer.set_rotary_enabled(True)
    doc.active_layer.set_rotary_diameter(40.0)
    doc.active_layer.set_rotary_module_uid(rm.uid)

    player = OpPlayer(
        ops, machine, doc, build_snapshots=False, time_ops=time_ops
    )
    player.set_playback_params(500.0, 3000.0, 0.0)
    return player, arcs


def test_rotary_arc_detached_frame_interpolates_linearly():
    """Regression test for issue #370 (rotary follow-up).

    Rotary mapping collapses endpoint Y, detaching the stored circles
    from their mapped endpoints. Angle interpolation in that broken
    frame stalled or wildly swung the rotary axis on some arcs. The
    playhead must reconstruct each arc from the rotary axis travel:
    X stays between the endpoints and every arc rotates through its
    full angular travel without stalling.
    """
    player, arcs = _make_true4th_broken_frame_player()
    player.seek_to_first_layer()
    assert player._rotary_axis == Axis.A

    total = player.get_cumulative_time(player.ops.len() - 1)
    t = 0.0
    tick = 1.0 / 60.0
    samples = {}
    prev_a = None
    while t < total:
        t = min(t + tick, total)
        player.set_sim_time(t)
        player.sync_state_to_playhead()
        p, _frac = player.playback_progress()
        if p not in (3, 4, 5, 6):
            continue
        st = player.render_state()
        x = st.axes[Axis.X]
        assert 28.0 <= x <= 30.5, f"X off path: {x}"
        a = st.axes[Axis.A]
        if prev_a is not None:
            assert a >= prev_a - 1e-6, f"A not monotonic: {prev_a} -> {a}"
        prev_a = a
        samples.setdefault(p, []).append(a)
    # The rotation completes at the last arc's mapped target angle.
    last_op = 2 + len(arcs)
    extra = player.ops.extra_axes(last_op) or {}
    a_target = extra.get(Axis.A)
    assert a_target is not None
    assert prev_a == pytest.approx(a_target, abs=0.05)
    # No stalls: every arc sweeps at least 80% of its angular travel.
    a_before = {3: -128.0, 4: arcs[0][1], 5: arcs[1][1], 6: arcs[2][1]}
    for op, (_x_end, a_end, _i, _j) in enumerate(arcs, start=3):
        values = samples.get(op, [])
        assert values, f"arc op {op} never rendered"
        sweep = max(values) - min(values)
        expected = abs(a_end - a_before[op])
        assert sweep >= 0.8 * expected, (
            f"rotary axis stalled during op {op}: swept "
            f"{sweep:.3f} of {expected:.3f} deg"
        )


def test_flat_arc_still_interpolates_along_circle():
    """The consistency fallback must not affect healthy flat arcs."""
    ops = _make_arc_player()
    player = OpPlayer(ops, _make_machine(), Doc())
    player.set_playback_params(600.0, 3000.0, 0.0)
    player.seek(1)  # state at move_to (0,0); arc starts there
    # Arc duration is ~6.28 s; a quarter second in, frac ~ 0.04.
    player.set_sim_time(0.25)
    p, _frac = player.playback_progress()
    assert p == 2
    st = player.render_state()
    # The CW full circle of r=10 is centered at (10, 0): the head must
    # stay ON the circle, not somewhere inside it.
    dist = math.hypot(st.axes[Axis.X] - 10.0, st.axes[Axis.Y])
    assert dist == pytest.approx(10.0, abs=1e-3)
