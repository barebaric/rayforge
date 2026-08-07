import math
from unittest.mock import patch

import pytest
from raygeo.ops import Ops
from raygeo.ops.axis import Axis

from rayforge.context import RayforgeContext
from rayforge.core.doc import Doc
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


def test_seek_into_layer_configures_machine_for_that_layer():
    machine = _make_machine()
    doc = _make_rotary_doc("test", machine)

    ops = Ops()
    ops.move_to(0, 0, 0)
    ops.layer_start("test")
    ops.line_to(10, 20, 0)

    player = OpPlayer(ops, machine, doc)
    with patch.object(machine, "configure_for_layer") as mock_cfg:
        player.seek(ops.len() - 1)
    mock_cfg.assert_called_with(doc.active_layer)


def test_seek_preamble_configures_machine_for_first_layer():
    machine = _make_machine()
    doc = _make_rotary_doc("test", machine)

    ops = Ops()
    ops.move_to(5, 5, 0)
    ops.set_power(0.5)
    ops.layer_start("test")
    ops.line_to(10, 20, 0)

    player = OpPlayer(ops, machine, doc)
    with patch.object(machine, "configure_for_layer") as mock_cfg:
        player.seek(0)
    mock_cfg.assert_called_with(doc.layers[0])


def test_seek_flat_layer_configures_machine_flat():
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
    assert not machine.assembly.has_rotary


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
    for target, state, source, rotary in snapshots:
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
    for _, state, _, _ in snapshots:
        assert len(state.reached_textures) == 0
