import pytest
from rayforge.simulator.op_player import OpPlayer
from rayforge.core.ops import Ops
from rayforge.core.ops.commands import ScanLinePowerCommand
from rayforge.core.ops.axis import Axis
from rayforge.core.doc import Doc
from rayforge.machine.models.machine import Machine
from rayforge.machine.models.rotary_module import RotaryModule, RotaryMode
from rayforge.context import RayforgeContext


def _make_machine():
    ctx = RayforgeContext()
    return Machine(ctx)


def _make_ops():
    ops = Ops()
    ops.set_power(0.5)
    ops.set_cut_speed(800)
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


def test_random_access_matches_sequential():
    machine = _make_machine()
    doc = Doc()
    ops = Ops()
    ops.set_power(0.3)
    ops.move_to(5.0, 5.0, 0.0)
    ops.set_power(0.7)
    ops.line_to(15.0, 25.0, 3.0)
    ops.set_cut_speed(1200)
    ops.line_to(50.0, 60.0, 0.0)

    sequential = OpPlayer(ops, machine, doc)
    sequential.seek(len(list(ops)) - 1)

    player = OpPlayer(ops, machine, doc)
    player.seek(1)
    player.seek(len(list(ops)) - 1)

    assert player.state.axes == sequential.state.axes
    assert player.state.power == sequential.state.power
    assert player.state.cut_speed == sequential.state.cut_speed


def test_scanline_tracked():
    ops = Ops()
    ops.move_to(0.0, 0.0, 0.0)
    ops.add(ScanLinePowerCommand((10.0, 0.0, 0.0), bytearray([100, 200])))
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


def test_passthrough_mode_no_rotary_mapping():
    from rayforge.core.ops.commands import LayerStartCommand

    machine = _make_machine()
    rm = RotaryModule()
    rm.set_mode(RotaryMode.PASSTHROUGH)
    rm.set_source_axis(Axis.Y)
    machine.add_rotary_module(rm)

    ops = Ops()
    ops.move_to(0, 0, 0)
    ops.add(LayerStartCommand(layer_uid="test"))
    ops.line_to(10, 20, 0)

    doc = Doc()
    doc.active_layer.uid = "test"
    doc.active_layer.set_rotary_enabled(True)
    doc.active_layer.set_rotary_module_uid(rm.uid)

    player = OpPlayer(ops, machine, doc)
    player.seek(len(list(ops)) - 1)

    assert player.state.axes.get(Axis.A, 0.0) == pytest.approx(0.0)


def test_true_4th_axis_copies_to_rotary():
    from rayforge.core.ops.commands import LayerStartCommand

    machine = _make_machine()
    rm = RotaryModule()
    rm.set_mode(RotaryMode.TRUE_4TH_AXIS)
    rm.set_axis(Axis.A)
    rm.set_source_axis(Axis.Y)
    machine.add_rotary_module(rm)

    ops = Ops()
    ops.move_to(0, 0, 0)
    ops.add(LayerStartCommand(layer_uid="test"))
    ops.line_to(10, 20, 0)

    doc = Doc()
    doc.active_layer.uid = "test"
    doc.active_layer.set_rotary_enabled(True)
    doc.active_layer.set_rotary_module_uid(rm.uid)

    player = OpPlayer(ops, machine, doc)
    player.seek(len(list(ops)) - 1)

    assert player.state.axes[Axis.A] == pytest.approx(20.0)
