import pytest
from rayforge.simulator.op_player import OpPlayer
from rayforge.core.ops import Ops
from rayforge.core.ops.commands import ScanLinePowerCommand
from rayforge.core.ops.axis import Axis


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
    player = OpPlayer(ops)
    player.seek(0)
    assert player.current_index == 0
    assert player.state.power == 0.5


def test_advance_from_start():
    ops = _make_ops()
    player = OpPlayer(ops)
    player.advance_to(2)
    player.advance_to(3)
    assert player.current_index == 3
    assert player.state.axes[Axis.X] == 10.0
    assert player.state.axes[Axis.Y] == 0.0


def test_seek_forward_then_backward_replays():
    ops = _make_ops()
    player = OpPlayer(ops)
    player.seek(6)
    assert player.state.axes[Axis.X] == 0.0
    assert player.state.power == 1.0

    player.seek(5)
    assert player.state.axes[Axis.X] == 10.0
    assert player.state.axes[Axis.Y] == 10.0


def test_seek_then_advance():
    ops = _make_ops()
    player = OpPlayer(ops)
    player.seek(3)
    player.advance_to(6)
    assert player.current_index == 6
    assert player.state.axes[Axis.X] == 0.0


def test_advance_backwards_raises():
    ops = _make_ops()
    player = OpPlayer(ops)
    player.advance_to(3)
    with pytest.raises(ValueError, match="Cannot advance backwards"):
        player.advance_to(2)


def test_seek_out_of_range_raises():
    ops = _make_ops()
    player = OpPlayer(ops)
    with pytest.raises(IndexError):
        player.seek(999)


def test_advance_out_of_range_raises():
    ops = _make_ops()
    player = OpPlayer(ops)
    with pytest.raises(IndexError):
        player.advance_to(999)


def test_empty_ops_raises():
    with pytest.raises(ValueError):
        OpPlayer(Ops())


def test_none_ops_raises():
    with pytest.raises(ValueError):
        OpPlayer(None)  # type: ignore[arg-type]


def test_seek_last_movement():
    ops = _make_ops()
    player = OpPlayer(ops)
    last = player.seek_last_movement()
    assert last == 6
    assert player.state.axes[Axis.X] == 0.0
    assert player.state.axes[Axis.Y] == 0.0


def test_random_access_matches_sequential():
    ops = Ops()
    ops.set_power(0.3)
    ops.move_to(5.0, 5.0, 0.0)
    ops.set_power(0.7)
    ops.line_to(15.0, 25.0, 3.0)
    ops.set_cut_speed(1200)
    ops.line_to(50.0, 60.0, 0.0)

    sequential = OpPlayer(ops)
    sequential.seek(len(list(ops)) - 1)

    player = OpPlayer(ops)
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

    player = OpPlayer(ops)
    player.seek(2)

    assert 1 in player.state.reached_textures
    assert 0 not in player.state.reached_textures
    assert 2 not in player.state.reached_textures
