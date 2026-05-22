import pytest
from rayforge.core.ops import Ops, CommandType


def test_flip_segment_empty_or_short():
    ops = Ops()
    flipped = ops.flip_ops()
    assert flipped.len() == 0

    ops = Ops()
    ops.move_to(1, 2, 3)
    flipped = ops.flip_ops()
    assert flipped.len() == 1
    assert flipped.command_type(0) == CommandType.MOVE_TO


def test_flip_segment_lines_only():
    ops = Ops()
    ops.move_to(0, 0, 0)
    ops.line_to(10, 0, 1)
    ops.line_to(10, 10, 2)

    flipped = ops.flip_ops()
    assert flipped.len() == 3
    assert flipped.endpoint(0) == (10, 10, 2)
    assert flipped.endpoint(1) == (10, 0, 1)
    assert flipped.endpoint(2) == (0, 0, 0)


def test_flip_segment_with_arc():
    ops = Ops()
    ops.move_to(0, 10, 0)
    ops.line_to(10, 0, 0)
    ops.arc_to(0, 0, -5, 0, False)

    flipped = ops.flip_ops()
    assert flipped.len() == 3
    assert flipped.endpoint(0) == (0, 0, 0)
    assert flipped.endpoint(1) == (10, 0, 0)
    assert flipped.endpoint(2) == (0, 10, 0)

    assert flipped.command_type(1) == CommandType.ARC_TO
    arc_i, arc_j, arc_cw = flipped.arc_params(1)
    assert arc_cw is True
    assert (arc_i, arc_j) == pytest.approx((5, 0))


def test_flip_segment_with_scanline():
    ops = Ops()
    ops.move_to(0, 0, 0)
    ops.scan_to(10, 10, 10, power_values=bytearray([10, 20, 30]))

    flipped = ops.flip_ops()
    assert flipped.len() == 2
    assert flipped.command_type(0) == CommandType.MOVE_TO
    assert flipped.command_type(1) == CommandType.SCAN_LINE
    assert flipped.endpoint(0) == (10, 10, 10)
    assert flipped.endpoint(1) == (0, 0, 0)

    pv = flipped.scanline_data(1)
    assert bytes(pv) == bytearray([30, 20, 10])
