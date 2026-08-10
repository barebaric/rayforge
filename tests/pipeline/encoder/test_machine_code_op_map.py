from rayforge.pipeline.encoder.base import MachineCodeOpMap


def test_op_for_line_valid_index():
    op_map = MachineCodeOpMap.from_lists([], [0, -1, 2])

    assert op_map.op_for_line(0) == 0
    assert op_map.op_for_line(2) == 2


def test_op_for_line_no_owning_op():
    op_map = MachineCodeOpMap.from_lists([], [0, -1, 2])

    assert op_map.op_for_line(1) is None


def test_op_for_line_out_of_range():
    op_map = MachineCodeOpMap.from_lists([], [0, 1])

    assert op_map.op_for_line(-1) is None
    assert op_map.op_for_line(2) is None


def test_op_for_line_empty_map():
    op_map = MachineCodeOpMap()

    assert op_map.op_for_line(0) is None


def test_span_for_op():
    op_map = MachineCodeOpMap.from_lists([(0, 1), (2, 3)], [])

    assert op_map.span_for_op(0) == (0, 1)
    assert op_map.span_for_op(1) == (2, 3)
    assert op_map.op_count == 2


def test_span_for_op_out_of_range():
    op_map = MachineCodeOpMap.from_lists([(0, 1)], [])

    try:
        op_map.span_for_op(5)
    except IndexError:
        pass
    else:
        raise AssertionError("span_for_op should raise IndexError")


def test_span_for_op_bytearray_roundtrip():
    import numpy as np

    spans = np.array([(0, 1), (2, 3)], dtype=np.int32)
    op_map = MachineCodeOpMap(op_to_machine_code=bytearray(spans.tobytes()))

    assert op_map.op_count == 2
    assert op_map.span_for_op(0) == (0, 1)
    assert op_map.span_for_op(1) == (2, 3)


def test_line_count():
    op_map = MachineCodeOpMap.from_lists([], [0, 1, 2, 3])

    assert op_map.line_count == 4
