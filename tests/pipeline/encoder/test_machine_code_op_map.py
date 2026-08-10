from rayforge.pipeline.encoder.base import MachineCodeOpMap


def test_op_for_line_valid_index():
    op_map = MachineCodeOpMap(machine_code_to_op=[0, -1, 2])

    assert op_map.op_for_line(0) == 0
    assert op_map.op_for_line(2) == 2


def test_op_for_line_no_owning_op():
    op_map = MachineCodeOpMap(machine_code_to_op=[0, -1, 2])

    assert op_map.op_for_line(1) is None


def test_op_for_line_out_of_range():
    op_map = MachineCodeOpMap(machine_code_to_op=[0, 1])

    assert op_map.op_for_line(-1) is None
    assert op_map.op_for_line(2) is None


def test_op_for_line_empty_map():
    op_map = MachineCodeOpMap()

    assert op_map.op_for_line(0) is None
