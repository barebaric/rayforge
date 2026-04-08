from rayforge.core.ops import (
    Ops,
    MoveToCommand,
    LineToCommand,
    ArcToCommand,
    SetPowerCommand,
)


class TestSplitIntoSubpaths:
    def test_empty_ops(self):
        ops = Ops()
        assert ops.split_into_subpaths() == []

    def test_single_move(self):
        ops = Ops()
        ops.move_to(0, 0)
        result = ops.split_into_subpaths()
        assert len(result) == 1
        assert len(result[0]) == 1
        assert isinstance(result[0][0], MoveToCommand)

    def test_single_subpath(self):
        ops = Ops()
        ops.move_to(0, 0)
        ops.line_to(10, 0)
        ops.line_to(10, 10)
        result = ops.split_into_subpaths()
        assert len(result) == 1
        assert len(result[0]) == 3

    def test_two_subpaths(self):
        ops = Ops()
        ops.move_to(0, 0)
        ops.line_to(10, 0)
        ops.move_to(20, 20)
        ops.line_to(30, 30)
        result = ops.split_into_subpaths()
        assert len(result) == 2
        assert isinstance(result[0][0], MoveToCommand)
        assert isinstance(result[1][0], MoveToCommand)

    def test_state_commands_grouped_with_subpath(self):
        ops = Ops()
        ops.move_to(0, 0)
        ops.set_power(0.5)
        ops.line_to(10, 0)
        result = ops.split_into_subpaths()
        assert len(result) == 1
        assert len(result[0]) == 3
        assert isinstance(result[0][1], SetPowerCommand)

    def test_subpath_starting_with_lineto(self):
        ops = Ops()
        ops.line_to(5, 5)
        ops.line_to(10, 10)
        result = ops.split_into_subpaths()
        assert len(result) == 1
        assert isinstance(result[0][0], LineToCommand)

    def test_three_subpaths(self):
        ops = Ops()
        ops.move_to(0, 0)
        ops.line_to(1, 1)
        ops.move_to(2, 2)
        ops.line_to(3, 3)
        ops.move_to(4, 4)
        ops.line_to(5, 5)
        result = ops.split_into_subpaths()
        assert len(result) == 3

    def test_preserves_arc_commands(self):
        ops = Ops()
        ops.move_to(0, 0)
        ops.arc_to(10, 0, 5, 0, clockwise=True)
        result = ops.split_into_subpaths()
        assert len(result) == 1
        assert isinstance(result[0][1], ArcToCommand)

    def test_single_moveto_no_draw(self):
        ops = Ops()
        ops.move_to(5, 5)
        ops.move_to(10, 10)
        result = ops.split_into_subpaths()
        assert len(result) == 2
        assert len(result[0]) == 1
        assert len(result[1]) == 1
