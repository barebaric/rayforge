from typing import List, Tuple

import pytest

from rayforge.core.ops import Ops
from rayforge.core.ops.commands import (
    ArcToCommand,
    BezierToCommand,
    State,
    SetPowerCommand,
)
from rayforge.core.ops.clipping import clip_ops_to_regions


def make_square_region(
    x: float, y: float, w: float, h: float
) -> List[Tuple[float, float]]:
    return [(x, y), (x + w, y), (x + w, y + h), (x, y + h)]


class TestClipOpsToRegionsLines:
    def test_empty_ops(self):
        ops = Ops()
        regions = [make_square_region(0, 0, 10, 10)]
        result = clip_ops_to_regions(ops, regions)
        assert len(result.commands) == 0

    def test_empty_regions(self):
        ops = Ops()
        ops.move_to(0, 0)
        ops.line_to(10, 0)
        result = clip_ops_to_regions(ops, [])
        assert result is ops

    def test_small_regions_skipped(self):
        ops = Ops()
        ops.move_to(0, 0)
        ops.line_to(10, 0)
        result = clip_ops_to_regions(ops, [[(0.0, 0.0), (1.0, 1.0)]])
        assert result is ops

    def test_line_fully_inside(self):
        ops = Ops()
        ops.move_to(2, 5)
        ops.line_to(8, 5)
        regions = [make_square_region(0, 0, 10, 10)]
        clip_ops_to_regions(ops, regions)
        segs = list(ops.segments())
        assert len(segs) == 1

    def test_line_fully_outside(self):
        ops = Ops()
        ops.move_to(20, 20)
        ops.line_to(30, 30)
        regions = [make_square_region(0, 0, 10, 10)]
        clip_ops_to_regions(ops, regions)
        assert len(list(ops.segments())) == 0

    def test_line_partially_clipped(self):
        ops = Ops()
        ops.move_to(0, 5)
        ops.line_to(20, 5)
        regions = [make_square_region(5, 0, 10, 10)]
        clip_ops_to_regions(ops, regions)
        segs = list(ops.segments())
        assert len(segs) == 1
        seg = segs[0]
        assert seg[0].end is not None
        assert seg[-1].end is not None
        assert seg[0].end[0] >= 5.0
        assert seg[-1].end[0] <= 15.0

    def test_multiple_regions(self):
        ops = Ops()
        ops.move_to(0, 5)
        ops.line_to(20, 5)
        regions = [
            make_square_region(0, 0, 5, 10),
            make_square_region(15, 0, 5, 10),
        ]
        clip_ops_to_regions(ops, regions)
        segs = list(ops.segments())
        assert len(segs) == 2


class TestClipOpsToRegionsArcs:
    def test_arc_fully_inside_preserved(self):
        ops = Ops()
        ops.move_to(4, 5)
        ops.arc_to(6, 5, 1, 0, clockwise=True)
        regions = [make_square_region(0, 0, 10, 10)]
        clip_ops_to_regions(ops, regions)
        arcs = [c for c in ops.commands if isinstance(c, ArcToCommand)]
        assert len(arcs) == 1
        assert arcs[0].end == pytest.approx((6, 5, 0), abs=1e-6)

    def test_arc_fully_outside_removed(self):
        ops = Ops()
        ops.move_to(50, 50)
        ops.arc_to(52, 50, 1, 0, clockwise=True)
        regions = [make_square_region(0, 0, 10, 10)]
        clip_ops_to_regions(ops, regions)
        assert len(list(ops.segments())) == 0

    def test_arc_partially_outside_refitted(self):
        ops = Ops()
        ops.move_to(1, 5)
        ops.arc_to(9, 5, 4, 0, clockwise=True)
        regions = [make_square_region(3, 0, 4, 10)]
        clip_ops_to_regions(ops, regions)
        arcs = [c for c in ops.commands if isinstance(c, ArcToCommand)]
        assert len(arcs) >= 1
        segs = list(ops.segments())
        for seg in segs:
            for cmd in seg:
                if hasattr(cmd, "end") and cmd.end is not None:
                    assert 3.0 <= cmd.end[0] <= 7.0

    def test_mixed_lines_and_arcs_inside(self):
        ops = Ops()
        ops.move_to(2, 5)
        ops.line_to(4, 5)
        ops.arc_to(6, 5, 1, 0, clockwise=True)
        ops.line_to(8, 5)
        regions = [make_square_region(0, 0, 10, 10)]
        clip_ops_to_regions(ops, regions)
        arcs = [c for c in ops.commands if isinstance(c, ArcToCommand)]
        assert len(arcs) == 1
        assert len(list(ops.segments())) == 1

    def test_rounded_rect_all_corners_preserved(self):
        r = 0.5
        x, y = 2, 3
        w, h = 6, 4
        ops = Ops()
        ops.move_to(x + r, y)
        ops.line_to(x + w - r, y)
        ops.arc_to(x + w, y + r, 0, r, clockwise=True)
        ops.line_to(x + w, y + h - r)
        ops.arc_to(x + w - r, y + h, -r, 0, clockwise=True)
        ops.line_to(x + r, y + h)
        ops.arc_to(x, y + h - r, 0, -r, clockwise=True)
        ops.line_to(x, y + r)
        ops.arc_to(x + r, y, r, 0, clockwise=True)
        regions = [make_square_region(0, 0, 10, 10)]
        clip_ops_to_regions(ops, regions)
        arcs = [c for c in ops.commands if isinstance(c, ArcToCommand)]
        assert len(arcs) == 4

    def test_arc_state_preserved_after_refit(self):
        ops = Ops()
        ops.move_to(1, 5)
        arc = ArcToCommand((9, 5, 0), (4, 0), clockwise=True)
        arc.state = State(power=0.8)
        ops.add(arc)
        regions = [make_square_region(3, 0, 4, 10)]
        clip_ops_to_regions(ops, regions)
        arcs = [c for c in ops.commands if isinstance(c, ArcToCommand)]
        for a in arcs:
            assert a.state is not None
            assert abs(a.state.power - 0.8) < 1e-6


class TestClipOpsToRegionsLeadingCommands:
    def test_state_commands_before_first_move_preserved(self):
        ops = Ops()
        ops.add(SetPowerCommand(0.5))
        ops.move_to(2, 5)
        ops.line_to(8, 5)
        regions = [make_square_region(0, 0, 10, 10)]
        clip_ops_to_regions(ops, regions)
        state_cmds = [
            c for c in ops.commands if isinstance(c, SetPowerCommand)
        ]
        assert len(state_cmds) == 1


class TestClipOpsToRegionsBezier:
    def test_bezier_fully_inside_preserved(self):
        ops = Ops()
        ops.move_to(3, 5)
        ops.bezier_to((4, 7, 0), (6, 7, 0), (7, 5, 0))
        regions = [make_square_region(0, 0, 10, 10)]
        clip_ops_to_regions(ops, regions)
        beziers = [c for c in ops.commands if isinstance(c, BezierToCommand)]
        assert len(beziers) == 1
        assert beziers[0].end == pytest.approx((7, 5, 0), abs=1e-6)

    def test_bezier_fully_outside_removed(self):
        ops = Ops()
        ops.move_to(50, 50)
        ops.bezier_to((51, 53, 0), (53, 53, 0), (54, 50, 0))
        regions = [make_square_region(0, 0, 10, 10)]
        clip_ops_to_regions(ops, regions)
        assert len(list(ops.segments())) == 0

    def test_bezier_partially_outside_refitted(self):
        ops = Ops()
        ops.move_to(1, 5)
        ops.bezier_to((3, 8, 0), (7, 8, 0), (9, 5, 0))
        regions = [make_square_region(3, 0, 4, 10)]
        clip_ops_to_regions(ops, regions)
        segs = list(ops.segments())
        assert len(segs) >= 1
        for seg in segs:
            for cmd in seg:
                if hasattr(cmd, "end") and cmd.end is not None:
                    assert 3.0 <= cmd.end[0] <= 7.0

    def test_bezier_state_preserved_after_refit(self):
        ops = Ops()
        ops.move_to(1, 5)
        bezier = BezierToCommand((9, 5, 0), (3, 8, 0), (7, 8, 0))
        bezier.state = State(power=0.8)
        ops.add(bezier)
        regions = [make_square_region(3, 0, 4, 10)]
        clip_ops_to_regions(ops, regions)
        for cmd in ops.commands:
            if isinstance(cmd, (ArcToCommand, BezierToCommand)):
                assert cmd.state is not None
                assert abs(cmd.state.power - 0.8) < 1e-6

    def test_mixed_lines_and_beziers_inside(self):
        ops = Ops()
        ops.move_to(2, 5)
        ops.line_to(3, 5)
        ops.bezier_to((4, 7, 0), (6, 7, 0), (7, 5, 0))
        ops.line_to(8, 5)
        regions = [make_square_region(0, 0, 10, 10)]
        clip_ops_to_regions(ops, regions)
        beziers = [c for c in ops.commands if isinstance(c, BezierToCommand)]
        assert len(beziers) == 1
        assert len(list(ops.segments())) == 1
