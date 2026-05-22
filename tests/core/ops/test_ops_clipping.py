from typing import List, Tuple

import pytest
from raygeo.ops import Ops
from raygeo.ops.types import CommandCategory, CommandType


def make_square_region(
    x: float, y: float, w: float, h: float
) -> List[Tuple[float, float]]:
    return [(x, y), (x + w, y), (x + w, y + h), (x, y + h)]


class TestClipOpsToRegionsLines:
    def test_empty_ops(self):
        ops = Ops()
        regions = [make_square_region(0, 0, 10, 10)]
        ops.clip_ops_to_regions(regions)
        assert ops.len() == 0

    def test_empty_regions(self):
        ops = Ops()
        ops.move_to(0, 0)
        ops.line_to(10, 0)
        ops.clip_ops_to_regions([])
        assert ops.len() == 2

    def test_small_regions_skipped(self):
        ops = Ops()
        ops.move_to(0, 0)
        ops.line_to(10, 0)
        ops.clip_ops_to_regions([[(0.0, 0.0), (1.0, 1.0)]])
        assert ops.len() == 2

    def test_line_fully_inside(self):
        ops = Ops()
        ops.move_to(2, 5)
        ops.line_to(8, 5)
        regions = [make_square_region(0, 0, 10, 10)]
        ops.clip_ops_to_regions(regions)
        segs = list(ops.segments())
        assert len(segs) == 1

    def test_line_fully_outside(self):
        ops = Ops()
        ops.move_to(20, 20)
        ops.line_to(30, 30)
        regions = [make_square_region(0, 0, 10, 10)]
        ops.clip_ops_to_regions(regions)
        assert len(list(ops.segments())) == 0

    def test_line_partially_clipped(self):
        ops = Ops()
        ops.move_to(0, 5)
        ops.line_to(20, 5)
        regions = [make_square_region(5, 0, 10, 10)]
        ops.clip_ops_to_regions(regions)
        segs = list(ops.segment_indices())
        assert len(segs) == 1
        seg = segs[0]
        assert ops.endpoint(seg[0]) is not None
        assert ops.endpoint(seg[-1]) is not None
        assert ops.endpoint(seg[0])[0] >= 5.0
        assert ops.endpoint(seg[-1])[0] <= 15.0

    def test_multiple_regions(self):
        ops = Ops()
        ops.move_to(0, 5)
        ops.line_to(20, 5)
        regions = [
            make_square_region(0, 0, 5, 10),
            make_square_region(15, 0, 5, 10),
        ]
        ops.clip_ops_to_regions(regions)
        segs = list(ops.segments())
        assert len(segs) == 2


class TestClipOpsToRegionsArcs:
    def test_arc_fully_inside_preserved(self):
        ops = Ops()
        ops.move_to(4, 5)
        ops.arc_to(6, 5, 1, 0, clockwise=True)
        regions = [make_square_region(0, 0, 10, 10)]
        ops.clip_ops_to_regions(regions)
        arc_indices = ops.indices_of(CommandType.ARC_TO)
        assert len(arc_indices) == 1
        assert ops.endpoint(arc_indices[0]) == pytest.approx(
            (6, 5, 0), abs=1e-6
        )

    def test_arc_fully_outside_removed(self):
        ops = Ops()
        ops.move_to(50, 50)
        ops.arc_to(52, 50, 1, 0, clockwise=True)
        regions = [make_square_region(0, 0, 10, 10)]
        ops.clip_ops_to_regions(regions)
        assert len(list(ops.segments())) == 0

    def test_arc_partially_outside_refitted(self):
        ops = Ops()
        ops.move_to(1, 5)
        ops.arc_to(9, 5, 4, 0, clockwise=True)
        regions = [make_square_region(3, 0, 4, 10)]
        ops.clip_ops_to_regions(regions)
        arc_indices = ops.indices_of(CommandType.ARC_TO)
        assert len(arc_indices) >= 1
        for seg_indices in ops.segment_indices():
            for i in seg_indices:
                if ops.category(i) == CommandCategory.MOVING:
                    assert 3.0 <= ops.endpoint(i)[0] <= 7.0

    def test_mixed_lines_and_arcs_inside(self):
        ops = Ops()
        ops.move_to(2, 5)
        ops.line_to(4, 5)
        ops.arc_to(6, 5, 1, 0, clockwise=True)
        ops.line_to(8, 5)
        regions = [make_square_region(0, 0, 10, 10)]
        ops.clip_ops_to_regions(regions)
        arc_indices = ops.indices_of(CommandType.ARC_TO)
        assert len(arc_indices) == 1
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
        ops.clip_ops_to_regions(regions)
        arc_indices = ops.indices_of(CommandType.ARC_TO)
        assert len(arc_indices) == 4

    def test_arc_state_preserved_after_refit(self):
        ops = Ops()
        ops.move_to(1, 5)
        ops.set_power(0.8)
        ops.arc_to(9, 5, 4, 0, clockwise=True)
        ops.preload_state()
        regions = [make_square_region(3, 0, 4, 10)]
        ops.clip_ops_to_regions(regions)
        arc_indices = ops.indices_of(CommandType.ARC_TO)
        for idx in arc_indices:
            state = ops.inspect(idx).state
            assert state is not None
            assert abs(state.power - 0.8) < 1e-6


class TestClipOpsToRegionsLeadingCommands:
    def test_state_commands_before_first_move_preserved(self):
        ops = Ops()
        ops.set_power(0.5)
        ops.move_to(2, 5)
        ops.line_to(8, 5)
        regions = [make_square_region(0, 0, 10, 10)]
        ops.clip_ops_to_regions(regions)
        state_indices = ops.indices_of(CommandType.SET_POWER)
        assert len(state_indices) == 1


class TestClipOpsToRegionsBezier:
    def test_bezier_fully_inside_preserved(self):
        ops = Ops()
        ops.move_to(3, 5)
        ops.bezier_to((4, 7, 0), (6, 7, 0), (7, 5, 0))
        regions = [make_square_region(0, 0, 10, 10)]
        ops.clip_ops_to_regions(regions)
        bezier_indices = ops.indices_of(CommandType.BEZIER_TO)
        assert len(bezier_indices) == 1
        assert ops.endpoint(bezier_indices[0]) == pytest.approx(
            (7, 5, 0), abs=1e-6
        )

    def test_bezier_fully_outside_removed(self):
        ops = Ops()
        ops.move_to(50, 50)
        ops.bezier_to((51, 53, 0), (53, 53, 0), (54, 50, 0))
        regions = [make_square_region(0, 0, 10, 10)]
        ops.clip_ops_to_regions(regions)
        assert len(list(ops.segments())) == 0

    def test_bezier_partially_outside_refitted(self):
        ops = Ops()
        ops.move_to(1, 5)
        ops.bezier_to((3, 8, 0), (7, 8, 0), (9, 5, 0))
        regions = [make_square_region(3, 0, 4, 10)]
        ops.clip_ops_to_regions(regions)
        segs = list(ops.segment_indices())
        assert len(segs) >= 1
        for seg_indices in segs:
            for i in seg_indices:
                if ops.category(i) == CommandCategory.MOVING:
                    assert 3.0 <= ops.endpoint(i)[0] <= 7.0

    def test_bezier_state_preserved_after_refit(self):
        ops = Ops()
        ops.move_to(1, 5)
        ops.set_power(0.8)
        ops.bezier_to((3, 8, 0), (7, 8, 0), (9, 5, 0))
        ops.preload_state()
        regions = [make_square_region(3, 0, 4, 10)]
        ops.clip_ops_to_regions(regions)
        for i in ops.indices_of(CommandType.ARC_TO) + ops.indices_of(
            CommandType.BEZIER_TO
        ):
            state = ops.inspect(i).state
            assert state is not None
            assert abs(state.power - 0.8) < 1e-6

    def test_mixed_lines_and_beziers_inside(self):
        ops = Ops()
        ops.move_to(2, 5)
        ops.line_to(3, 5)
        ops.bezier_to((4, 7, 0), (6, 7, 0), (7, 5, 0))
        ops.line_to(8, 5)
        regions = [make_square_region(0, 0, 10, 10)]
        ops.clip_ops_to_regions(regions)
        bezier_indices = ops.indices_of(CommandType.BEZIER_TO)
        assert len(bezier_indices) == 1
        assert len(list(ops.segments())) == 1
