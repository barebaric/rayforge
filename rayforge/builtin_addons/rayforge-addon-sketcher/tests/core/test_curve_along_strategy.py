import math

import pytest
from sketcher.core.arrays import (
    CurveAlongArrayStrategy,
    InstancePlacement,
    PlacementKind,
    sample_path,
)
from sketcher.core.arrays.curve_along import (
    _cumulative_lengths,
    _point_at_arclength,
)
from sketcher.core.sketch import Sketch

# ----------------------------------------------------------------------
# Factory / strategy basics
# ----------------------------------------------------------------------


def test_placements_without_registry_are_empty():
    strat = CurveAlongArrayStrategy(count=4)
    assert strat.member_placements((0.0, 0.0)) == []


def test_no_master_geometry():
    strat = CurveAlongArrayStrategy(count=4)
    assert strat.create_master_geometry(None, None) == ([], [], [])


# ----------------------------------------------------------------------
# Path sampling
# ----------------------------------------------------------------------


def _line_path_sketch(x1=0.0, y1=0.0, x2=10.0, y2=0.0):
    sketch = Sketch()
    p0 = sketch.registry.add_point(x1, y1)
    p1 = sketch.registry.add_point(x2, y2)
    line = sketch.registry.add_line(p0, p1)
    return sketch, line


def test_sample_line_path_endpoints():
    sketch, line = _line_path_sketch(0.0, 0.0, 10.0, 0.0)
    samples = sample_path(sketch.registry, line, count=3)
    assert len(samples) == 3
    # Endpoints sit on the path ends.
    assert samples[0][0] == pytest.approx((0.0, 0.0))
    assert samples[-1][0] == pytest.approx((10.0, 0.0))
    # Tangent points along +x.
    for _pt, angle in samples:
        assert angle == pytest.approx(0.0)


def test_sample_line_path_midpoint():
    sketch, _ = _line_path_sketch(0.0, 0.0, 10.0, 0.0)
    line_id = sketch.registry.entities[0].id
    samples = sample_path(sketch.registry, line_id, count=3)
    assert samples[1][0] == pytest.approx((5.0, 0.0))


def test_sample_arc_path_tangent_rotates():
    sketch = Sketch()
    c = sketch.registry.add_point(0.0, 0.0)
    s = sketch.registry.add_point(10.0, 0.0)
    e = sketch.registry.add_point(0.0, 10.0)
    arc = sketch.registry.add_arc(s, e, c, cw=False)
    samples = sample_path(sketch.registry, arc, count=4)
    assert len(samples) == 4
    # Start tangent at (10,0) on a CCW arc about origin points roughly
    # in +y (the discrete polyline chord is a small approximation).
    start_angle = samples[0][1]
    assert math.sin(start_angle) > 0.99
    assert abs(start_angle - math.pi / 2) < math.radians(5)


def test_sample_offset_clamps_to_path():
    sketch, _ = _line_path_sketch(0.0, 0.0, 10.0, 0.0)
    line_id = sketch.registry.entities[0].id
    # Offset larger than the path clamps; the first sample moves
    # forward but the last still lands at the end.
    samples = sample_path(
        sketch.registry, line_id, count=3, offset_to_start=100.0
    )
    assert samples[0][0] == pytest.approx(samples[1][0])
    assert samples[-1][0] == pytest.approx((10.0, 0.0))


def test_sample_unknown_entity_returns_empty():
    sketch, _ = _line_path_sketch()
    assert sample_path(sketch.registry, 9999, count=4) == []


# ----------------------------------------------------------------------
# Arc length helpers
# ----------------------------------------------------------------------


def test_cumulative_lengths():
    cum = _cumulative_lengths([(0, 0), (3, 0), (3, 4)])
    assert cum == pytest.approx([0.0, 3.0, 7.0])


def test_point_at_arclength_interpolates():
    poly = [(0.0, 0.0), (10.0, 0.0)]
    cum = _cumulative_lengths(poly)
    point, tan = _point_at_arclength(poly, cum, 4.0)
    assert point == pytest.approx((4.0, 0.0))
    assert tan == pytest.approx((1.0, 0.0))


def test_point_at_arclength_clamps_to_end():
    poly = [(0.0, 0.0), (10.0, 0.0)]
    cum = _cumulative_lengths(poly)
    point, _tan = _point_at_arclength(poly, cum, 100.0)
    assert point == pytest.approx((10.0, 0.0))


# ----------------------------------------------------------------------
# Placements along a path
# ----------------------------------------------------------------------


def test_member_placements_along_line_path():
    sketch, line = _line_path_sketch(0.0, 0.0, 30.0, 0.0)
    strat = CurveAlongArrayStrategy(count=4, path_entity_id=line)
    template = strat.template_placement((0.0, 0.0), sketch.registry)
    assert template is not None
    assert template.target_center == pytest.approx((0.0, 0.0))
    assert template.angle == pytest.approx(0.0)

    placements = strat.member_placements((0.0, 0.0), sketch.registry)
    # count=4 members: template at the start + 3 copies.
    assert len(placements) == 3
    # Each copy carries the template from sample 0 onto sample j.
    assert all(p.center == pytest.approx((0.0, 0.0)) for p in placements)
    assert placements[-1].target_center == pytest.approx((30.0, 0.0))
    # All placements point along +x (tangent of horizontal line).
    assert all(p.angle == 0.0 for p in placements)
    assert all(p.kind == PlacementKind.CURVE_ALIGNED for p in placements)


def test_spacing_drives_count_along_line_path():
    """When spacing > 0, count is derived from the usable path length."""
    sketch, line = _line_path_sketch(0.0, 0.0, 30.0, 0.0)
    # 30-unit path, spacing 10 -> count 4 (30/10 + 1).
    strat = CurveAlongArrayStrategy(
        path_entity_id=line, spacing=10.0, count=99
    )
    placements = strat.member_placements((0.0, 0.0), sketch.registry)
    assert len(placements) == 3  # template at start + 3 copies


def test_spacing_respects_start_offset():
    sketch, line = _line_path_sketch(0.0, 0.0, 30.0, 0.0)
    # Path 30, offset 10 -> usable 20. Spacing 5 -> count 5 (20/5 + 1).
    strat = CurveAlongArrayStrategy(
        path_entity_id=line, spacing=5.0, offset_to_start=10.0
    )
    template = strat.template_placement((0.0, 0.0), sketch.registry)
    assert template is not None
    assert template.target_center == pytest.approx((10.0, 0.0))
    placements = strat.member_placements((0.0, 0.0), sketch.registry)
    assert len(placements) == 4  # template at the offset start + 4 copies
    assert placements[-1].target_center == pytest.approx((30.0, 0.0))


def test_placements_without_align_have_zero_angle():
    sketch, line = _line_path_sketch(0.0, 0.0, 30.0, 0.0)
    strat = CurveAlongArrayStrategy(
        count=4, path_entity_id=line, align_to_tangent=False
    )
    placements = strat.member_placements((0.0, 0.0), sketch.registry)
    assert all(p.angle == 0.0 for p in placements)


def test_placement_transform_point_translates_and_rotates():
    """A point on the template maps rigidly to the target frame."""
    placement = InstancePlacement(
        kind=PlacementKind.CURVE_ALIGNED,
        angle=math.pi / 2,
        center=(0.0, 0.0),
        target_center=(10.0, 5.0),
    )
    # The template's "forward" point (1, 0) relative to center (0,0)
    # rotates
    # 90deg to (0,1) then translates to (10, 6).
    x, y = placement.transform_point(1.0, 0.0)
    assert x == pytest.approx(10.0)
    assert y == pytest.approx(6.0)


def test_placement_transform_offset_rotates_only():
    placement = InstancePlacement(
        kind=PlacementKind.CURVE_ALIGNED,
        angle=math.pi / 2,
        center=(0.0, 0.0),
        target_center=(10.0, 5.0),
    )
    # Offsets are rotation-only (no translation).
    dx, dy = placement.transform_offset(1.0, 0.0)
    assert dx == pytest.approx(0.0)
    assert dy == pytest.approx(1.0)
