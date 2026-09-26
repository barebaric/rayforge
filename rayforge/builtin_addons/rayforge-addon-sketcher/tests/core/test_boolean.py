"""Tests for boolean region extraction and boolean operations."""

import math

import pytest
from raygeo.geo.shape.polygon import get_polygon_signed_area
from sketcher.core import Sketch
from sketcher.core.boolean import (
    BooleanOp,
    apply_boolean,
    build_boolean_regions,
)
from sketcher.core.entities import Circle, PolygonEntity


def _square_sketch(sketch, x=0.0, y=0.0, size=20.0):
    ids = [
        sketch.add_point(x, y),
        sketch.add_point(x + size, y),
        sketch.add_point(x + size, y + size),
        sketch.add_point(x, y + size),
    ]
    entity_ids = [
        sketch.add_line(ids[0], ids[1]),
        sketch.add_line(ids[1], ids[2]),
        sketch.add_line(ids[2], ids[3]),
        sketch.add_line(ids[3], ids[0]),
    ]
    return entity_ids


def _circle_sketch(sketch, cx=0.0, cy=0.0, r=10.0):
    center = sketch.add_point(cx, cy)
    radius_pt = sketch.add_point(cx + r, cy)
    return sketch.add_circle(center, radius_pt)


def _lens_area(d, r):
    """Area of the intersection of two equal circles of radius r whose
    centers are d apart."""
    return 2 * r * r * math.acos(d / (2 * r)) - (d / 2) * math.sqrt(
        4 * r * r - d * d
    )


def _region_outers(regions):
    return [region.rings[0] for region in regions]


def _ring_bbox(ring):
    xs = [p[0] for p in ring]
    ys = [p[1] for p in ring]
    return min(xs), min(ys), max(xs), max(ys)


def _regions(sketch, entity_ids):
    regions = build_boolean_regions(sketch, entity_ids)
    assert regions is not None
    return regions


def test_regions_from_two_circles():
    sketch = Sketch()
    c1 = _circle_sketch(sketch, 0.0)
    c2 = _circle_sketch(sketch, 8.0)
    regions = _regions(sketch, [c1, c2])
    assert regions is not None and len(regions) == 2
    for region in regions:
        assert len(region.rings) == 1
        assert get_polygon_signed_area(region.rings[0]) > 0


def test_regions_from_closed_line_chain_and_circle():
    sketch = Sketch()
    lines = _square_sketch(sketch)
    circle = _circle_sketch(sketch, 10.0)
    regions = _regions(sketch, lines + [circle])
    assert regions is not None and len(regions) == 2


def test_regions_include_polygon_holes():
    sketch = Sketch()
    center = sketch.add_point(0.0, 0.0)
    handle = sketch.add_point(10.0, 0.0)
    _center_pt, _handle_pt, polygon = PolygonEntity.build(
        9000,
        center,
        handle,
        (0.0, 0.0),
        (10.0, 0.0),
        [(0.0, 0.0), (20.0, 0.0), (20.0, 20.0), (0.0, 20.0)],
        closed=True,
        world_holes=[[(5.0, 5.0), (9.0, 5.0), (9.0, 9.0), (5.0, 9.0)]],
    )
    sketch.registry.points.extend([_center_pt, _handle_pt])
    sketch.registry.entities.append(polygon)
    sketch.registry._entity_map[polygon.id] = polygon
    circle = _circle_sketch(sketch, 40.0)
    regions = _regions(sketch, [polygon.id, circle])
    solid = next(r for r in regions if len(r.rings) == 2)
    assert get_polygon_signed_area(solid.rings[0]) > 0
    assert get_polygon_signed_area(solid.rings[1]) < 0


def test_regions_reject_open_chain():
    sketch = Sketch()
    p1 = sketch.add_point(0.0, 0.0)
    p2 = sketch.add_point(20.0, 0.0)
    line = sketch.add_line(p1, p2)
    circle = _circle_sketch(sketch, 40.0)
    assert build_boolean_regions(sketch, [line, circle]) is None


def test_regions_skip_construction_entities():
    sketch = Sketch()
    c1 = _circle_sketch(sketch, 0.0)
    c2 = _circle_sketch(sketch, 8.0)
    entity = sketch.registry.get_entity(c2)
    assert entity is not None
    entity.construction = True
    assert build_boolean_regions(sketch, [c1, c2]) is None


def test_regions_reject_single_region():
    sketch = Sketch()
    c1 = _circle_sketch(sketch, 0.0)
    assert build_boolean_regions(sketch, [c1]) is None


def test_union_merges_overlapping_circles():
    sketch = Sketch()
    c1 = _circle_sketch(sketch, 0.0)
    c2 = _circle_sketch(sketch, 8.0)
    regions = _regions(sketch, [c1, c2])
    solids = apply_boolean(BooleanOp.UNION, regions)
    assert solids is not None and len(solids) == 1
    xs = [p[0] for p in solids[0].outer]
    assert min(xs) == pytest.approx(-10.0, abs=0.05)
    assert max(xs) == pytest.approx(18.0, abs=0.05)
    assert solids[0].holes == []


def test_union_keeps_disjoint_pieces():
    sketch = Sketch()
    c1 = _circle_sketch(sketch, 0.0, 5.0)
    c2 = _circle_sketch(sketch, 50.0, 5.0)
    regions = _regions(sketch, [c1, c2])
    solids = apply_boolean(BooleanOp.UNION, regions)
    assert solids is not None and len(solids) == 2


def test_difference_carries_hole_with_winding():
    sketch = Sketch()
    lines = _square_sketch(sketch)
    circle = _circle_sketch(sketch, 10.0, 10.0, 6.0)
    regions = _regions(sketch, lines + [circle])
    solids = apply_boolean(BooleanOp.DIFFERENCE, regions)
    assert solids is not None and len(solids) == 1
    assert len(solids[0].holes) == 1
    assert get_polygon_signed_area(solids[0].outer) > 0
    assert get_polygon_signed_area(solids[0].holes[0]) < 0
    x_min, y_min, x_max, y_max = _ring_bbox(solids[0].outer)
    assert (x_min, y_min, x_max, y_max) == pytest.approx(
        (0.0, 0.0, 20.0, 20.0)
    )


def test_difference_uses_stacking_order():
    sketch = Sketch()
    lines = _square_sketch(sketch)
    circle = _circle_sketch(sketch, 10.0, 10.0, 6.0)
    regions = _regions(sketch, lines + [circle])
    regions.reverse()
    solids = apply_boolean(BooleanOp.DIFFERENCE, regions)
    assert solids is not None
    assert len(solids[0].holes) == 1


def test_intersection_produces_lens():
    sketch = Sketch()
    c1 = _circle_sketch(sketch, 0.0)
    c2 = _circle_sketch(sketch, 8.0)
    regions = _regions(sketch, [c1, c2])
    solids = apply_boolean(BooleanOp.INTERSECTION, regions)
    assert solids is not None and len(solids) == 1
    area = get_polygon_signed_area(solids[0].outer)
    assert area == pytest.approx(_lens_area(8.0, 10.0), rel=0.01)


def test_intersection_without_overlap_is_empty():
    sketch = Sketch()
    c1 = _circle_sketch(sketch, 0.0, 5.0)
    c2 = _circle_sketch(sketch, 50.0, 5.0)
    regions = _regions(sketch, [c1, c2])
    assert apply_boolean(BooleanOp.INTERSECTION, regions) is None


def test_exclude_produces_two_crescents():
    sketch = Sketch()
    c1 = _circle_sketch(sketch, 0.0)
    c2 = _circle_sketch(sketch, 8.0)
    regions = _regions(sketch, [c1, c2])
    solids = apply_boolean(BooleanOp.EXCLUDE, regions)
    assert solids is not None and len(solids) == 2
    for solid in solids:
        assert solid.holes == []
        assert get_polygon_signed_area(solid.outer) > 0
    total = sum(get_polygon_signed_area(solid.outer) for solid in solids)
    assert total == pytest.approx(
        2 * math.pi * 100 - 2 * _lens_area(8.0, 10.0), rel=0.01
    )


def test_entities_of_type_circle_sampled_fully():
    sketch = Sketch()
    c1 = _circle_sketch(sketch, 0.0)
    c2 = _circle_sketch(sketch, 8.0)
    regions = _regions(sketch, [c1, c2])
    for region in regions:
        assert isinstance(
            sketch.registry.get_entity(region.entity_ids[0]), Circle
        )
        assert len(region.rings[0]) >= 16
