"""Tests for BooleanCommand: baking, source replacement and undo."""

import math

import pytest
from raygeo.geo.shape.polygon import get_polygon_signed_area
from sketcher.core import Sketch
from sketcher.core.boolean import BooleanOp
from sketcher.core.commands import BooleanCommand
from sketcher.core.entities import Circle, PolygonEntity


def _square(sketch, x=0.0, y=0.0, size=20.0):
    ids = [
        sketch.add_point(x, y),
        sketch.add_point(x + size, y),
        sketch.add_point(x + size, y + size),
        sketch.add_point(x, y + size),
    ]
    return [
        sketch.add_line(ids[0], ids[1]),
        sketch.add_line(ids[1], ids[2]),
        sketch.add_line(ids[2], ids[3]),
        sketch.add_line(ids[3], ids[0]),
    ]


def _circle(sketch, cx=0.0, cy=0.0, r=10.0):
    center = sketch.add_point(cx, cy)
    radius_pt = sketch.add_point(cx + r, cy)
    return sketch.add_circle(center, radius_pt)


def _polygons(sketch):
    return [
        e for e in sketch.registry.entities if isinstance(e, PolygonEntity)
    ]


def test_union_replaces_sources_with_polygon():
    sketch = Sketch()
    c1 = _circle(sketch, 0.0)
    c2 = _circle(sketch, 8.0)
    cmd = BooleanCommand(sketch, [c1, c2], BooleanOp.UNION)
    cmd.execute()

    assert sketch.registry.get_entity(c1) is None
    assert sketch.registry.get_entity(c2) is None
    polygons = _polygons(sketch)
    assert len(polygons) == 1
    assert polygons[0].id in cmd.new_entity_ids
    vertices = polygons[0].get_world_vertices(sketch.registry)
    xs = [v[0] for v in vertices]
    assert min(xs) == pytest.approx(-10.0, abs=0.05)
    assert max(xs) == pytest.approx(18.0, abs=0.05)


def test_union_undo_restores_sources():
    sketch = Sketch()
    c1 = _circle(sketch, 0.0)
    c2 = _circle(sketch, 8.0)
    cmd = BooleanCommand(sketch, [c1, c2], BooleanOp.UNION)
    cmd.execute()
    cmd.undo()

    assert isinstance(sketch.registry.get_entity(c1), Circle)
    assert isinstance(sketch.registry.get_entity(c2), Circle)
    assert _polygons(sketch) == []


def _point_state(sketch):
    return {p.id: (p.x, p.y) for p in sketch.registry.points}


def test_undo_restores_exact_geometry():
    sketch = Sketch()
    c1 = _circle(sketch, 0.0)
    c2 = _circle(sketch, 8.0)
    before = _point_state(sketch)
    cmd = BooleanCommand(sketch, [c1, c2], BooleanOp.UNION)
    cmd.execute()
    cmd.undo()
    assert _point_state(sketch) == pytest.approx(before)


def test_difference_bakes_hole_into_single_entity():
    sketch = Sketch()
    lines = _square(sketch)
    circle = _circle(sketch, 10.0, 10.0, 6.0)
    cmd = BooleanCommand(sketch, lines + [circle], BooleanOp.DIFFERENCE)
    cmd.execute()

    polygons = _polygons(sketch)
    assert len(polygons) == 1
    polygon = polygons[0]
    assert polygon.closed is True
    assert len(polygon.rings) == 2
    rings = polygon.get_world_rings(sketch.registry)
    assert get_polygon_signed_area(rings[0]) > 0
    assert get_polygon_signed_area(rings[1]) < 0
    assert polygon.contains_point(sketch.registry, 1.0, 1.0) is True
    assert polygon.contains_point(sketch.registry, 10.0, 10.0) is False
    assert polygon.enclosed_signed_area(sketch.registry) == pytest.approx(
        400 - math.pi * 36, rel=0.01
    )


def test_difference_subtracts_topmost_from_bottom():
    sketch = Sketch()
    lines = _square(sketch)
    circle = _circle(sketch, 10.0, 10.0, 6.0)
    reverse = BooleanCommand(sketch, [circle] + lines, BooleanOp.DIFFERENCE)
    reverse.execute()

    polygons = _polygons(sketch)
    assert len(polygons) == 1
    assert len(polygons[0].rings) == 2


def test_multi_ring_polygon_survives_serialization():
    sketch = Sketch()
    lines = _square(sketch)
    circle = _circle(sketch, 10.0, 10.0, 6.0)
    cmd = BooleanCommand(sketch, lines + [circle], BooleanOp.DIFFERENCE)
    cmd.execute()

    polygon = _polygons(sketch)[0]
    clone = PolygonEntity.from_dict(polygon.to_dict())
    assert len(clone.rings) == 2
    assert clone.closed is True
    for ring_clone, ring in zip(
        clone.get_world_rings(sketch.registry),
        polygon.get_world_rings(sketch.registry),
    ):
        assert ring_clone == pytest.approx(ring)


def test_result_is_selectable_and_cuttable():
    sketch = Sketch()
    lines = _square(sketch)
    circle = _circle(sketch, 10.0, 10.0, 6.0)
    cmd = BooleanCommand(sketch, lines + [circle], BooleanOp.DIFFERENCE)
    cmd.execute()

    polygon = _polygons(sketch)[0]
    assert polygon.is_closed_loop() is True
    assert polygon.hit_test(0.0, 0.0, 0.5, sketch.registry) is True
    assert (
        polygon.is_contained_by((-1.0, -1.0, 21.0, 21.0), sketch.registry)
        is True
    )
    geo = polygon.to_geometry(sketch.registry)
    assert len(geo.to_polygons(0.02)) == 2


def test_no_overlap_aborts_without_changes():
    sketch = Sketch()
    c1 = _circle(sketch, 0.0, 0.0, 5.0)
    c2 = _circle(sketch, 50.0, 0.0, 5.0)
    cmd = BooleanCommand(sketch, [c1, c2], BooleanOp.INTERSECTION)
    cmd.execute()

    assert isinstance(sketch.registry.get_entity(c1), Circle)
    assert isinstance(sketch.registry.get_entity(c2), Circle)
    assert _polygons(sketch) == []


def test_insufficient_selection_aborts():
    sketch = Sketch()
    c1 = _circle(sketch, 0.0)
    cmd = BooleanCommand(sketch, [c1], BooleanOp.UNION)
    cmd.execute()
    assert isinstance(sketch.registry.get_entity(c1), Circle)


def test_prepare_solids_is_side_effect_free():
    sketch = Sketch()
    c1 = _circle(sketch, 0.0)
    c2 = _circle(sketch, 8.0)
    points_before = _point_state(sketch)
    entities_before = len(sketch.registry.entities)
    solids = BooleanCommand.prepare_solids(sketch, [c1, c2], BooleanOp.UNION)
    assert solids is not None and len(solids) == 1
    assert _point_state(sketch) == pytest.approx(points_before)
    assert len(sketch.registry.entities) == entities_before
