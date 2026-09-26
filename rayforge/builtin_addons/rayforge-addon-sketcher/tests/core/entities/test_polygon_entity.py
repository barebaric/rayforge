"""Tests for PolygonEntity: frame math, serialization, and queries."""

import itertools
import math

import pytest
from raygeo.geo.shape.polygon import get_polygon_signed_area
from sketcher.core import Sketch
from sketcher.core.commands import MirrorAxis, MirrorDirection
from sketcher.core.entities import PolygonEntity
from sketcher.core.entities.polygon import offset_outline

SQUARE = [(0.0, 0.0), (20.0, 0.0), (20.0, 20.0), (0.0, 20.0)]


def _sketch_with_polygon(closed=True):
    sketch = Sketch()
    center_id = sketch.add_point(0.0, 0.0)
    handle_id = sketch.add_point(5.0, 0.0)
    polygon = PolygonEntity(
        9000,
        center_id,
        handle_id,
        PolygonEntity.normalize_vertices((0.0, 0.0), (5.0, 0.0), SQUARE),
        closed=closed,
    )
    sketch.registry.entities.append(polygon)
    sketch.registry._entity_map[polygon.id] = polygon
    return sketch, polygon


def test_world_vertex_roundtrip():
    sketch, polygon = _sketch_with_polygon()
    assert polygon.get_world_vertices(sketch.registry) == pytest.approx(SQUARE)


def test_frame_translation_follows_center():
    sketch, polygon = _sketch_with_polygon()
    center = sketch.registry.get_point(polygon.center_idx)
    handle = sketch.registry.get_point(polygon.handle_idx)
    center.x += 100.0
    center.y += 50.0
    handle.x += 100.0
    handle.y += 50.0
    assert polygon.get_world_vertices(sketch.registry) == pytest.approx(
        [(x + 100, y + 50) for x, y in SQUARE]
    )


def test_frame_rotation_follows_handle():
    sketch, polygon = _sketch_with_polygon()
    handle = sketch.registry.get_point(polygon.handle_idx)
    handle.x, handle.y = 0.0, 5.0
    assert polygon.get_world_vertices(sketch.registry) == pytest.approx(
        [(-y, x) for x, y in SQUARE], abs=1e-9
    )


def test_hit_test_on_edge():
    sketch, polygon = _sketch_with_polygon()
    registry = sketch.registry
    assert polygon.hit_test(10.0, 0.0, 0.5, registry)
    assert polygon.hit_test(0.0, 10.0, 0.5, registry)
    assert not polygon.hit_test(10.0, 10.0, 0.5, registry)


def test_to_geometry_closed_ring():
    sketch, polygon = _sketch_with_polygon(closed=True)
    polygons = polygon.to_geometry(sketch.registry).to_polygons(0.01)
    assert len(polygons) == 1
    assert len(polygons[0]) == 4
    assert set(map(tuple, polygons[0])) == set(map(tuple, SQUARE))


def test_to_polyline_open_stays_open():
    sketch, polygon = _sketch_with_polygon(closed=False)
    polyline = polygon.to_polyline(sketch.registry)
    assert polyline[0] != polyline[-1]


def test_serialization_roundtrip():
    _, polygon = _sketch_with_polygon()
    clone = PolygonEntity.from_dict(polygon.to_dict())
    assert clone.center_idx == polygon.center_idx
    assert clone.handle_idx == polygon.handle_idx
    assert clone.closed == polygon.closed
    assert clone.vertices == polygon.vertices


def test_state_roundtrip():
    _, polygon = _sketch_with_polygon()
    state = polygon.get_state()
    assert state is not None
    polygon.vertices = [(1.0, 2.0), (3.0, 4.0)]
    polygon.set_state(state)
    expected = PolygonEntity.normalize_vertices((0, 0), (5, 0), SQUARE)
    assert polygon.vertices == pytest.approx(expected)


def test_mirror_negates_local_v():
    sketch, polygon = _sketch_with_polygon()
    center = sketch.registry.get_point(polygon.center_idx)
    handle = sketch.registry.get_point(polygon.handle_idx)

    axis = MirrorAxis(MirrorDirection.HORIZONTAL, 0.0)
    polygon.mirror(axis)
    center.x, center.y = axis.apply(center.x, center.y)
    handle.x, handle.y = axis.apply(handle.x, handle.y)

    mirrored = sorted(polygon.get_world_vertices(sketch.registry))
    assert mirrored == pytest.approx(
        sorted([(-x, y) for x, y in SQUARE]), abs=1e-9
    )


def test_constrained_status_follows_frame_points():
    sketch, polygon = _sketch_with_polygon()
    polygon.update_constrained_status(sketch.registry, [])
    assert polygon.constrained is False
    for pid in polygon.get_point_ids():
        sketch.registry.get_point(pid).constrained = True
    polygon.update_constrained_status(sketch.registry, [])
    assert polygon.constrained is True


def test_rigid_and_anchor_points():
    _, polygon = _sketch_with_polygon()
    assert polygon.get_rigidly_connected_points(polygon.center_idx) == [
        polygon.center_idx,
        polygon.handle_idx,
    ]
    assert polygon.get_rigidly_connected_points(polygon.handle_idx) == []
    assert polygon.get_drag_anchor_points(polygon.handle_idx) == [
        polygon.center_idx
    ]


def test_offset_outline_square_grows():
    results = offset_outline(SQUARE, True, 2.0)
    assert len(results) == 1
    vertices, closed = results[0]
    assert closed is True
    xs = [v[0] for v in vertices]
    ys = [v[1] for v in vertices]
    assert max(xs) - min(xs) == pytest.approx(24.0, abs=0.05)
    assert max(ys) - min(ys) == pytest.approx(24.0, abs=0.05)


def test_offset_outline_line_becomes_slot():
    results = offset_outline([(0.0, 0.0), (40.0, 0.0)], False, 5.0)
    assert len(results) == 1
    vertices, closed = results[0]
    assert closed is True
    xs = [v[0] for v in vertices]
    ys = [v[1] for v in vertices]
    assert max(xs) - min(xs) == pytest.approx(50.0, abs=0.05)
    assert max(ys) - min(ys) == pytest.approx(10.0, abs=0.05)


def test_offset_outline_collapse_returns_empty():
    assert offset_outline(SQUARE, True, -100.0) == []


def _handle_on_contour_dist_sq(grown, handle):
    from raygeo.geo.shape.polygon import get_polygon_closest_point

    result = get_polygon_closest_point(grown, handle[0], handle[1])
    assert result is not None
    return result[2]


def test_frame_closed_center_inside_handle_on_contour():
    from raygeo.geo.shape.polygon import is_point_inside_polygon
    from sketcher.core.entities.polygon import _frame_for_outline

    grown = offset_outline(SQUARE, True, 2.0)[0][0]
    center, handle = _frame_for_outline(grown, closed=True)
    assert is_point_inside_polygon(center, grown)
    assert _handle_on_contour_dist_sq(grown, handle) < 1e-6


def test_frame_handle_at_long_axis_end():
    """For an elongated outline the handle sits at the far end of the
    long axis, not at the narrow inscribed-circle edge."""
    from sketcher.core.entities.polygon import _frame_for_outline

    stadium = offset_outline([(0.0, 0.0), (100.0, 0.0)], False, 5.0)[0][0]
    center, handle = _frame_for_outline(stadium, closed=True)
    assert _handle_on_contour_dist_sq(stadium, handle) < 1e-6
    frame_scale = math.hypot(handle[0] - center[0], handle[1] - center[1])
    half_long = 55.0
    assert frame_scale >= half_long - 1.0


def test_frame_open_handle_on_polyline():
    from sketcher.core.entities.polygon import (
        _closest_outline_point,
        _frame_for_outline,
    )

    path = [(0.0, 0.0), (30.0, 10.0), (60.0, 0.0)]
    _center, handle = _frame_for_outline(path, closed=False)
    result = _closest_outline_point(path, False, *handle)
    assert result is not None
    assert result[1] < 1e-6


def test_frame_diagonal_line_nondegenerate():
    """A straight diagonal through its bbox center still yields a
    usable frame (handle strictly off-center)."""
    from sketcher.core.entities.polygon import _frame_for_outline

    center, handle = _frame_for_outline([(0.0, 0.0), (40.0, 0.0)], False)
    assert (center[0], center[1]) != (handle[0], handle[1])
    assert math.hypot(handle[0] - center[0], handle[1] - center[1]) > 1e-6


# ---------------------------------------------------------------------------
# Multi-ring polygons (holes carried by ring winding: outer CCW, holes CW)
# ---------------------------------------------------------------------------


class _MirrorPlacement:
    def transform_point(self, x, y):
        return (-x, y)

    def transform_offset(self, dx, dy):
        return (-dx, dy)


class _TranslationPlacement:
    def __init__(self, dx, dy):
        self.dx = dx
        self.dy = dy

    def transform_point(self, x, y):
        return (x + self.dx, y + self.dy)

    def transform_offset(self, dx, dy):
        return (dx, dy)


def _build_solid_sketch():
    """A 20x20 square with a 4x4 hole at (5,5)-(9,9), built through
    PolygonEntity.build so holes share the outer ring's frame."""
    sketch = Sketch()
    center_id = sketch.add_point(0.0, 0.0)
    handle_id = sketch.add_point(10.0, 0.0)
    center_pt, handle_pt, polygon = PolygonEntity.build(
        9100,
        center_id,
        handle_id,
        (0.0, 0.0),
        (10.0, 0.0),
        [(0.0, 0.0), (20.0, 0.0), (20.0, 20.0), (0.0, 20.0)],
        closed=True,
        world_holes=[[(5.0, 5.0), (5.0, 9.0), (9.0, 9.0), (9.0, 5.0)]],
    )
    sketch.registry.points.extend([center_pt, handle_pt])
    sketch.registry.entities.append(polygon)
    sketch.registry._entity_map[polygon.id] = polygon
    return sketch, polygon


def test_hole_winding_normalized_to_cw():
    sketch, polygon = _build_solid_sketch()
    areas = [
        get_polygon_signed_area(ring)
        for ring in polygon.get_world_rings(sketch.registry)
    ]
    assert areas[0] > 0
    assert all(area < 0 for area in areas[1:])


def test_contains_point_respects_holes():
    sketch, polygon = _build_solid_sketch()
    assert polygon.contains_point(sketch.registry, 1.0, 1.0) is True
    assert polygon.contains_point(sketch.registry, 7.0, 7.0) is False
    assert polygon.contains_point(sketch.registry, 25.0, 10.0) is False


def test_enclosed_signed_area_subtracts_holes():
    sketch, polygon = _build_solid_sketch()
    assert polygon.enclosed_signed_area(sketch.registry) == pytest.approx(
        400 - 16
    )


def test_multi_ring_serialization_roundtrip():
    sketch, polygon = _build_solid_sketch()
    clone = PolygonEntity.from_dict(polygon.to_dict())
    assert len(clone.rings) == 2
    assert clone.closed is True
    for ring_clone, ring in zip(
        clone.get_world_rings(sketch.registry),
        polygon.get_world_rings(sketch.registry),
    ):
        assert ring_clone == pytest.approx(ring)


def test_multi_ring_legacy_dict_without_rings():
    data = {
        "id": 9101,
        "center_idx": 1,
        "handle_idx": 2,
        "vertices": [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0]],
        "closed": True,
    }
    clone = PolygonEntity.from_dict(data)
    assert len(clone.rings) == 1


def test_mirror_preserves_winding_convention():
    sketch, polygon = _build_solid_sketch()
    before = polygon.get_world_rings(sketch.registry)
    polygon.mirror(MirrorAxis(MirrorDirection.HORIZONTAL, 0.0))
    after = polygon.get_world_rings(sketch.registry)
    for ring_before, ring_after in zip(before, after):
        assert get_polygon_signed_area(ring_after) == pytest.approx(
            get_polygon_signed_area(ring_before)
        )
        mirrored = sorted([(x, -y) for x, y in ring_before])
        assert sorted(ring_after) == pytest.approx(mirrored)


def test_transform_offsets_mirror_flips_rings():
    sketch, polygon = _build_solid_sketch()
    polygon.transform_offsets(_MirrorPlacement())
    rings = polygon.get_world_rings(sketch.registry)
    assert get_polygon_signed_area(rings[0]) > 0
    assert get_polygon_signed_area(rings[1]) < 0


def test_rewrite_offsets_from_template():
    _sketch, template = _build_solid_sketch()
    copy = PolygonEntity(
        9199,
        template.center_idx,
        template.handle_idx,
        template.rings[0],
        closed=True,
        rings=template.rings,
    )
    copy.rewrite_offsets_from(template, _TranslationPlacement(100.0, 50.0))
    assert copy.rings == template.rings


def test_multi_ring_hit_test_and_containment():
    sketch, polygon = _build_solid_sketch()
    assert polygon.hit_test(0.0, 0.0, 0.5, sketch.registry) is True
    assert polygon.hit_test(5.0, 5.0, 0.5, sketch.registry) is True
    assert (
        polygon.is_contained_by((-1.0, -1.0, 21.0, 21.0), sketch.registry)
        is True
    )
    assert (
        polygon.is_contained_by((1.0, 1.0, 19.0, 19.0), sketch.registry)
        is False
    )


def test_multi_ring_to_geometry_keeps_holes():
    sketch, polygon = _build_solid_sketch()
    polygons = polygon.to_geometry(sketch.registry).to_polygons(0.02)
    assert len(polygons) == 2
    assert get_polygon_signed_area(polygons[0]) > 0
    assert get_polygon_signed_area(polygons[1]) < 0


def test_multi_ring_offset_shrinks_holes():
    sketch, polygon = _build_solid_sketch()
    plan = polygon.plan_offset(
        sketch.registry, 1.0, itertools.count(1000).__next__
    )
    assert plan is not None and len(plan.entities) == 1
    assert plan.removed_entity_ids == [polygon.id]
    result = plan.entities[0]
    assert isinstance(result, PolygonEntity)
    sketch.registry.points.extend(plan.points)
    rings = result.get_world_rings(sketch.registry)
    assert len(rings) == 2
    xs = [p[0] for p in rings[0]]
    ys = [p[1] for p in rings[0]]
    assert (min(xs), min(ys), max(xs), max(ys)) == pytest.approx(
        (-1.0, -1.0, 21.0, 21.0), abs=0.05
    )
    hole_xs = [p[0] for p in rings[1]]
    assert max(hole_xs) - min(hole_xs) == pytest.approx(2.0, abs=0.05)


def test_multi_ring_offset_vanishes_hole_first():
    sketch, polygon = _build_solid_sketch()
    plan = polygon.plan_offset(
        sketch.registry, -4.5, itertools.count(1000).__next__
    )
    assert plan is not None and len(plan.entities) == 1
    result = plan.entities[0]
    assert isinstance(result, PolygonEntity)
    sketch.registry.points.extend(plan.points)
    rings = result.get_world_rings(sketch.registry)
    assert len(rings) == 1


def test_multi_ring_offset_collapse_returns_none():
    sketch, polygon = _build_solid_sketch()
    plan = polygon.plan_offset(
        sketch.registry, -12.0, itertools.count(1000).__next__
    )
    assert plan is None
