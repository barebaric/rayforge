"""Tests for PolygonEntity: frame math, serialization, and queries."""

import math

import pytest
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

    assert polygon.get_world_vertices(sketch.registry) == pytest.approx(
        [(-x, y) for x, y in SQUARE], abs=1e-9
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
