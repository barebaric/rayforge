"""Tests for OffsetCommand: in-place updates and contour replacement."""

import pytest
from raygeo.geo.shape.polygon import (
    get_polygon_closest_point,
    is_point_inside_polygon,
)
from sketcher.core import Sketch
from sketcher.core.commands import OffsetCommand
from sketcher.core.entities import Circle, Ellipse, PolygonEntity


def _circle_sketch():
    sketch = Sketch()
    center = sketch.add_point(0, 0)
    radius_pt = sketch.add_point(10, 0)
    circle_id = sketch.add_circle(center, radius_pt)
    return sketch, circle_id


def test_circle_offset_updates_in_place():
    sketch, circle_id = _circle_sketch()
    cmd = OffsetCommand(sketch, [circle_id], 5.0)
    cmd.execute()

    circles = [e for e in sketch.registry.entities if isinstance(e, Circle)]
    assert len(circles) == 1
    assert circles[0].id == circle_id
    radius_pt = sketch.registry.get_point(circles[0].radius_pt_idx)
    assert radius_pt.x == pytest.approx(15.0)
    assert radius_pt.y == pytest.approx(0.0)

    cmd.undo()
    radius_pt = sketch.registry.get_point(circles[0].radius_pt_idx)
    assert radius_pt.x == pytest.approx(10.0)


def test_circle_offset_shrinks():
    sketch, circle_id = _circle_sketch()
    cmd = OffsetCommand(sketch, [circle_id], -4.0)
    cmd.execute()

    circle = sketch.registry.get_entity(circle_id)
    assert isinstance(circle, Circle)
    radius_pt = sketch.registry.get_point(circle.radius_pt_idx)
    assert radius_pt.x == pytest.approx(6.0)


def test_circle_offset_collapse_aborts():
    sketch, circle_id = _circle_sketch()
    cmd = OffsetCommand(sketch, [circle_id], -50.0)
    cmd.execute()
    circle = sketch.registry.get_entity(circle_id)
    assert isinstance(circle, Circle)
    radius_pt = sketch.registry.get_point(circle.radius_pt_idx)
    assert radius_pt.x == pytest.approx(10.0)


def test_ellipse_offset_updates_in_place():
    sketch = Sketch()
    center = sketch.add_point(0, 0)
    rx_pt = sketch.add_point(10, 0)
    ry_pt = sketch.add_point(0, 5)
    ellipse_id = sketch.registry.add_ellipse(center, rx_pt, ry_pt)

    cmd = OffsetCommand(sketch, [ellipse_id], 2.0)
    cmd.execute()

    ellipses = [e for e in sketch.registry.entities if isinstance(e, Ellipse)]
    assert len(ellipses) == 1
    assert ellipses[0].id == ellipse_id
    rx_pt = sketch.registry.get_point(ellipses[0].radius_x_pt_idx)
    ry_pt = sketch.registry.get_point(ellipses[0].radius_y_pt_idx)
    assert rx_pt.x == pytest.approx(12.0)
    assert ry_pt.y == pytest.approx(7.0)

    cmd.undo()
    rx_pt = sketch.registry.get_point(ellipses[0].radius_x_pt_idx)
    ry_pt = sketch.registry.get_point(ellipses[0].radius_y_pt_idx)
    assert rx_pt.x == pytest.approx(10.0)
    assert ry_pt.y == pytest.approx(5.0)


def test_square_offset_replaces_lines_with_polygon():
    sketch = Sketch()
    ids = [
        sketch.add_point(0, 0),
        sketch.add_point(20, 0),
        sketch.add_point(20, 20),
        sketch.add_point(0, 20),
    ]
    entity_ids = [
        sketch.add_line(ids[0], ids[1]),
        sketch.add_line(ids[1], ids[2]),
        sketch.add_line(ids[2], ids[3]),
        sketch.add_line(ids[3], ids[0]),
    ]

    cmd = OffsetCommand(sketch, entity_ids, 2.0)
    cmd.execute()

    for eid in entity_ids:
        assert sketch.registry.get_entity(eid) is None

    polygons = [
        e for e in sketch.registry.entities if isinstance(e, PolygonEntity)
    ]
    assert len(polygons) == 1
    assert polygons[0].closed is True
    vertices = polygons[0].get_world_vertices(sketch.registry)
    xs = [v[0] for v in vertices]
    ys = [v[1] for v in vertices]
    assert min(xs) == pytest.approx(-2.0, abs=0.05)
    assert max(xs) == pytest.approx(22.0, abs=0.05)
    assert min(ys) == pytest.approx(-2.0, abs=0.05)
    assert max(ys) == pytest.approx(22.0, abs=0.05)

    center = sketch.registry.get_point(polygons[0].center_idx)
    handle = sketch.registry.get_point(polygons[0].handle_idx)
    assert is_point_inside_polygon((center.x, center.y), vertices)
    closest = get_polygon_closest_point(vertices, handle.x, handle.y)
    assert closest is not None
    assert closest[2] < 1e-6

    cmd.undo()
    for eid in entity_ids:
        assert sketch.registry.get_entity(eid) is not None
    assert not [
        e for e in sketch.registry.entities if isinstance(e, PolygonEntity)
    ]


def test_line_offset_becomes_slot():
    sketch = Sketch()
    p1 = sketch.add_point(0, 0)
    p2 = sketch.add_point(40, 0)
    line_id = sketch.add_line(p1, p2)

    cmd = OffsetCommand(sketch, [line_id], 5.0)
    cmd.execute()

    assert sketch.registry.get_entity(line_id) is None

    polygons = [
        e for e in sketch.registry.entities if isinstance(e, PolygonEntity)
    ]
    assert len(polygons) == 1
    assert polygons[0].closed is True
    vertices = polygons[0].get_world_vertices(sketch.registry)
    ys = [v[1] for v in vertices]
    assert max(ys) - min(ys) == pytest.approx(10.0, abs=0.05)


def test_preview_polylines_for_circle():
    sketch, circle_id = _circle_sketch()
    items = OffsetCommand.prepare_items(sketch, [circle_id])
    assert items is not None
    polylines = OffsetCommand.preview_polylines(items, sketch.registry, 5.0)
    assert polylines is not None and len(polylines) == 1
    xs = [p[0] for p in polylines[0]]
    assert max(xs) - min(xs) == pytest.approx(30.0, abs=0.05)


def test_mixed_selection_all_or_nothing():
    """A collapsing circle aborts the offset of the other item too."""
    sketch, circle_id = _circle_sketch()

    p1 = sketch.add_point(100, 0)
    p2 = sketch.add_point(140, 0)
    line_id = sketch.add_line(p1, p2)

    cmd = OffsetCommand(sketch, [circle_id, line_id], -50.0)
    cmd.execute()

    assert sketch.registry.get_entity(line_id) is not None
    assert not [
        e for e in sketch.registry.entities if isinstance(e, PolygonEntity)
    ]


def test_empty_selection_aborts():
    sketch = Sketch()
    cmd = OffsetCommand(sketch, [], 2.0)
    cmd.execute()
    assert not sketch.registry.entities
