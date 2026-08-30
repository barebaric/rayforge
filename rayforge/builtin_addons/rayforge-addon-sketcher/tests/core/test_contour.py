"""Tests for selection preprocessing into offsettable items."""

from sketcher.core import Sketch
from sketcher.core.contour import build_offset_items
from sketcher.core.entities import Arc, Circle, Ellipse, PolygonOutline


def _square_sketch():
    """A closed 20x20 square of four connected lines."""
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
    return sketch, entity_ids


def test_closed_square_chain():
    sketch, entity_ids = _square_sketch()
    items = build_offset_items(sketch, entity_ids)
    assert items is not None and len(items) == 1
    outline = items[0]
    assert isinstance(outline, PolygonOutline)
    assert outline.closed is True
    assert len(outline.vertices) == 4
    assert sorted(outline.source_ids) == sorted(entity_ids)
    assert set(map(tuple, outline.vertices)) == {
        (0.0, 0.0),
        (20.0, 0.0),
        (20.0, 20.0),
        (0.0, 20.0),
    }


def test_open_chain_stays_open():
    sketch = Sketch()
    p1 = sketch.add_point(0, 0)
    p2 = sketch.add_point(10, 0)
    p3 = sketch.add_point(10, 10)
    line_id = sketch.add_line(p1, p2)
    line2_id = sketch.add_line(p2, p3)

    items = build_offset_items(sketch, [line_id, line2_id])
    assert items is not None and len(items) == 1
    outline = items[0]
    assert isinstance(outline, PolygonOutline)
    assert outline.closed is False
    assert outline.vertices[0] == (0.0, 0.0)
    assert outline.vertices[-1] == (10.0, 10.0)


def test_lone_line_becomes_open_outline():
    sketch = Sketch()
    p1 = sketch.add_point(0, 0)
    p2 = sketch.add_point(40, 0)
    line_id = sketch.add_line(p1, p2)

    items = build_offset_items(sketch, [line_id])
    assert items is not None and len(items) == 1
    outline = items[0]
    assert isinstance(outline, PolygonOutline)
    assert outline.closed is False
    assert len(outline.vertices) == 2


def test_lone_circle_passes_through():
    sketch = Sketch()
    center = sketch.add_point(0, 0)
    radius_pt = sketch.add_point(10, 0)
    circle_id = sketch.add_circle(center, radius_pt)

    items = build_offset_items(sketch, [circle_id])
    assert items is not None and len(items) == 1
    assert isinstance(items[0], Circle)
    assert items[0].id == circle_id


def test_lone_arc_passes_through():
    sketch = Sketch()
    start = sketch.add_point(10, 0)
    end = sketch.add_point(-10, 0)
    center = sketch.add_point(0, 0)
    arc_id = sketch.add_arc(start, end, center, clockwise=False)

    items = build_offset_items(sketch, [arc_id])
    assert items is not None and len(items) == 1
    assert isinstance(items[0], Arc)
    assert items[0].id == arc_id


def test_lone_ellipse_passes_through():
    sketch = Sketch()
    center = sketch.add_point(0, 0)
    rx_pt = sketch.add_point(10, 0)
    ry_pt = sketch.add_point(0, 5)
    ellipse_id = sketch.registry.add_ellipse(center, rx_pt, ry_pt)

    items = build_offset_items(sketch, [ellipse_id])
    assert items is not None and len(items) == 1
    assert isinstance(items[0], Ellipse)
    assert items[0].id == ellipse_id


def test_multiple_components_partitioned():
    sketch, square_ids = _square_sketch()
    circle_center = sketch.add_point(100, 100)
    circle_radius = sketch.add_point(110, 100)
    circle_id = sketch.add_circle(circle_center, circle_radius)

    items = build_offset_items(sketch, square_ids + [circle_id])
    assert items is not None and len(items) == 2
    kinds = sorted(type(item).__name__ for item in items)
    assert kinds == ["Circle", "PolygonOutline"]


def test_t_junction_rejected():
    sketch = Sketch()
    p1 = sketch.add_point(0, 0)
    corner = sketch.add_point(10, 0)
    p3 = sketch.add_point(20, 0)
    p4 = sketch.add_point(10, 10)
    e1 = sketch.add_line(p1, corner)
    e2 = sketch.add_line(corner, p3)
    e3 = sketch.add_line(corner, p4)

    assert build_offset_items(sketch, [e1, e2, e3]) is None


def test_disconnected_lines_partition_into_two_items():
    sketch = Sketch()
    p1 = sketch.add_point(0, 0)
    p2 = sketch.add_point(10, 0)
    p3 = sketch.add_point(100, 0)
    p4 = sketch.add_point(110, 0)
    e1 = sketch.add_line(p1, p2)
    e2 = sketch.add_line(p3, p4)

    items = build_offset_items(sketch, [e1, e2])
    assert items is not None and len(items) == 2
    assert all(
        isinstance(item, PolygonOutline) and not item.closed for item in items
    )


def test_coincident_endpoints_are_merged():
    sketch = Sketch()
    p1 = sketch.add_point(0, 0)
    p2 = sketch.add_point(10, 0)
    p2_dup = sketch.add_point(10, 0)
    p3 = sketch.add_point(10, 10)
    e1 = sketch.add_line(p1, p2)
    e2 = sketch.add_line(p2_dup, p3)
    sketch.constrain_coincident(p2, p2_dup)

    items = build_offset_items(sketch, [e1, e2])
    assert items is not None and len(items) == 1
    outline = items[0]
    assert isinstance(outline, PolygonOutline)
    assert outline.closed is False
    assert len(outline.vertices) == 3


def test_empty_selection_returns_none():
    sketch = Sketch()
    assert build_offset_items(sketch, []) is None
