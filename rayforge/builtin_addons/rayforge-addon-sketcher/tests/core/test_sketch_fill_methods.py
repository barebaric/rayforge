import pytest

from sketcher.core import Sketch
from sketcher.core.sketch import Fill
from sketcher.core.entities import Line


@pytest.fixture
def sketch():
    return Sketch()


@pytest.fixture
def triangle_sketch(sketch):
    p1 = sketch.add_point(0, 0)
    p2 = sketch.add_point(100, 0)
    p3 = sketch.add_point(50, 86.6)

    sketch.add_line(p1, p2)
    sketch.add_line(p2, p3)
    sketch.add_line(p3, p1)

    return sketch


def test_validate_and_cleanup_fills_removes_invalid(sketch):
    p1 = sketch.add_point(0, 0)
    p2 = sketch.add_point(100, 0)
    p3 = sketch.add_point(50, 86.6)

    line1_id = sketch.add_line(p1, p2)
    line2_id = sketch.add_line(p2, p3)
    line3_id = sketch.add_line(p3, p1)

    valid_fill = Fill(
        "valid-fill",
        [(line1_id, True), (line2_id, True), (line3_id, True)],
    )
    sketch.fills.append(valid_fill)

    invalid_fill = Fill(
        "invalid-fill",
        [(line1_id, True), (9999, True)],
    )
    sketch.fills.append(invalid_fill)

    sketch._validate_and_cleanup_fills()

    assert valid_fill in sketch.fills
    assert invalid_fill not in sketch.fills


def test_validate_and_cleanup_fills_keeps_circle(sketch):
    center = sketch.add_point(50, 50)
    radius = sketch.add_point(60, 50)
    circle_id = sketch.add_circle(center, radius)

    fill = Fill("circle-fill", [(circle_id, True)])
    sketch.fills.append(fill)

    sketch._validate_and_cleanup_fills()

    assert fill in sketch.fills


def test_validate_and_cleanup_fills_keeps_all_valid(triangle_sketch):
    line_ids = [
        e.id for e in triangle_sketch.registry.entities if isinstance(e, Line)
    ]

    fill = Fill(
        "triangle-fill",
        [(line_ids[0], True), (line_ids[1], True), (line_ids[2], True)],
    )
    triangle_sketch.fills.append(fill)

    triangle_sketch._validate_and_cleanup_fills()

    assert fill in triangle_sketch.fills


def test_get_fill_geometries_empty(sketch):
    result = sketch.get_fill_geometries()
    assert result == []


def test_get_fill_geometries_circle(sketch):
    center = sketch.add_point(50, 50)
    radius = sketch.add_point(70, 50)
    circle_id = sketch.add_circle(center, radius)

    fill = Fill("circle-fill", [(circle_id, True)])
    sketch.fills.append(fill)

    geometries = sketch.get_fill_geometries()

    assert len(geometries) == 1


def test_get_fill_geometries_triangle(triangle_sketch):
    line_ids = [
        e.id for e in triangle_sketch.registry.entities if isinstance(e, Line)
    ]

    fill = Fill(
        "triangle-fill",
        [(line_ids[0], True), (line_ids[1], True), (line_ids[2], True)],
    )
    triangle_sketch.fills.append(fill)

    geometries = triangle_sketch.get_fill_geometries()

    assert len(geometries) == 1


def test_get_fill_geometries_excludes_ids(sketch):
    center = sketch.add_point(50, 50)
    radius = sketch.add_point(70, 50)
    circle_id = sketch.add_circle(center, radius)

    fill = Fill("circle-fill", [(circle_id, True)])
    sketch.fills.append(fill)

    geometries = sketch.get_fill_geometries(exclude_ids={circle_id})

    assert len(geometries) == 0


def test_get_fill_geometries_missing_entity(sketch):
    fill = Fill("missing-fill", [(9999, True)])
    sketch.fills.append(fill)

    geometries = sketch.get_fill_geometries()

    assert len(geometries) == 0


def test_get_loop_at_point_inside_triangle(triangle_sketch):
    loop = triangle_sketch.get_loop_at_point(50, 30)

    assert loop is not None
    assert len(loop) == 3


def test_get_loop_at_point_outside_triangle(triangle_sketch):
    loop = triangle_sketch.get_loop_at_point(200, 200)

    assert loop is None


def test_get_loop_at_point_inside_circle(sketch):
    center = sketch.add_point(50, 50)
    radius = sketch.add_point(80, 50)
    sketch.add_circle(center, radius)

    loop = sketch.get_loop_at_point(50, 50)

    assert loop is not None
    assert len(loop) == 1


def test_get_loop_at_point_outside_circle(sketch):
    center = sketch.add_point(50, 50)
    radius = sketch.add_point(80, 50)
    sketch.add_circle(center, radius)

    loop = sketch.get_loop_at_point(200, 200)

    assert loop is None


def test_get_loop_at_point_nested_loops(sketch):
    outer_p1 = sketch.add_point(0, 0)
    outer_p2 = sketch.add_point(100, 0)
    outer_p3 = sketch.add_point(100, 100)
    outer_p4 = sketch.add_point(0, 100)

    sketch.add_line(outer_p1, outer_p2)
    sketch.add_line(outer_p2, outer_p3)
    sketch.add_line(outer_p3, outer_p4)
    sketch.add_line(outer_p4, outer_p1)

    inner_p1 = sketch.add_point(25, 25)
    inner_p2 = sketch.add_point(75, 25)
    inner_p3 = sketch.add_point(75, 75)
    inner_p4 = sketch.add_point(25, 75)

    sketch.add_line(inner_p1, inner_p2)
    sketch.add_line(inner_p2, inner_p3)
    sketch.add_line(inner_p3, inner_p4)
    sketch.add_line(inner_p4, inner_p1)

    outer_loop = sketch.get_loop_at_point(10, 10)
    inner_loop = sketch.get_loop_at_point(50, 50)

    assert outer_loop is not None
    assert inner_loop is not None
    assert len(inner_loop) == 4


def test_get_loop_at_point_open_path(sketch):
    p1 = sketch.add_point(0, 0)
    p2 = sketch.add_point(100, 0)
    p3 = sketch.add_point(50, 86.6)

    sketch.add_line(p1, p2)
    sketch.add_line(p2, p3)

    loop = sketch.get_loop_at_point(50, 30)

    assert loop is None
