import pytest

from rayforge.core.sketcher import Sketch
from rayforge.core.sketcher.commands import (
    TangentConstraintCommand,
    TangentConstraintParams,
)


@pytest.fixture
def sketch():
    return Sketch()


@pytest.fixture
def line_and_arc(sketch):
    p1 = sketch.add_point(0, 0)
    p2 = sketch.add_point(100, 0)
    line_id = sketch.add_line(p1, p2)

    center = sketch.add_point(50, 50)
    radius = sketch.add_point(60, 50)
    arc_id = sketch.add_arc(center, radius, 0, 180)

    return sketch, line_id, arc_id


@pytest.fixture
def line_and_circle(sketch):
    p1 = sketch.add_point(0, 0)
    p2 = sketch.add_point(100, 0)
    line_id = sketch.add_line(p1, p2)

    center = sketch.add_point(50, 50)
    radius = sketch.add_point(60, 50)
    circle_id = sketch.add_circle(center, radius)

    return sketch, line_id, circle_id


def test_identify_entities_line_and_arc(line_and_arc):
    sketch, line_id, arc_id = line_and_arc

    result = TangentConstraintCommand.identify_entities(
        sketch.registry, [line_id, arc_id]
    )

    assert result is not None
    assert isinstance(result, TangentConstraintParams)
    assert result.line_id == line_id
    assert result.shape_id == arc_id


def test_identify_entities_line_and_circle(line_and_circle):
    sketch, line_id, circle_id = line_and_circle

    result = TangentConstraintCommand.identify_entities(
        sketch.registry, [line_id, circle_id]
    )

    assert result is not None
    assert isinstance(result, TangentConstraintParams)
    assert result.line_id == line_id
    assert result.shape_id == circle_id


def test_identify_entities_arc_and_line(line_and_arc):
    sketch, line_id, arc_id = line_and_arc

    result = TangentConstraintCommand.identify_entities(
        sketch.registry, [arc_id, line_id]
    )

    assert result is not None
    assert result.line_id == line_id
    assert result.shape_id == arc_id


def test_identify_entities_no_line(sketch):
    center = sketch.add_point(50, 50)
    radius = sketch.add_point(60, 50)
    circle_id = sketch.add_circle(center, radius)

    result = TangentConstraintCommand.identify_entities(
        sketch.registry, [circle_id]
    )

    assert result is None


def test_identify_entities_no_shape(sketch):
    p1 = sketch.add_point(0, 0)
    p2 = sketch.add_point(100, 0)
    line_id = sketch.add_line(p1, p2)

    result = TangentConstraintCommand.identify_entities(
        sketch.registry, [line_id]
    )

    assert result is None


def test_identify_entities_two_lines(sketch):
    p1 = sketch.add_point(0, 0)
    p2 = sketch.add_point(100, 0)
    line1_id = sketch.add_line(p1, p2)

    p3 = sketch.add_point(0, 50)
    p4 = sketch.add_point(100, 50)
    line2_id = sketch.add_line(p3, p4)

    result = TangentConstraintCommand.identify_entities(
        sketch.registry, [line1_id, line2_id]
    )

    assert result is None


def test_identify_entities_two_circles(sketch):
    center1 = sketch.add_point(50, 50)
    radius1 = sketch.add_point(60, 50)
    circle1_id = sketch.add_circle(center1, radius1)

    center2 = sketch.add_point(150, 50)
    radius2 = sketch.add_point(160, 50)
    circle2_id = sketch.add_circle(center2, radius2)

    result = TangentConstraintCommand.identify_entities(
        sketch.registry, [circle1_id, circle2_id]
    )

    assert result is None


def test_identify_entities_empty_selection(sketch):
    result = TangentConstraintCommand.identify_entities(sketch.registry, [])

    assert result is None


def test_identify_entities_multiple_shapes_uses_first(sketch):
    p1 = sketch.add_point(0, 0)
    p2 = sketch.add_point(100, 0)
    line_id = sketch.add_line(p1, p2)

    center1 = sketch.add_point(50, 50)
    radius1 = sketch.add_point(60, 50)
    circle_id = sketch.add_circle(center1, radius1)

    center2 = sketch.add_point(150, 50)
    radius2 = sketch.add_point(160, 50)
    arc_id = sketch.add_arc(center2, radius2, 0, 90)

    result = TangentConstraintCommand.identify_entities(
        sketch.registry, [line_id, circle_id, arc_id]
    )

    assert result is not None
    assert result.line_id == line_id
    assert result.shape_id in (circle_id, arc_id)
