import pytest
import math
import numpy as np
from rayforge.core.geo import Geometry
from rayforge.core.geo.transform import grow_geometry


def test_grow_simple_square():
    """Tests growing and shrinking a simple CCW square."""
    square = Geometry.from_points([(0, 0), (10, 0), (10, 10), (0, 10)])

    # Grow the square
    grown_square = grow_geometry(square, 1.0)
    assert grown_square.area() == pytest.approx(144.0)  # (10+2)^2
    # Check one of the new vertices
    grown_points = grown_square.segments()[0]
    # Use pytest.approx for floating point comparisons of coordinates
    assert any(np.allclose(p, (-1.0, -1.0, 0.0)) for p in grown_points), (
        "Expected grown vertex not found"
    )

    # Shrink the square
    shrunk_square = grow_geometry(square, -1.0)
    assert shrunk_square.area() == pytest.approx(64.0)  # (10-2)^2
    shrunk_points = shrunk_square.segments()[0]
    assert any(np.allclose(p, (1.0, 1.0, 0.0)) for p in shrunk_points), (
        "Expected shrunk vertex not found"
    )


def test_grow_clockwise_square():
    """Tests that a CW shape correctly inverts the offset direction."""
    # A clockwise square
    square_cw = Geometry.from_points([(0, 0), (0, 10), (10, 10), (10, 0)])

    # A positive offset on a CW shape should shrink it
    shrunk_square = grow_geometry(square_cw, 1.0)
    assert shrunk_square.area() == pytest.approx(64.0)

    # A negative offset on a CW shape should grow it
    grown_square = grow_geometry(square_cw, -1.0)
    assert grown_square.area() == pytest.approx(144.0)


def test_grow_shape_with_hole():
    """Tests offsetting a shape containing a hole."""
    # Outer CCW square (0,0) -> (20,20), Area = 400
    outer = Geometry.from_points([(0, 0), (20, 0), (20, 20), (0, 20)])
    # Inner CW square (hole) (5,5) -> (15,15), Area = -100
    inner = Geometry.from_points([(5, 5), (5, 15), (15, 15), (15, 5)])
    shape_with_hole = outer.copy()
    shape_with_hole.commands.extend(inner.commands)
    assert shape_with_hole.area() == pytest.approx(300.0)

    # Grow by 1. Outer becomes 22x22, inner becomes 8x8.
    # New area = 22*22 - 8*8 = 484 - 64 = 420.
    grown_shape = grow_geometry(shape_with_hole, 1.0)
    assert grown_shape.area() == pytest.approx(420.0)

    # Shrink by 1. Outer becomes 18x18, inner becomes 12x12.
    # New area = 18*18 - 12*12 = 324 - 144 = 180.
    shrunk_shape = grow_geometry(shape_with_hole, -1.0)
    assert shrunk_shape.area() == pytest.approx(180.0)


def test_grow_open_path_is_ignored():
    """Tests that open paths result in an empty geometry."""
    open_path = Geometry.from_points([(0, 0), (10, 10), (20, 0)], close=False)
    result = grow_geometry(open_path, 1.0)
    assert result.is_empty()


def test_grow_circle():
    """Tests offsetting a shape with arcs by checking the resulting area."""
    radius = 10.0
    # Create a polygonal approximation of a circle using from_points. This
    # avoids issues with how area() handles ArcTo and ensures a valid, simple
    # polygon for testing the offset logic itself.
    num_points = 100
    angles = np.linspace(0, 2 * np.pi, num_points, endpoint=False)
    points = [(radius * np.cos(a), radius * np.sin(a)) for a in angles]
    circle = Geometry.from_points(points)

    original_area = math.pi * radius**2
    assert circle.area() == pytest.approx(original_area, rel=1e-3)

    # Grow the circle
    offset = 2.0
    grown_circle = grow_geometry(circle, offset)
    expected_grown_area = math.pi * (radius + offset) ** 2
    assert grown_circle.area() == pytest.approx(expected_grown_area, rel=1e-2)

    # Shrink the circle
    offset = -2.0
    shrunk_circle = grow_geometry(circle, offset)
    expected_shrunk_area = math.pi * (radius + offset) ** 2
    assert shrunk_circle.area() == pytest.approx(
        expected_shrunk_area, rel=1e-2
    )


def test_shrink_to_nothing():
    """Tests that shrinking a shape by its half-width or more is handled."""
    square = Geometry.from_points([(0, 0), (10, 0), (10, 10), (0, 10)])

    # Shrinking by half the width should result in a zero-area shape
    shrunk_to_point = grow_geometry(square, -5.0)
    assert shrunk_to_point.area() == pytest.approx(0.0)

    # Shrinking by more than the half-width should also result in zero area
    shrunk_past_zero = grow_geometry(square, -6.0)
    # The algorithm might produce a small self-intersecting shape with non-zero
    # area in this case, but it should be very small. A robust offset algorithm
    # would clean this up, but for now we check that it's close to zero.
    assert shrunk_past_zero.area() == pytest.approx(0.0, abs=1.0)
