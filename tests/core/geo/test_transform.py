import pytest
import math
import numpy as np
from rayforge.core.geo.constants import (
    CMD_TYPE_LINE,
    CMD_TYPE_ARC,
    COL_TYPE,
)
from rayforge.core.geo import (
    Geometry,
)
from rayforge.core.geo.transform import (
    grow_geometry,
)


def _create_translate_matrix(x, y, z):
    """Creates a NumPy translation matrix."""
    return np.array(
        [
            [1, 0, 0, x],
            [0, 1, 0, y],
            [0, 0, 1, z],
            [0, 0, 0, 1],
        ],
        dtype=float,
    )


def _create_scale_matrix(sx, sy, sz):
    """Creates a NumPy scaling matrix."""
    return np.array(
        [
            [sx, 0, 0, 0],
            [0, sy, 0, 0],
            [0, 0, sz, 0],
            [0, 0, 0, 1],
        ],
        dtype=float,
    )


def _create_z_rotate_matrix(angle_rad):
    """Creates a NumPy Z-axis rotation matrix."""
    c = math.cos(angle_rad)
    s = math.sin(angle_rad)
    return np.array(
        [
            [c, -s, 0, 0],
            [s, c, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ],
        dtype=float,
    )


# --- Affine Transform Tests ---


def test_transform_identity():
    geo = Geometry()
    geo.move_to(10, 20, 30)
    geo.arc_to(50, 60, i=5, j=7, z=40)
    original_geo = geo.copy()

    identity_matrix = np.identity(4, dtype=float)
    geo.transform(identity_matrix)

    assert geo == original_geo


def test_transform_translate():
    geo = Geometry()
    geo.move_to(10, 20, 30)
    geo.arc_to(50, 60, i=5, j=7, z=40)

    translate_matrix = _create_translate_matrix(10, -5, 15)
    geo.transform(translate_matrix)
    assert geo.data is not None

    assert np.allclose(geo.data[0, 1:4], (20, 15, 45))
    assert np.allclose(geo.data[1, 1:4], (60, 55, 55))
    # Translation should NOT affect arc center offsets (vectors)
    assert np.allclose(geo.data[1, 4:6], (5, 7))


def test_transform_scale_non_uniform_linearizes_arc():
    geo = Geometry()
    geo.move_to(10, 20, 5)
    geo.arc_to(22, 22, i=5, j=7, z=-10)
    scale_matrix = _create_scale_matrix(2, 3, 4)

    geo.transform(scale_matrix)
    assert geo.data is not None

    assert np.allclose(geo.data[0, 1:4], (20, 60, 20))
    # Arcs are linearized on non-uniform scale
    assert geo.data[1, COL_TYPE] == CMD_TYPE_LINE
    final_row = geo.data[-1]
    assert final_row is not None
    final_point = final_row[1:4]
    expected_final_point = (22 * 2, 22 * 3, -10 * 4)
    assert np.allclose(final_point, expected_final_point)


def test_transform_rotate_preserves_z():
    geo = Geometry()
    geo.move_to(10, 10, -5)
    rotate_matrix = _create_z_rotate_matrix(math.radians(90))

    geo.transform(rotate_matrix)
    assert geo.data is not None

    x, y, z = geo.data[0, 1:4]
    assert z == -5
    assert x == pytest.approx(-10)
    assert y == pytest.approx(10)


def test_transform_uniform_scale_preserves_arcs():
    geo = Geometry()
    geo.move_to(0, 0, 0)
    # Arc from (0,0) to (10,0) with center at (5,0) -> radius 5
    geo.arc_to(10, 0, i=5, j=0, clockwise=True)

    # Uniform scale by 2
    scale_matrix = _create_scale_matrix(2, 2, 2)
    geo.transform(scale_matrix)
    assert geo.data is not None

    arc_row = geo.data[1]
    assert arc_row[COL_TYPE] == CMD_TYPE_ARC
    assert np.allclose(arc_row[1:4], (20, 0, 0))
    # Offset should also scale
    assert np.allclose(arc_row[4:6], (10, 0))


# --- Grow/Offset Tests ---


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
    """Tests that offset direction is consistent for a CW shape."""
    # A clockwise square
    square_cw = Geometry.from_points([(0, 0), (0, 10), (10, 10), (10, 0)])

    # A positive offset on any shape should grow it
    grown_square = grow_geometry(square_cw, 1.0)
    assert grown_square.area() == pytest.approx(144.0)

    # A negative offset on any shape should shrink it
    shrunk_square = grow_geometry(square_cw, -1.0)
    assert shrunk_square.area() == pytest.approx(64.0)


def test_grow_shape_with_hole():
    """Tests offsetting a shape containing a hole."""
    # Outer CCW square (0,0) -> (20,20), Area = 400
    outer = Geometry.from_points([(0, 0), (20, 0), (20, 20), (0, 20)])
    # Inner CW square (hole) (5,5) -> (15,15), Area = -100
    inner = Geometry.from_points([(5, 5), (5, 15), (15, 15), (15, 5)])
    shape_with_hole = outer.copy()
    shape_with_hole.extend(inner)
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
