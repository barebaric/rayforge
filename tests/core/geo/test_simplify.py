import numpy as np
from rayforge.core.geo.simplify import (
    simplify_points,
    simplify_points_to_array,
)


def test_simplify_straight_line():
    """Tests that collinear points on a straight line are removed."""
    points = [(0, 0), (1, 1), (2, 2), (3, 3), (10, 10)]

    result = simplify_points(points, tolerance=0.001)
    assert len(result) == 2
    assert result[0] == (0, 0)
    assert result[1] == (10, 10)


def test_simplify_straight_line_array():
    """Tests array-based simplification on a straight line."""
    points = np.array([(0, 0), (1, 1), (2, 2), (3, 3), (10, 10)])

    result = simplify_points_to_array(points, tolerance=0.001)
    assert len(result) == 2
    np.testing.assert_array_equal(result[0], (0, 0))
    np.testing.assert_array_equal(result[1], (10, 10))


def test_simplify_significant_corner():
    """Tests that points forming a corner > tolerance are kept."""
    points = [(0, 0), (5, 5), (10, 0)]

    result = simplify_points(points, tolerance=1.0)
    assert len(result) == 3
    assert result[1] == (5, 5)


def test_simplify_insignificant_bump():
    """Tests that a small bump within tolerance is removed."""
    points = [(0, 0), (5, 0.1), (10, 0)]

    result = simplify_points(points, tolerance=0.5)
    assert len(result) == 2
    assert result[1] == (10, 0)


def test_simplify_zigzag_removal():
    """Tests removal of high frequency zigzag noise within tolerance."""
    points: list[tuple[float, float]] = [(0, 0)]
    for x in range(1, 10):
        y = 0.05 if x % 2 else -0.05
        points.append((float(x), y))
    points.append((10.0, 0.0))

    result = simplify_points(points, tolerance=0.1)
    assert len(result) == 2
    assert result[0] == (0, 0)
    assert result[1] == (10, 0)


def test_simplify_duplicate_points():
    """Tests that consecutive duplicate points are removed/handled."""
    points = [(0, 0), (0, 0), (10, 10), (10, 10)]

    result = simplify_points(points, tolerance=0.001)
    assert len(result) == 2
    assert result[0] == (0, 0)
    assert result[1] == (10, 10)


def test_simplify_empty_points():
    """Tests that an empty point list is handled gracefully."""
    result = simplify_points([], tolerance=0.1)
    assert result == []


def test_simplify_single_segment():
    """Tests that a single segment (2 points) is not reduced."""
    points = [(0, 0), (10, 10)]

    result = simplify_points(points, tolerance=100.0)
    assert len(result) == 2
    assert result[0] == (0, 0)
    assert result[1] == (10, 10)


def test_simplify_two_points_array():
    """Tests array-based simplification with only 2 points."""
    points = np.array([(0, 0), (10, 10)])

    result = simplify_points_to_array(points, tolerance=100.0)
    assert len(result) == 2
    np.testing.assert_array_equal(result[0], (0, 0))
    np.testing.assert_array_equal(result[1], (10, 10))


def test_simplify_three_points_all_kept():
    """Tests that all 3 points are kept when deviation > tolerance."""
    points = [(0, 0), (5, 5), (10, 0)]

    result = simplify_points(points, tolerance=0.1)
    assert len(result) == 3
    assert result[0] == (0, 0)
    assert result[1] == (5, 5)
    assert result[2] == (10, 0)


def test_simplify_three_points_middle_removed():
    """Tests that middle point is removed when deviation < tolerance."""
    points = [(0, 0), (5, 0.01), (10, 0)]

    result = simplify_points(points, tolerance=0.1)
    assert len(result) == 2
    assert result[0] == (0, 0)
    assert result[1] == (10, 0)


def test_simplify_complex_shape():
    """Tests simplification on a more complex point sequence."""
    points = [
        (0, 0),
        (1, 0.1),
        (2, -0.1),
        (3, 0.05),
        (4, -0.05),
        (5, 5),
        (6, 5.1),
        (7, 4.9),
        (10, 0),
    ]

    result = simplify_points(points, tolerance=0.5)
    assert len(result) == 6
    assert result[0] == (0, 0)
    assert result[1] == (4, -0.05)
    assert result[2] == (5, 5)
    assert result[3] == (6, 5.1)
    assert result[4] == (7, 4.9)
    assert result[5] == (10, 0)


def test_simplify_array_with_z_axis():
    """Tests that Z coordinates are preserved when using array input."""
    points = np.array([(0, 0, 1), (5, 0, 2), (10, 0, 3)])

    result = simplify_points_to_array(points, tolerance=0.1)
    assert len(result) == 2
    np.testing.assert_array_equal(result[0], (0, 0, 1))
    np.testing.assert_array_equal(result[1], (10, 0, 3))


def test_simplify_zero_tolerance():
    """Tests that zero tolerance keeps all points."""
    points = [(0, 0), (5, 5), (10, 0)]

    result = simplify_points(points, tolerance=0.0)
    assert len(result) == 3


def test_simplify_negative_tolerance():
    """Tests that negative tolerance is treated as zero."""
    points = [(0, 0), (5, 5), (10, 0)]

    result = simplify_points(points, tolerance=-1.0)
    assert len(result) == 3


def test_simplify_large_tolerance():
    """Tests that very large tolerance reduces to endpoints only."""
    points = [(0, 0), (1, 1), (2, 2), (3, 3), (10, 10)]

    result = simplify_points(points, tolerance=1000.0)
    assert len(result) == 2
    assert result[0] == (0, 0)
    assert result[1] == (10, 10)


def test_simplify_vertical_line():
    """Tests simplification on a vertical line."""
    points = [(0, 0), (0, 1), (0, 2), (0, 3), (0, 10)]

    result = simplify_points(points, tolerance=0.001)
    assert len(result) == 2
    assert result[0] == (0, 0)
    assert result[1] == (0, 10)
