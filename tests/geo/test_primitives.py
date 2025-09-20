import pytest

from rayforge.core.geo.primitives import (
    is_point_in_polygon,
    line_segment_intersection,
    get_segment_region_intersections,
)


@pytest.fixture
def square_polygon():
    return [(0, 0), (10, 0), (10, 10), (0, 10)]


def test_is_point_in_polygon(square_polygon):
    # Points inside
    assert is_point_in_polygon((5, 5), square_polygon) is True
    assert is_point_in_polygon((0.1, 0.1), square_polygon) is True

    # Points outside
    assert is_point_in_polygon((15, 5), square_polygon) is False
    assert is_point_in_polygon((-5, 5), square_polygon) is False
    assert is_point_in_polygon((5, 15), square_polygon) is False
    assert is_point_in_polygon((5, -5), square_polygon) is False

    # Points on edge should be considered inside
    assert is_point_in_polygon((5, 0), square_polygon) is True  # Bottom edge
    assert is_point_in_polygon((10, 5), square_polygon) is True  # Right edge
    assert is_point_in_polygon((5, 10), square_polygon) is True  # Top edge
    assert is_point_in_polygon((0, 5), square_polygon) is True  # Left edge
    assert is_point_in_polygon((0, 0), square_polygon) is True  # Corner
    assert is_point_in_polygon((10, 10), square_polygon) is True  # Corner


def test_line_segment_intersection():
    # Crossing lines
    p1, p2 = (0, 0), (10, 10)
    p3, p4 = (0, 10), (10, 0)
    assert line_segment_intersection(p1, p2, p3, p4) == pytest.approx((5, 5))

    # T-junction (endpoint on segment)
    p1, p2 = (0, 0), (10, 0)
    p3, p4 = (5, -5), (5, 5)
    assert line_segment_intersection(p1, p2, p3, p4) == pytest.approx((5, 0))

    # No intersection (parallel)
    p1, p2 = (0, 0), (10, 0)
    p3, p4 = (0, 5), (10, 5)
    assert line_segment_intersection(p1, p2, p3, p4) is None

    # No intersection (not parallel, but segments don't meet)
    p1, p2 = (0, 0), (1, 1)
    p3, p4 = (0, 10), (1, 9)
    assert line_segment_intersection(p1, p2, p3, p4) is None

    # Collinear, overlapping
    p1, p2 = (0, 0), (5, 0)
    p3, p4 = (3, 0), (8, 0)
    # Our simple implementation returns None for collinear cases.
    assert line_segment_intersection(p1, p2, p3, p4) is None


def test_get_segment_region_intersections():
    p1 = (0.0, 50.0)
    p2 = (100.0, 50.0)
    region = [(40.0, 45.0), (60.0, 45.0), (60.0, 55.0), (40.0, 55.0)]

    # Test a simple crossing
    intersections = get_segment_region_intersections(p1, p2, [region])
    # Should find intersections at 40% and 60% of the line
    assert intersections == pytest.approx([0.0, 0.4, 0.6, 1.0])

    # Test a line fully outside
    p_out1 = (-20, 0)
    p_out2 = (-10, 0)
    intersections = get_segment_region_intersections(p_out1, p_out2, [region])
    # Should only return the start and end points
    assert intersections == pytest.approx([0.0, 1.0])
