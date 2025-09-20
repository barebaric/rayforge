import pytest

from rayforge.core.geo.primitives import (
    is_point_in_polygon,
    line_segment_intersection,
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
