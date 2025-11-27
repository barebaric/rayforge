import pytest
import math

from rayforge.core.geo.primitives import (
    is_point_in_polygon,
    line_segment_intersection,
    get_segment_region_intersections,
    line_intersection,
    is_angle_between,
    get_arc_bounding_box,
)


@pytest.mark.parametrize(
    "target, start, end, clockwise, expected",
    [
        # Counter-Clockwise (CCW), No Wrap
        (math.pi / 2, math.pi / 4, 3 * math.pi / 4, False, True),  # Inside
        (0, math.pi / 4, 3 * math.pi / 4, False, False),  # Outside (before)
        (
            math.pi,
            math.pi / 4,
            3 * math.pi / 4,
            False,
            False,
        ),  # Outside (after)
        (
            math.pi / 4,
            math.pi / 4,
            3 * math.pi / 4,
            False,
            True,
        ),  # On start boundary
        (
            3 * math.pi / 4,
            math.pi / 4,
            3 * math.pi / 4,
            False,
            True,
        ),  # On end boundary
        # CCW, With Wrap (e.g., 315 deg to 45 deg)
        (0, 7 * math.pi / 4, math.pi / 4, False, True),  # Inside (on axis)
        (
            2 * math.pi - 0.1,
            7 * math.pi / 4,
            math.pi / 4,
            False,
            True,
        ),  # Inside
        (math.pi, 7 * math.pi / 4, math.pi / 4, False, False),  # Outside
        (
            7 * math.pi / 4,
            7 * math.pi / 4,
            math.pi / 4,
            False,
            True,
        ),  # On start
        (math.pi / 4, 7 * math.pi / 4, math.pi / 4, False, True),  # On end
        # Clockwise (CW), No Wrap
        (math.pi / 2, 3 * math.pi / 4, math.pi / 4, True, True),  # Inside
        (
            math.pi,
            3 * math.pi / 4,
            math.pi / 4,
            True,
            False,
        ),  # Outside (before)
        (0, 3 * math.pi / 4, math.pi / 4, True, False),  # Outside (after)
        (
            3 * math.pi / 4,
            3 * math.pi / 4,
            math.pi / 4,
            True,
            True,
        ),  # On start
        (math.pi / 4, 3 * math.pi / 4, math.pi / 4, True, True),  # On end
        # CW, With Wrap (e.g., 45 deg to 315 deg)
        (0, math.pi / 4, 7 * math.pi / 4, True, True),  # Inside (on axis)
        (math.pi, math.pi / 4, 7 * math.pi / 4, True, False),  # Outside
        (math.pi / 4, math.pi / 4, 7 * math.pi / 4, True, True),  # On start
        (7 * math.pi / 4, math.pi / 4, 7 * math.pi / 4, True, True),  # On end
        # Angle Normalization Tests (should behave identically to above)
        (math.pi / 2, math.pi / 4 + 2 * math.pi, 3 * math.pi / 4, False, True),
        (math.pi / 2, math.pi / 4, 3 * math.pi / 4 - 4 * math.pi, False, True),
        (math.pi / 2 + 6 * math.pi, math.pi / 4, 3 * math.pi / 4, False, True),
        # Zero-length arc
        (math.pi / 4, math.pi / 4, math.pi / 4, False, True),  # On point
        (math.pi / 2, math.pi / 4, math.pi / 4, False, False),  # Off point
        (math.pi / 4, math.pi / 4, math.pi / 4, True, True),  # On point
        (math.pi / 2, math.pi / 4, math.pi / 4, True, False),  # Off point
    ],
)
def test_is_angle_between(target, start, end, clockwise, expected):
    assert is_angle_between(target, start, end, clockwise) is expected


def get_arc_params(center, radius, start_angle, end_angle):
    """Helper to generate arc parameters for testing."""
    start_pos = (
        center[0] + radius * math.cos(start_angle),
        center[1] + radius * math.sin(start_angle),
    )
    end_pos = (
        center[0] + radius * math.cos(end_angle),
        center[1] + radius * math.sin(end_angle),
    )
    center_offset = (center[0] - start_pos[0], center[1] - start_pos[1])
    return start_pos, end_pos, center_offset


def test_get_arc_bounding_box_within_quadrant():
    """Test arc that does not cross any cardinal axes."""
    center, radius = (100, 200), 50
    start_angle, end_angle = math.pi / 6, math.pi / 3  # Both in Q1
    start_pos, end_pos, center_offset = get_arc_params(
        center, radius, start_angle, end_angle
    )

    # CCW
    bbox = get_arc_bounding_box(start_pos, end_pos, center_offset, False)
    expected_bbox = (
        min(start_pos[0], end_pos[0]),
        min(start_pos[1], end_pos[1]),
        max(start_pos[0], end_pos[0]),
        max(start_pos[1], end_pos[1]),
    )
    assert bbox == pytest.approx(expected_bbox)

    # CW (swap start and end)
    bbox_cw = get_arc_bounding_box(end_pos, start_pos, center_offset, True)
    assert bbox_cw == pytest.approx(expected_bbox)


def test_get_arc_bounding_box_crosses_east():
    """Test arc crossing the 0 radian axis."""
    center, radius = (100, 200), 50
    start_angle, end_angle = -math.pi / 4, math.pi / 4
    start_pos, end_pos, center_offset = get_arc_params(
        center, radius, start_angle, end_angle
    )

    bbox = get_arc_bounding_box(start_pos, end_pos, center_offset, False)
    expected_max_x = center[0] + radius
    assert bbox[2] == pytest.approx(expected_max_x)
    assert bbox[0] == pytest.approx(min(start_pos[0], end_pos[0]))
    assert bbox[1] == pytest.approx(min(start_pos[1], end_pos[1]))
    assert bbox[3] == pytest.approx(max(start_pos[1], end_pos[1]))


def test_get_arc_bounding_box_crosses_north_y_up():
    """Test arc crossing the PI/2 radian axis."""
    center, radius = (100, 200), 50
    start_angle, end_angle = math.pi / 4, 3 * math.pi / 4
    start_pos, end_pos, center_offset = get_arc_params(
        center, radius, start_angle, end_angle
    )

    bbox = get_arc_bounding_box(start_pos, end_pos, center_offset, False)
    expected_max_y = center[1] + radius
    assert bbox[3] == pytest.approx(expected_max_y)
    assert bbox[0] == pytest.approx(min(start_pos[0], end_pos[0]))
    assert bbox[1] == pytest.approx(min(start_pos[1], end_pos[1]))
    assert bbox[2] == pytest.approx(max(start_pos[0], end_pos[0]))


def test_get_arc_bounding_box_crosses_west():
    """Test arc crossing the PI radian axis."""
    center, radius = (100, 200), 50
    start_angle, end_angle = 3 * math.pi / 4, 5 * math.pi / 4
    start_pos, end_pos, center_offset = get_arc_params(
        center, radius, start_angle, end_angle
    )

    bbox = get_arc_bounding_box(start_pos, end_pos, center_offset, False)
    expected_min_x = center[0] - radius
    assert bbox[0] == pytest.approx(expected_min_x)
    assert bbox[1] == pytest.approx(min(start_pos[1], end_pos[1]))
    assert bbox[2] == pytest.approx(max(start_pos[0], end_pos[0]))
    assert bbox[3] == pytest.approx(max(start_pos[1], end_pos[1]))


def test_get_arc_bounding_box_crosses_south_y_up():
    """Test arc crossing the 3*PI/2 radian axis."""
    center, radius = (100, 200), 50
    start_angle, end_angle = 5 * math.pi / 4, 7 * math.pi / 4
    start_pos, end_pos, center_offset = get_arc_params(
        center, radius, start_angle, end_angle
    )

    bbox = get_arc_bounding_box(start_pos, end_pos, center_offset, False)
    expected_min_y = center[1] - radius
    assert bbox[1] == pytest.approx(expected_min_y)
    assert bbox[0] == pytest.approx(min(start_pos[0], end_pos[0]))
    assert bbox[2] == pytest.approx(max(start_pos[0], end_pos[0]))
    assert bbox[3] == pytest.approx(max(start_pos[1], end_pos[1]))


def test_get_arc_bounding_box_semicircle():
    """Test a 180-degree arc."""
    center, radius = (100, 200), 50
    start_angle, end_angle = math.pi, 0
    start_pos, end_pos, center_offset = get_arc_params(
        center, radius, start_angle, end_angle
    )

    # A CCW arc from pi to 0 traces the BOTTOM semicircle, crossing 3*pi/2.
    bbox_ccw = get_arc_bounding_box(start_pos, end_pos, center_offset, False)
    expected_bbox_bottom = (
        center[0] - radius,
        center[1] - radius,
        center[0] + radius,
        center[1],
    )
    assert bbox_ccw == pytest.approx(expected_bbox_bottom)

    # A CW arc from pi to 0 traces the TOP semicircle, crossing pi/2.
    bbox_cw = get_arc_bounding_box(start_pos, end_pos, center_offset, True)
    expected_bbox_top = (
        center[0] - radius,
        center[1],
        center[0] + radius,
        center[1] + radius,
    )
    assert bbox_cw == pytest.approx(expected_bbox_top)


def test_get_arc_bounding_box_large_arc_crossing_three_axes():
    """Test a large arc that spans multiple quadrants."""
    center, radius = (100, 200), 50
    # CCW from 45 degrees to 315 degrees
    start_angle, end_angle = math.pi / 4, 7 * math.pi / 4
    start_pos, end_pos, center_offset = get_arc_params(
        center, radius, start_angle, end_angle
    )

    # Crosses North (PI/2), West (PI), and South (3*PI/2)
    bbox = get_arc_bounding_box(start_pos, end_pos, center_offset, False)
    expected_bbox = (
        center[0] - radius,  # From crossing West
        center[1] - radius,  # From crossing South
        start_pos[0],  # Max x is from start/end points
        center[1] + radius,  # From crossing North
    )
    assert bbox == pytest.approx(expected_bbox)


def test_get_arc_bounding_box_zero_length_arc():
    """Test a zero-length arc."""
    center, radius = (100, 200), 50
    start_angle, end_angle = math.pi / 4, math.pi / 4
    start_pos, end_pos, center_offset = get_arc_params(
        center, radius, start_angle, end_angle
    )

    bbox = get_arc_bounding_box(start_pos, end_pos, center_offset, False)
    expected_bbox = (start_pos[0], start_pos[1], end_pos[0], end_pos[1])
    assert bbox == pytest.approx(expected_bbox)


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


def test_line_intersection():
    # Intersection
    p1, p2 = (0, 0), (10, 10)
    p3, p4 = (0, 10), (10, 0)
    assert line_intersection(p1, p2, p3, p4) == pytest.approx((5, 5))

    # Parallel lines
    p1, p2 = (0, 0), (10, 0)
    p3, p4 = (0, 1), (10, 1)
    assert line_intersection(p1, p2, p3, p4) is None

    # Intersection outside segments (infinite lines)
    p1, p2 = (0, 0), (1, 0)
    p3, p4 = (0, 1), (0, 2)
    # x-axis and y-axis intersect at 0,0
    assert line_intersection(p1, p2, p3, p4) == pytest.approx((0, 0))
