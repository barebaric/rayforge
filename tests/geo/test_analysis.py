import pytest
import math
import numpy as np
from rayforge.core.geo import Geometry
from rayforge.core.geo.analysis import (
    get_path_winding_order,
    get_point_and_tangent_at,
    get_outward_normal_at,
    get_angle_at_vertex,
    are_collinear,
    fit_circle_to_points,
    get_arc_to_polyline_deviation,
    remove_duplicates,
    is_clockwise,
    arc_direction_is_clockwise,
)


@pytest.fixture
def ccw_square_geometry():
    geo = Geometry()
    geo.move_to(0, 0)  # cmd 0
    geo.line_to(10, 0)  # cmd 1: bottom
    geo.line_to(10, 10)  # cmd 2: right
    geo.line_to(0, 10)  # cmd 3: top
    geo.close_path()  # cmd 4: left
    return geo


@pytest.fixture
def cw_square_geometry():
    geo = Geometry()
    geo.move_to(0, 0)  # cmd 0
    geo.line_to(0, 10)  # cmd 1: left
    geo.line_to(10, 10)  # cmd 2: top
    geo.line_to(10, 0)  # cmd 3: right
    geo.close_path()  # cmd 4: bottom
    return geo


def test_get_winding_order(ccw_square_geometry, cw_square_geometry):
    # Test CCW
    assert get_path_winding_order(ccw_square_geometry.commands, 1) == "ccw"
    assert get_path_winding_order(ccw_square_geometry.commands, 3) == "ccw"

    # Test CW
    assert get_path_winding_order(cw_square_geometry.commands, 1) == "cw"
    assert get_path_winding_order(cw_square_geometry.commands, 3) == "cw"

    # Test open path
    open_geo = Geometry()
    open_geo.move_to(0, 0)
    open_geo.line_to(10, 10)
    assert get_path_winding_order(open_geo.commands, 1) == "unknown"


def test_get_point_and_tangent_at():
    geo = Geometry()
    geo.move_to(0, 0)
    geo.line_to(10, 0)  # cmd 1

    # Test horizontal line
    result = get_point_and_tangent_at(geo.commands, 1, 0.5)
    assert result is not None
    pt, tan = result
    assert pt == pytest.approx((5, 0))
    assert tan == pytest.approx((1, 0))

    geo.line_to(10, 10)  # cmd 2
    # Test vertical line
    result = get_point_and_tangent_at(geo.commands, 2, 0.25)
    assert result is not None
    pt, tan = result
    assert pt == pytest.approx((10, 2.5))
    assert tan == pytest.approx((0, 1))

    # Test arc (CCW 90 degree from (10,10) to (0,10))
    # Start: (10,10). Center offset: (-10,0). Center: (0,10). Radius: 10.
    geo.arc_to(0, 10, i=-10, j=0, clockwise=False)  # cmd 3
    # Start of arc
    result = get_point_and_tangent_at(geo.commands, 3, 0.0)
    assert result is not None
    pt, tan = result
    assert pt == pytest.approx((10, 10))
    assert tan == pytest.approx((0, 1))  # Tangent is vertical up

    # Midpoint of arc
    result = get_point_and_tangent_at(geo.commands, 3, 0.5)
    assert result is not None
    pt, tan = result
    # This arc is a spiral from (10,10) to its center (0,10), because the
    # end radius is 0. The angle is constant (0 rads).
    # At t=0.5, the radius is half the starting radius (5).
    # Point is (center_x + r*cos(0), center_y + r*sin(0))
    #       -> (0+5, 10+0) -> (5,10)
    assert pt == pytest.approx((5, 10))
    # Tangent for a spiral towards the center should be perpendicular to the
    # radius vector from the center. Radius vec is (5,0), so tangent is (0,5).
    assert tan == pytest.approx((0, 1))


def test_get_outward_normal_at(ccw_square_geometry, cw_square_geometry):
    # Test CCW square
    # Bottom edge, tangent (1,0) -> outward normal (0,-1)
    normal = get_outward_normal_at(ccw_square_geometry.commands, 1, 0.5)
    assert normal is not None
    assert normal == pytest.approx((0, -1))
    # Right edge, tangent (0,1) -> outward normal (1,0)
    normal = get_outward_normal_at(ccw_square_geometry.commands, 2, 0.5)
    assert normal is not None
    assert normal == pytest.approx((1, 0))
    # Top edge, tangent (-1,0) -> outward normal (0,1)
    normal = get_outward_normal_at(ccw_square_geometry.commands, 3, 0.5)
    assert normal is not None
    assert normal == pytest.approx((0, 1))
    # Left edge, tangent (0,-1) -> outward normal (-1,0)
    normal = get_outward_normal_at(ccw_square_geometry.commands, 4, 0.5)
    assert normal is not None
    assert normal == pytest.approx((-1, 0))

    # Test CW square
    # Left edge, tangent (0,1) -> outward normal (-1,0)
    normal = get_outward_normal_at(cw_square_geometry.commands, 1, 0.5)
    assert normal is not None
    assert normal == pytest.approx((-1, 0))
    # Bottom edge, tangent (-1,0) -> outward normal (0,-1)
    normal = get_outward_normal_at(cw_square_geometry.commands, 4, 0.5)
    assert normal is not None
    assert normal == pytest.approx((0, -1))


def test_get_angle_at_vertex():
    # 90 degree corner
    p0, p1, p2 = (0.0, 10.0), (0.0, 0.0), (10.0, 0.0)
    assert get_angle_at_vertex(p0, p1, p2) == pytest.approx(math.pi / 2)

    # Straight line (180 degrees)
    p0, p1, p2 = (-10.0, 0.0), (0.0, 0.0), (10.0, 0.0)
    assert get_angle_at_vertex(p0, p1, p2) == pytest.approx(math.pi)

    # 45 degree corner
    p0, p1, p2 = (0.0, 10.0), (0.0, 0.0), (10.0, 10.0)
    assert get_angle_at_vertex(p0, p1, p2) == pytest.approx(math.pi / 4)

    # Coincident points
    p0, p1, p2 = (10.0, 10.0), (0.0, 0.0), (0.0, 0.0)
    assert get_angle_at_vertex(p0, p1, p2) == pytest.approx(math.pi)


def test_are_collinear():
    # Collinear points (horizontal)
    points = [(0.0, 0.0, 0.0), (5.0, 0.0, 0.0), (10.0, 0.0, 0.0)]
    assert are_collinear(points) is True

    # Collinear points (vertical)
    points = [(0.0, 0.0, 0.0), (0.0, 5.0, 0.0), (0.0, 10.0, 0.0)]
    assert are_collinear(points) is True

    # Non-collinear points
    points = [(0.0, 0.0, 0.0), (1.0, 1.0, 0.0), (2.0, 2.1, 0.0)]
    assert are_collinear(points) is False


def test_fit_circle_to_points_collinear_returns_none():
    """Test collinear points return None."""
    points = [(0.0, 0.0, 0.0), (2.0, 2.0, 0.0), (5.0, 5.0, 0.0)]
    assert fit_circle_to_points(points) is None


def test_fit_circle_to_points_perfect_circle():
    """Test perfect circle fitting."""
    center = (2.0, 3.0)
    radius = 5.0
    angles = np.linspace(0, 2 * np.pi, 20)
    points = [
        (
            center[0] + radius * np.cos(theta),
            center[1] + radius * np.sin(theta),
            0.0,
        )
        for theta in angles
    ]
    result = fit_circle_to_points(points)
    assert result is not None

    (xc, yc), r, error = result
    assert xc == pytest.approx(center[0], abs=1e-6)
    assert yc == pytest.approx(center[1], abs=1e-6)
    assert r == pytest.approx(radius, abs=1e-6)
    assert error < 1e-6


def test_fit_circle_to_points_noisy_circle():
    """Test circle fitting with noisy points."""
    center = (-1.0, 4.0)
    radius = 3.0
    np.random.seed(42)  # For reproducibility
    angles = np.linspace(0, 2 * np.pi, 30)
    noise = np.random.normal(scale=0.1, size=(len(angles), 2))

    points = [
        (
            center[0] + radius * np.cos(theta) + dx,
            center[1] + radius * np.sin(theta) + dy,
            0.0,
        )
        for (theta, (dx, dy)) in zip(angles, noise)
    ]
    result = fit_circle_to_points(points)
    assert result is not None

    (xc, yc), r, error = result
    assert xc == pytest.approx(center[0], abs=0.15)
    assert yc == pytest.approx(center[1], abs=0.15)
    assert r == pytest.approx(radius, abs=0.15)
    assert error < 0.2


def test_fit_circle_to_points_insufficient_points():
    """Test 1-2 points or duplicates return None."""
    assert fit_circle_to_points([(0.0, 0.0, 0.0)]) is None
    assert fit_circle_to_points([(1.0, 2.0, 0.0), (3.0, 4.0, 0.0)]) is None
    assert (
        fit_circle_to_points(
            [(5.0, 5.0, 0.0), (5.0, 5.0, 0.0), (5.0, 5.0, 0.0)]
        )
        is None
    )


def test_fit_circle_to_points_small_radius():
    """Test small-radius circle fitting."""
    center = (0.0, 0.0)
    radius = 0.1
    angles = np.linspace(0, 2 * np.pi, 10)
    points = [
        (
            center[0] + radius * np.cos(theta),
            center[1] + radius * np.sin(theta),
            0.0,
        )
        for theta in angles
    ]
    result = fit_circle_to_points(points)
    assert result is not None
    (xc, yc), r, error = result
    assert r == pytest.approx(radius, rel=0.01)


def test_fit_circle_to_points_semicircle_accuracy():
    """
    Verify fit_circle() returns correct parameters for a perfect semicircle.
    """
    center = (5.0, 0.0)
    radius = 10.0
    angles = np.linspace(0, np.pi, 20)
    points = [
        (
            center[0] + radius * np.cos(theta),
            center[1] + radius * np.sin(theta),
            0.0,
        )
        for theta in angles
    ]
    result = fit_circle_to_points(points)
    assert result is not None
    (xc, yc), r, error = result
    assert np.isclose(xc, 5.0, atol=0.001)
    assert np.isclose(yc, 0.0, atol=0.001)
    assert np.isclose(r, 10.0, rtol=0.001)
    assert error < 1e-6


def test_get_arc_to_polyline_deviation_perfect_arc():
    """Test deviation for a perfect 90-degree arc."""
    center = (7.0, 3.0)
    radius = 5.0
    angles = np.linspace(np.pi / 2, np.pi, 10)
    points = [
        (center[0] + radius * np.cos(t), center[1] + radius * np.sin(t), 0.0)
        for t in angles
    ]
    deviation = get_arc_to_polyline_deviation(points, center, radius)
    assert deviation < 0.05, f"Deviation too large: {deviation}"


def test_get_arc_to_polyline_deviation_too_large():
    """Test deviation for a coarse 90-degree arc is correctly high."""
    center = (7.0, 3.0)
    radius = 5.0
    angles = np.linspace(np.pi / 2, np.pi, 5)  # Coarse sampling
    points = [
        (center[0] + radius * np.cos(t), center[1] + radius * np.sin(t), 0.0)
        for t in angles
    ]
    deviation = get_arc_to_polyline_deviation(points, center, radius)
    assert deviation > 0.05, f"Expected larger deviation: {deviation}"


def test_remove_duplicates():
    points = [(1.0, 1.0), (1.0, 1.0), (2.0, 2.0), (2.0, 2.0)]
    assert remove_duplicates(points) == [(1.0, 1.0), (2.0, 2.0)]


def test_is_clockwise():
    # Clockwise points (right half-circle)
    points = [(0.0, 0.0, 0.0), (1.0, 1.0, 0.0), (2.0, 0.0, 0.0)]
    assert is_clockwise(points) is True

    # Counter-clockwise points (left half-circle)
    points = [(0.0, 0.0, 0.0), (-1.0, 1.0, 0.0), (-2.0, 0.0, 0.0)]
    assert is_clockwise(points) is False


def test_arc_direction_is_clockwise_half_circle():
    """Test a semicircle moving clockwise."""
    center = (0.0, 0.0)
    points = [
        (1.0, 0.0, 0.0),
        (0.0, -1.0, 0.0),
        (-1.0, 0.0, 0.0),
    ]
    assert arc_direction_is_clockwise(points, center) is True


def test_arc_direction_is_counter_clockwise_half_circle():
    """Test a semicircle moving counter-clockwise."""
    center = (0.0, 0.0)
    points = [
        (1.0, 0.0, 0.0),
        (0.0, 1.0, 0.0),
        (-1.0, 0.0, 0.0),
    ]
    assert arc_direction_is_clockwise(points, center) is False


def test_arc_direction_is_clockwise_full_circle():
    """Test a full clockwise circle."""
    center = (0.0, 0.0)
    points = [
        (1.0, 0.0, 0.0),
        (0.0, -1.0, 0.0),
        (-1.0, 0.0, 0.0),
        (0.0, 1.0, 0.0),
        (1.0, 0.0, 0.0),
    ]
    assert arc_direction_is_clockwise(points, center) is True


def test_arc_direction_is_counter_clockwise_full_circle():
    """Test a full counter-clockwise circle."""
    center = (0.0, 0.0)
    points = [
        (1.0, 0.0, 0.0),
        (0.0, 1.0, 0.0),
        (-1.0, 0.0, 0.0),
        (0.0, -1.0, 0.0),
        (1.0, 0.0, 0.0),
    ]
    assert arc_direction_is_clockwise(points, center) is False


def test_arc_direction_is_minimal_clockwise_arc():
    """Test a minimal 3-point clockwise arc."""
    center = (0.0, 0.0)
    points = [
        (2.0, 0.0, 0.0),
        (1.0, -1.0, 0.0),
        (0.0, 0.0, 0.0),
    ]
    assert arc_direction_is_clockwise(points, center) is True


def test_arc_direction_is_minimal_counter_clockwise_arc():
    """Test a minimal 3-point counter-clockwise arc."""
    center = (0.0, 0.0)
    points = [
        (2.0, 0.0, 0.0),
        (1.0, 1.0, 0.0),
        (0.0, 0.0, 0.0),
    ]
    assert arc_direction_is_clockwise(points, center) is False


def test_arc_direction_is_crossing_angle_discontinuity_counter_clockwise():
    """Test an arc crossing the π/-π discontinuity (counter-clockwise)."""
    center = (0.0, 0.0)
    points = [
        (1.0, 0.1, 0.0),
        (0.0, 1.0, 0.0),
        (-1.0, 0.1, 0.0),
    ]
    assert arc_direction_is_clockwise(points, center) is False


def test_arc_direction_is_small_radius_arc():
    """Test a small-radius clockwise arc."""
    center = (1.0, 1.0)
    points = [
        (1.1, 1.0, 0.0),
        (1.0, 0.9, 0.0),
        (0.9, 1.0, 0.0),
    ]
    assert arc_direction_is_clockwise(points, center) is True
