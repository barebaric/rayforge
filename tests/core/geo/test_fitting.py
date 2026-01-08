import pytest
import numpy as np

from rayforge.core.geo.fitting import (
    are_collinear,
    fit_circle_to_points,
    get_arc_to_polyline_deviation,
    convert_arc_to_beziers_from_array,
)
from rayforge.core.geo.constants import (
    CMD_TYPE_BEZIER,
    COL_X,
    COL_Z,
    COL_C1X,
    COL_C1Y,
    COL_C2X,
    COL_C2Y,
    COL_TYPE,
)


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


# --- Tests for convert_arc_to_beziers_from_array ---


def test_convert_arc_90_degree_ccw():
    """Tests a 90-degree CCW arc conversion to a single Bezier."""
    start = (10.0, 0.0, 0.0)
    end = (0.0, 10.0, 5.0)
    center_offset = (-10.0, 0.0)
    clockwise = False

    beziers = convert_arc_to_beziers_from_array(
        start, end, center_offset, clockwise
    )

    assert len(beziers) == 1
    b = beziers[0]

    # Check command type and endpoint
    assert b[COL_TYPE] == CMD_TYPE_BEZIER
    np.testing.assert_allclose(b[COL_X : COL_Z + 1], end, atol=1e-9)

    # Check control points
    # kappa for 90 deg = 4/3 * tan(pi/8) approx 0.5522847
    expected_c1 = (10.0, 5.522847)
    expected_c2 = (5.522847, 10.0)
    np.testing.assert_allclose(
        (b[COL_C1X], b[COL_C1Y]), expected_c1, rtol=1e-6
    )
    np.testing.assert_allclose(
        (b[COL_C2X], b[COL_C2Y]), expected_c2, rtol=1e-6
    )


def test_convert_arc_180_degree_cw():
    """Tests a 180-degree CW arc, expecting two Bezier segments."""
    start = (10.0, 0.0, 0.0)
    end = (-10.0, 0.0, 10.0)
    center_offset = (-10.0, 0.0)
    clockwise = True

    beziers = convert_arc_to_beziers_from_array(
        start, end, center_offset, clockwise
    )

    assert len(beziers) == 2

    # Check first segment (10,0) -> (0,-10) CW
    b1 = beziers[0]
    midpoint = (0.0, -10.0, 5.0)
    assert b1[COL_TYPE] == CMD_TYPE_BEZIER
    np.testing.assert_allclose(b1[COL_X : COL_Z + 1], midpoint, atol=1e-9)

    # Check second segment (0,-10) -> (-10,0) CW
    b2 = beziers[1]
    assert b2[COL_TYPE] == CMD_TYPE_BEZIER
    np.testing.assert_allclose(b2[COL_X : COL_Z + 1], end, atol=1e-9)

    # Check a control point on the second segment
    # Start of seg2 is (0, -10), end is (-10, 0)
    # kappa ~ 0.5522847
    # C1 should be (0,-10) + 10 * kappa * (-1, 0) = (-5.522847, -10)
    expected_c1_seg2 = (-5.522847, -10.0)
    np.testing.assert_allclose(
        (b2[COL_C1X], b2[COL_C1Y]), expected_c1_seg2, rtol=1e-6
    )


def test_convert_arc_full_circle_ccw():
    """Tests a 360-degree CCW arc, expecting four segments."""
    start = (5.0, 0.0, 2.0)
    # For a full circle, start and end points must be identical.
    end = (5.0, 0.0, 2.0)
    center_offset = (-5.0, 0.0)
    clockwise = False

    beziers = convert_arc_to_beziers_from_array(
        start, end, center_offset, clockwise
    )

    assert len(beziers) == 4

    # Check endpoints of the segments are correct on the circle
    ep1 = (0.0, 5.0, 2.0)
    ep2 = (-5.0, 0.0, 2.0)
    ep3 = (0.0, -5.0, 2.0)

    np.testing.assert_allclose(beziers[0][COL_X : COL_Z + 1], ep1, atol=1e-6)
    np.testing.assert_allclose(beziers[1][COL_X : COL_Z + 1], ep2, atol=1e-6)
    np.testing.assert_allclose(beziers[2][COL_X : COL_Z + 1], ep3, atol=1e-6)
    # Final endpoint should match original end point
    np.testing.assert_allclose(beziers[3][COL_X : COL_Z + 1], end, atol=1e-6)


def test_convert_arc_zero_length():
    """Tests that a zero-length arc produces no beziers."""
    # Case 1: Zero radius (center_offset is zero)
    start = (1.0, 1.0, 1.0)
    end = (2.0, 2.0, 2.0)
    center_offset_zero = (0.0, 0.0)
    beziers_1 = convert_arc_to_beziers_from_array(
        start, end, center_offset_zero, False
    )
    assert len(beziers_1) == 0

    # Case 2: Non-zero radius, but near-zero sweep angle
    start_2 = (1.0, 0.0, 1.0)
    end_2 = (1.0, 1e-9, 1.0)
    center_offset_2 = (-1.0, 0.0)  # center is (0,0)
    beziers_2 = convert_arc_to_beziers_from_array(
        start_2, end_2, center_offset_2, False
    )
    assert len(beziers_2) == 0


def test_convert_arc_spiral_graceful_handling():
    """Tests that an arc with non-constant radius (spiral) is handled."""
    start = (10.0, 0.0, 0.0)
    end = (0.0, 5.0, 1.0)  # End radius is 5, start is 10
    center_offset = (-10.0, 0.0)
    clockwise = False

    beziers = convert_arc_to_beziers_from_array(
        start, end, center_offset, clockwise
    )

    assert len(beziers) == 1
    b = beziers[0]

    assert b[COL_TYPE] == CMD_TYPE_BEZIER
    np.testing.assert_allclose(b[COL_X : COL_Z + 1], end, atol=1e-9)

    # kappa for 90 deg ~ 0.5522847
    # C1 is based on start radius (10.0)
    # C2 is based on end radius (5.0)
    expected_c1 = (10.0, 10.0 * 0.5522847)
    expected_c2 = (5.0 * 0.5522847, 5.0)

    np.testing.assert_allclose(
        (b[COL_C1X], b[COL_C1Y]), expected_c1, rtol=1e-6
    )
    np.testing.assert_allclose(
        (b[COL_C2X], b[COL_C2Y]), expected_c2, rtol=1e-6
    )
