import pytest
import numpy as np

from rayforge.core.geo import Geometry
from rayforge.core.geo.fitting import (
    are_collinear,
    fit_circle_3_points,
    fit_circle_to_points,
    get_arc_to_polyline_deviation,
    convert_arc_to_beziers_from_array,
    fit_points_to_primitives,
    get_max_line_deviation,
    create_line_cmd,
    create_arc_cmd,
    fit_points_recursive,
    fit_arcs,
    optimize_path_from_array,
)
from rayforge.core.geo.constants import (
    CMD_TYPE_BEZIER,
    CMD_TYPE_LINE,
    CMD_TYPE_ARC,
    CMD_TYPE_MOVE,
    COL_X,
    COL_Y,
    COL_Z,
    COL_C1X,
    COL_C1Y,
    COL_C2X,
    COL_C2Y,
    COL_TYPE,
    COL_I,
    COL_J,
    COL_CW,
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


def test_fit_circle_3_points_perfect_circle():
    """Test fitting a circle through three points on a perfect circle."""
    center = (2.0, 3.0)
    radius = 5.0
    angles = [0, np.pi / 3, 2 * np.pi / 3]
    points = [
        (
            center[0] + radius * np.cos(theta),
            center[1] + radius * np.sin(theta),
            0.0,
        )
        for theta in angles
    ]
    result = fit_circle_3_points(points[0], points[1], points[2])
    assert result is not None

    (xc, yc), r = result
    assert xc == pytest.approx(center[0], abs=1e-6)
    assert yc == pytest.approx(center[1], abs=1e-6)
    assert r == pytest.approx(radius, abs=1e-6)


def test_fit_circle_3_points_collinear_returns_none():
    """Test collinear points return None."""
    p1 = (0.0, 0.0, 0.0)
    p2 = (2.0, 2.0, 0.0)
    p3 = (5.0, 5.0, 0.0)
    assert fit_circle_3_points(p1, p2, p3) is None


def test_fit_circle_3_points_2d_points():
    """Test fitting with 2D points (no z coordinate)."""
    p1 = (0.0, 0.0)
    p2 = (0.0, 2.0)
    p3 = (2.0, 0.0)
    result = fit_circle_3_points(p1, p2, p3)
    assert result is not None

    (xc, yc), r = result
    assert xc == pytest.approx(1.0, abs=1e-6)
    assert yc == pytest.approx(1.0, abs=1e-6)
    assert r == pytest.approx(np.sqrt(2), abs=1e-6)


def test_fit_circle_3_points_3d_points():
    """Test fitting with 3D points (z coordinate is ignored)."""
    p1 = (0.0, 0.0, 5.0)
    p2 = (0.0, 2.0, 10.0)
    p3 = (2.0, 0.0, -3.0)
    result = fit_circle_3_points(p1, p2, p3)
    assert result is not None

    (xc, yc), r = result
    assert xc == pytest.approx(1.0, abs=1e-6)
    assert yc == pytest.approx(1.0, abs=1e-6)
    assert r == pytest.approx(np.sqrt(2), abs=1e-6)


def test_fit_circle_3_points_small_radius():
    """Test fitting a small-radius circle."""
    center = (0.0, 0.0)
    radius = 0.1
    angles = [0, np.pi / 2, np.pi]
    points = [
        (
            center[0] + radius * np.cos(theta),
            center[1] + radius * np.sin(theta),
            0.0,
        )
        for theta in angles
    ]
    result = fit_circle_3_points(points[0], points[1], points[2])
    assert result is not None

    (xc, yc), r = result
    assert xc == pytest.approx(center[0], abs=1e-6)
    assert yc == pytest.approx(center[1], abs=1e-6)
    assert r == pytest.approx(radius, abs=1e-6)


def test_fit_circle_3_points_nearly_collinear():
    """Test nearly collinear points (should return None)."""
    p1 = (0.0, 0.0, 0.0)
    p2 = (1.0, 1e-10, 0.0)
    p3 = (2.0, 2e-10, 0.0)
    assert fit_circle_3_points(p1, p2, p3) is None


def test_fit_circle_3_points_offset_center():
    """Test fitting with offset center."""
    center = (10.0, -5.0)
    radius = 3.0
    angles = [np.pi / 4, np.pi / 2, 3 * np.pi / 4]
    points = [
        (
            center[0] + radius * np.cos(theta),
            center[1] + radius * np.sin(theta),
            0.0,
        )
        for theta in angles
    ]
    result = fit_circle_3_points(points[0], points[1], points[2])
    assert result is not None

    (xc, yc), r = result
    assert xc == pytest.approx(center[0], abs=1e-6)
    assert yc == pytest.approx(center[1], abs=1e-6)
    assert r == pytest.approx(radius, abs=1e-6)


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


def test_fit_points_to_primitives_single_line():
    """Tests that collinear points form a single line."""
    points = [(0.0, 0.0, 0.0), (5.0, 0.0, 0.0), (10.0, 0.0, 0.0)]
    tolerance = 0.1
    cmds = fit_points_to_primitives(points, tolerance)

    assert len(cmds) == 1
    cmd = cmds[0]
    assert cmd[COL_TYPE] == CMD_TYPE_LINE
    assert np.allclose((cmd[COL_X], cmd[COL_Y]), (10.0, 0.0))


def test_fit_points_to_primitives_single_arc():
    """Tests that points on a circle form a single arc."""
    center = (0.0, 0.0)
    radius = 10.0
    # 90 degree arc
    angles = np.linspace(0, np.pi / 2, 20)
    points = [
        (
            center[0] + radius * np.cos(t),
            center[1] + radius * np.sin(t),
            0.0,
        )
        for t in angles
    ]
    tolerance = 0.1
    cmds = fit_points_to_primitives(points, tolerance)

    assert len(cmds) == 1
    cmd = cmds[0]
    assert cmd[COL_TYPE] == CMD_TYPE_ARC
    assert np.allclose((cmd[COL_X], cmd[COL_Y]), (0.0, 10.0))
    # Center offset from start point (10, 0) is (-10, 0)
    assert np.allclose((cmd[COL_I], cmd[COL_J]), (-10.0, 0.0))
    # CCW
    assert cmd[COL_CW] == 0.0


def test_fit_points_to_primitives_corner_split():
    """Tests that a sharp corner splits into two lines."""
    # Line 1: (0,0) -> (10,0)
    l1 = [(x, 0.0, 0.0) for x in np.linspace(0, 10, 10)]
    # Line 2: (10,0) -> (10,10)
    l2 = [(10.0, y, 0.0) for y in np.linspace(1, 10, 10)]
    points = l1 + l2

    tolerance = 0.1
    cmds = fit_points_to_primitives(points, tolerance)

    assert len(cmds) == 2
    assert cmds[0][COL_TYPE] == CMD_TYPE_LINE
    assert np.allclose((cmds[0][COL_X], cmds[0][COL_Y]), (10.0, 0.0))
    assert cmds[1][COL_TYPE] == CMD_TYPE_LINE
    assert np.allclose((cmds[1][COL_X], cmds[1][COL_Y]), (10.0, 10.0))


def test_fit_points_to_primitives_line_arc_mixed():
    """Tests a straight line followed by an arc."""
    # Line segment
    line_pts = [(x, 0.0, 0.0) for x in np.linspace(0, 10, 11)]
    # Arc segment (tangent start at 10,0)
    # Center at (10, 5), radius 5. Start angle -pi/2, end 0
    angles = np.linspace(-np.pi / 2, 0, 11)
    arc_pts = [
        (10.0 + 5.0 * np.cos(t), 5.0 + 5.0 * np.sin(t), 0.0) for t in angles
    ]
    # Remove duplicate point at transition
    points = line_pts + arc_pts[1:]

    tolerance = 0.1
    cmds = fit_points_to_primitives(points, tolerance)

    # Should detect at least one line and one arc.
    # Depending on resolution and tolerance, it might be perfect or
    # slightly split, but we expect basic types.
    assert len(cmds) >= 2
    assert cmds[0][COL_TYPE] == CMD_TYPE_LINE
    assert cmds[-1][COL_TYPE] == CMD_TYPE_ARC


def test_get_max_line_deviation_collinear():
    """Test max deviation for collinear points."""
    points = [(0.0, 0.0, 0.0), (5.0, 0.0, 0.0), (10.0, 0.0, 0.0)]
    max_dist, max_idx = get_max_line_deviation(points, 0, 2)
    assert max_dist < 1e-9
    assert max_idx == 0


def test_get_max_line_deviation_with_deviation():
    """Test max deviation finds the furthest point."""
    points = [(0.0, 0.0, 0.0), (5.0, 1.0, 0.0), (10.0, 0.0, 0.0)]
    max_dist, max_idx = get_max_line_deviation(points, 0, 2)
    assert max_dist == pytest.approx(1.0, abs=1e-6)
    assert max_idx == 1


def test_get_max_line_deviation_coincident_endpoints():
    """Test max deviation with coincident endpoints."""
    points = [(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 0.0, 0.0)]
    max_dist, max_idx = get_max_line_deviation(points, 0, 2)
    assert max_dist == pytest.approx(1.0, abs=1e-6)
    assert max_idx == 1


def test_create_line_cmd_2d():
    """Test creating a line command from 2D point."""
    end_point = (10.0, 20.0)
    cmd = create_line_cmd(end_point)
    assert cmd[COL_TYPE] == CMD_TYPE_LINE
    assert cmd[COL_X] == 10.0
    assert cmd[COL_Y] == 20.0
    assert cmd[COL_Z] == 0.0


def test_create_line_cmd_3d():
    """Test creating a line command from 3D point."""
    end_point = (10.0, 20.0, 5.0)
    cmd = create_line_cmd(end_point)
    assert cmd[COL_TYPE] == CMD_TYPE_LINE
    assert cmd[COL_X] == 10.0
    assert cmd[COL_Y] == 20.0
    assert cmd[COL_Z] == 5.0


def test_create_arc_cmd_ccw():
    """Test creating an arc command for CCW direction."""
    start = (10.0, 0.0, 0.0)
    end = (0.0, 10.0, 0.0)
    center = (0.0, 0.0)
    cmd = create_arc_cmd(end, center, start)
    assert cmd[COL_TYPE] == CMD_TYPE_ARC
    assert cmd[COL_X] == 0.0
    assert cmd[COL_Y] == 10.0
    assert cmd[COL_Z] == 0.0
    assert cmd[COL_I] == -10.0
    assert cmd[COL_J] == 0.0
    assert cmd[COL_CW] == 0.0


def test_create_arc_cmd_cw():
    """Test creating an arc command for CW direction."""
    start = (0.0, 10.0, 0.0)
    end = (10.0, 0.0, 0.0)
    center = (0.0, 0.0)
    cmd = create_arc_cmd(end, center, start)
    assert cmd[COL_TYPE] == CMD_TYPE_ARC
    assert cmd[COL_X] == 10.0
    assert cmd[COL_Y] == 0.0
    assert cmd[COL_Z] == 0.0
    assert cmd[COL_I] == 0.0
    assert cmd[COL_J] == -10.0
    assert cmd[COL_CW] == 1.0


def test_fit_points_recursive_line():
    """Test recursive fitting produces a line for collinear points."""
    points = [(0.0, 0.0, 0.0), (5.0, 0.0, 0.0), (10.0, 0.0, 0.0)]
    cmds = fit_points_recursive(points, 0.1, 0, 2)
    assert len(cmds) == 1
    assert cmds[0][COL_TYPE] == CMD_TYPE_LINE


def test_fit_points_recursive_arc():
    """Test recursive fitting produces an arc for circular points."""
    center = (0.0, 0.0)
    radius = 10.0
    angles = np.linspace(0, np.pi / 2, 20)
    points = [
        (center[0] + radius * np.cos(t), center[1] + radius * np.sin(t), 0.0)
        for t in angles
    ]
    cmds = fit_points_recursive(points, 0.1, 0, len(points) - 1)
    assert len(cmds) == 1
    assert cmds[0][COL_TYPE] == CMD_TYPE_ARC


def test_fit_points_recursive_split():
    """Test recursive fitting splits at corner."""
    points = [(0.0, 0.0, 0.0), (10.0, 0.0, 0.0), (10.0, 10.0, 0.0)]
    cmds = fit_points_recursive(points, 0.1, 0, 2)
    assert len(cmds) == 2
    assert cmds[0][COL_TYPE] == CMD_TYPE_LINE
    assert cmds[1][COL_TYPE] == CMD_TYPE_LINE


def test_fit_points_recursive_empty():
    """Test recursive fitting with invalid range."""
    points = [(0.0, 0.0, 0.0), (10.0, 0.0, 0.0)]
    cmds = fit_points_recursive(points, 0.1, 0, 0)
    assert len(cmds) == 0


def test_fit_points_recursive_single_point():
    """Test recursive fitting with single point."""
    points = [(0.0, 0.0, 0.0), (10.0, 0.0, 0.0)]
    cmds = fit_points_recursive(points, 0.1, 0, 1)
    assert len(cmds) == 1
    assert cmds[0][COL_TYPE] == CMD_TYPE_LINE


def test_fit_arcs_simple_line():
    """Tests fit_arcs with a simple line geometry."""
    geo = Geometry()
    geo.move_to(0, 0)
    geo.line_to(10, 0)

    geo._sync_to_numpy()
    data = geo.data

    result = fit_arcs(data, 0.1)

    # Should preserve move and line commands
    assert result is not None
    assert len(result) == 2
    assert result[0][COL_TYPE] == CMD_TYPE_MOVE
    assert result[1][COL_TYPE] == CMD_TYPE_LINE


def test_fit_arcs_with_bezier():
    """Tests fit_arcs with a bezier curve."""
    geo = Geometry()
    geo.move_to(0, 0)
    geo.bezier_to(10, 10, c1x=2, c1y=5, c2x=8, c2y=5)

    geo._sync_to_numpy()
    data = geo.data

    result = fit_arcs(data, 0.1)

    # Should convert bezier to lines/arcs
    assert result is not None
    assert len(result) >= 1
    assert result[0][COL_TYPE] == CMD_TYPE_MOVE
    # The bezier should be simplified and fitted
    assert result[1][COL_TYPE] in (CMD_TYPE_LINE, CMD_TYPE_ARC)


def test_fit_arcs_empty():
    """Tests fit_arcs with empty geometry."""
    result = fit_arcs(None, 0.1)
    assert result is None


def test_fit_arcs_with_progress():
    """Tests fit_arcs with progress callback."""
    geo = Geometry()
    # Create a larger geometry to trigger progress callbacks
    for i in range(100):
        geo.move_to(i, 0)
        geo.line_to(i + 1, 0)

    geo._sync_to_numpy()
    data = geo.data

    progress_values = []

    def progress_callback(value: float) -> None:
        progress_values.append(value)

    fit_arcs(data, 0.1, progress_callback)

    # Progress should be called
    assert len(progress_values) > 0
    # Last progress value should be close to 1.0
    # Note: fit_arcs only calls progress every 50 rows
    assert progress_values[-1] >= 0.75


def test_optimize_path_from_array_rdp_simplification():
    """Tests RDP simplification (fit_arcs=False)."""
    data = np.array([
        [CMD_TYPE_MOVE, 0.0, 0.0, 0.0, 0, 0, 0, 0],
        [CMD_TYPE_LINE, 1.0, 1.0, 0.0, 0, 0, 0, 0],
        [CMD_TYPE_LINE, 2.0, 2.0, 0.0, 0, 0, 0, 0],
        [CMD_TYPE_LINE, 3.0, 3.0, 0.0, 0, 0, 0, 0],
        [CMD_TYPE_LINE, 10.0, 10.0, 0.0, 0, 0, 0, 0],
    ], dtype=np.float64)

    result = optimize_path_from_array(data, tolerance=0.1, fit_arcs=False)

    assert result is not None
    assert len(result) == 2
    assert result[0, COL_TYPE] == CMD_TYPE_MOVE
    assert np.allclose(result[0, 1:4], (0, 0, 0))
    assert result[1, COL_TYPE] == CMD_TYPE_LINE
    assert np.allclose(result[1, 1:4], (10, 10, 0))


def test_optimize_path_from_array_arc_fitting():
    """Tests arc fitting (fit_arcs=True)."""
    data = np.array([
        [CMD_TYPE_MOVE, 0.0, 0.0, 0.0, 0, 0, 0, 0],
        [CMD_TYPE_LINE, 1.0, 0.0, 0.0, 0, 0, 0, 0],
        [CMD_TYPE_LINE, 2.0, 0.0, 0.0, 0, 0, 0, 0],
        [CMD_TYPE_LINE, 3.0, 0.0, 0.0, 0, 0, 0, 0],
        [CMD_TYPE_LINE, 4.0, 0.0, 0.0, 0, 0, 0, 0],
        [CMD_TYPE_LINE, 5.0, 0.0, 0.0, 0, 0, 0, 0],
    ], dtype=np.float64)

    result = optimize_path_from_array(data, tolerance=0.1, fit_arcs=True)

    assert result is not None
    assert len(result) == 2
    assert result[0, COL_TYPE] == CMD_TYPE_MOVE
    assert result[1, COL_TYPE] == CMD_TYPE_LINE


def test_optimize_path_from_array_preserves_arcs():
    """Tests that arcs are preserved."""
    data = np.array([
        [CMD_TYPE_MOVE, 0.0, 0.0, 0.0, 0, 0, 0, 0],
        [CMD_TYPE_LINE, 1.0, 0.0, 0.0, 0, 0, 0, 0],
        [CMD_TYPE_LINE, 2.0, 0.0, 0.0, 0, 0, 0, 0],
        [CMD_TYPE_ARC, 4.0, 0.0, 0.0, 1.0, 0.0, 0, 0],
        [CMD_TYPE_LINE, 5.0, 0.0, 0.0, 0, 0, 0, 0],
        [CMD_TYPE_LINE, 6.0, 0.0, 0.0, 0, 0, 0, 0],
    ], dtype=np.float64)

    result = optimize_path_from_array(data, tolerance=0.1, fit_arcs=False)

    assert result is not None
    assert len(result) == 4
    assert result[0, COL_TYPE] == CMD_TYPE_MOVE
    assert result[1, COL_TYPE] == CMD_TYPE_LINE
    assert result[2, COL_TYPE] == CMD_TYPE_ARC
    assert result[3, COL_TYPE] == CMD_TYPE_LINE


def test_optimize_path_from_array_empty():
    """Tests empty input."""
    result = optimize_path_from_array(None, tolerance=0.1, fit_arcs=False)
    assert result is not None
    assert len(result) == 0


def test_optimize_path_from_array_empty_array():
    """Tests empty array input."""
    data = np.array([])
    result = optimize_path_from_array(data, tolerance=0.1, fit_arcs=False)
    assert result is not None
    assert len(result) == 0


def test_optimize_path_from_array_moveto_breaks_chain():
    """Tests that MoveTo breaks the simplification chain."""
    data = np.array([
        [CMD_TYPE_MOVE, 0.0, 0.0, 0.0, 0, 0, 0, 0],
        [CMD_TYPE_LINE, 5.0, 0.0, 0.0, 0, 0, 0, 0],
        [CMD_TYPE_LINE, 10.0, 0.0, 0.0, 0, 0, 0, 0],
        [CMD_TYPE_MOVE, 20.0, 0.0, 0.0, 0, 0, 0, 0],
        [CMD_TYPE_LINE, 25.0, 0.0, 0.0, 0, 0, 0, 0],
        [CMD_TYPE_LINE, 30.0, 0.0, 0.0, 0, 0, 0, 0],
    ], dtype=np.float64)

    result = optimize_path_from_array(data, tolerance=0.1, fit_arcs=False)

    assert result is not None
    assert len(result) == 4
    assert result[0, COL_TYPE] == CMD_TYPE_MOVE
    assert result[1, COL_TYPE] == CMD_TYPE_LINE
    assert result[2, COL_TYPE] == CMD_TYPE_MOVE
    assert result[3, COL_TYPE] == CMD_TYPE_LINE


def test_optimize_path_from_array_z_axis_preservation():
    """Tests that Z coordinates are preserved."""
    data = np.array([
        [CMD_TYPE_MOVE, 0.0, 0.0, 1.0, 0, 0, 0, 0],
        [CMD_TYPE_LINE, 5.0, 0.0, 2.0, 0, 0, 0, 0],
        [CMD_TYPE_LINE, 10.0, 0.0, 3.0, 0, 0, 0, 0],
    ], dtype=np.float64)

    result = optimize_path_from_array(data, tolerance=0.1, fit_arcs=False)

    assert result is not None
    assert len(result) == 2
    assert np.allclose(result[0, 1:4], (0, 0, 1))
    assert np.allclose(result[1, 1:4], (10, 0, 3))
