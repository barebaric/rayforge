import pytest
import math
import numpy as np
from rayforge.core.ops import Ops, MoveToCommand, LineToCommand, ArcToCommand
from rayforge.pipeline.transformer.arcwelder import (
    ArcWeld,
    fit_circle,
    remove_duplicates,
    are_colinear,
    is_clockwise,
    arc_direction,
    arc_to_polyline_deviation,
)


def test_remove_duplicates():
    segment = [(1, 1), (1, 1), (2, 2), (2, 2)]
    assert remove_duplicates(segment) == [(1, 1), (2, 2)]


def test_are_colinear():
    # Colinear points (horizontal)
    points = [(0, 0, 0), (5, 0, 0), (10, 0, 0)]
    assert are_colinear(points) is True

    # Colinear points (vertical)
    points = [(0, 0, 0), (0, 5, 0), (0, 10, 0)]
    assert are_colinear(points) is True

    # Non-colinear points
    points = [(0, 0, 0), (1, 1, 0), (2, 2.1, 0)]
    assert are_colinear(points) is False


def test_is_clockwise():
    # Clockwise points (right half-circle)
    points = [(0, 0, 0), (1, 1, 0), (2, 0, 0)]
    assert is_clockwise(points) is True

    # Counter-clockwise points (left half-circle)
    points = [(0, 0, 0), (-1, 1, 0), (-2, 0, 0)]
    assert is_clockwise(points) is False


def test_arc_direction_clockwise_half_circle():
    """Test a semicircle moving clockwise."""
    center = (0, 0)
    points = [
        (1, 0, 0),  # 0 radians
        (0, -1, 0),  # -π/2 (or 3π/2)
        (-1, 0, 0),  # π radians (unwrapped to -π)
    ]
    assert arc_direction(points, center) is True


def test_arc_direction_counter_clockwise_half_circle():
    """Test a semicircle moving counter-clockwise."""
    center = (0, 0)
    points = [
        (1, 0, 0),  # 0 radians
        (0, 1, 0),  # π/2
        (-1, 0, 0),  # π
    ]
    assert arc_direction(points, center) is False


def test_arc_direction_clockwise_full_circle():
    """Test a full clockwise circle."""
    center = (0, 0)
    points = [
        (1, 0, 0),
        (0, -1, 0),
        (-1, 0, 0),
        (0, 1, 0),
        (1, 0, 0),
    ]
    assert arc_direction(points, center) is True


def test_arc_direction_counter_clockwise_full_circle():
    """Test a full counter-clockwise circle."""
    center = (0, 0)
    points = [
        (1, 0, 0),
        (0, 1, 0),
        (-1, 0, 0),
        (0, -1, 0),
        (1, 0, 0),
    ]
    assert arc_direction(points, center) is False


def test_arc_direction_minimal_clockwise_arc():
    """Test a minimal 3-point clockwise arc."""
    center = (0, 0)
    points = [
        (2, 0, 0),
        (1, -1, 0),
        (0, 0, 0),
    ]
    assert arc_direction(points, center) is True


def test_arc_direction_minimal_counter_clockwise_arc():
    """Test a minimal 3-point counter-clockwise arc."""
    center = (0, 0)
    points = [
        (2, 0, 0),
        (1, 1, 0),
        (0, 0, 0),
    ]
    assert arc_direction(points, center) is False


def test_arc_direction_crossing_angle_discontinuity_counter_clockwise():
    """Test an arc crossing the π/-π discontinuity (counter-clockwise)."""
    center = (0, 0)
    points = [
        (1, 0.1, 0),  # ~0 radians
        (0, 1, 0),  # π/2
        (-1, 0.1, 0),  # ~π radians (unwrapped to -π)
    ]
    assert arc_direction(points, center) is False


def test_arc_direction_small_radius_arc():
    """Test a small-radius clockwise arc."""
    center = (1, 1)
    points = [
        (1.1, 1, 0),
        (1, 0.9, 0),
        (0.9, 1, 0),
    ]
    assert arc_direction(points, center) is True


def test_fit_circle_collinear_returns_none():
    """Test collinear points return None."""
    points = [(0, 0, 0), (2, 2, 0), (5, 5, 0)]
    assert fit_circle(points) is None


def test_fit_circle_perfect_circle():
    """Test perfect circle fitting."""
    center = (2, 3)
    radius = 5
    angles = np.linspace(0, 2 * np.pi, 20)
    points = [
        (
            center[0] + radius * np.cos(theta),
            center[1] + radius * np.sin(theta),
            0.0,
        )
        for theta in angles
    ]
    result = fit_circle(points)
    assert result is not None

    (xc, yc), r, error = result
    assert xc == pytest.approx(center[0], abs=1e-6)
    assert yc == pytest.approx(center[1], abs=1e-6)
    assert r == pytest.approx(radius, abs=1e-6)
    assert error < 1e-6


def test_fit_circle_noisy_circle():
    """Test circle fitting with noisy points."""
    center = (-1, 4)
    radius = 3
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
    result = fit_circle(points)
    assert result is not None

    (xc, yc), r, error = result
    assert xc == pytest.approx(center[0], abs=0.15)
    assert yc == pytest.approx(center[1], abs=0.15)
    assert r == pytest.approx(radius, abs=0.15)
    assert error < 0.2


def test_fit_circle_insufficient_points():
    """Test 1-2 points or duplicates return None."""
    assert fit_circle([(0, 0, 0)]) is None
    assert fit_circle([(1, 2, 0), (3, 4, 0)]) is None
    assert fit_circle([(5, 5, 0), (5, 5, 0), (5, 5, 0)]) is None


def test_fit_circle_small_radius():
    """Test small-radius circle fitting."""
    center = (0, 0)
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
    result = fit_circle(points)
    assert result is not None

    (xc, yc), r, error = result
    assert r == pytest.approx(radius, rel=0.01)


def test_fit_circle_nearly_collinear_but_valid():
    """Test near-colinear points that still form a valid circle."""
    points = [
        (0, 0, 0),
        (1, 0, 0),
        (2, 0.2, 0),
    ]  # Area = 0.2 (>1e-3 threshold)
    result = fit_circle(points)
    assert result is not None  # Should attempt to fit

    (xc, yc), r, error = result
    assert error < 1.0  # Large error expected, but still returns a circle


def test_fit_circle_horizontal_line_fails():
    """Test horizontal line (collinear) returns None."""
    points = [(0, 0, 0), (1, 0, 0), (2, 0, 0)]
    assert fit_circle(points) is None


def test_fit_circle_semicircle_accuracy():
    """
    Verify fit_circle() returns correct parameters for a perfect semicircle
    """
    center = (5, 0)
    radius = 10.0
    num_points = 20

    # Generate semicircle points (180 degrees)
    angles = np.linspace(0, np.pi, num_points)
    points = [
        (
            center[0] + radius * np.cos(theta),
            center[1] + radius * np.sin(theta),
            0.0,
        )
        for theta in angles
    ]

    # Fit circle
    result = fit_circle(points)
    assert result is not None, "Failed to fit semicircle"

    (xc, yc), r, error = result

    # Validate center coordinates (should match real center (5,0))
    assert np.isclose(xc, 5.0, atol=0.001), (
        f"Center X error: {xc:.3f} != 5.000"
    )
    assert np.isclose(yc, 0.0, atol=0.001), (
        f"Center Y error: {yc:.3f} != 0.000"
    )

    # Validate radius
    assert np.isclose(r, 10.0, rtol=0.001), f"Radius error: {r:.3f} != 10.000"

    # Validate maximum deviation
    assert error < 1e-6, f"Fitting error too large: {error:.6f}"


def test_fit_circle_partial_arc_accuracy():
    """
    Verify fit_circle() accuracy for a 90-degree arc offset from the centroid
    """
    center = (7, 3)
    radius = 5.0
    # Generate points along a 90-degree arc (π/2 to π radians)
    angles = np.linspace(np.pi / 2, np.pi, 50)
    points = [
        (
            center[0] + radius * np.cos(theta),
            center[1] + radius * np.sin(theta),
            0.0,
        )
        for theta in angles
    ]

    # Fit circle
    result = fit_circle(points)
    assert result is not None, "Failed to fit partial arc"

    (xc, yc), r, error = result

    # Validate center (true center is (7,3))
    assert np.isclose(xc, 7.0, atol=0.01), f"Center X: {xc:.3f} ≠ 7.0"
    assert np.isclose(yc, 3.0, atol=0.01), f"Center Y: {yc:.3f} ≠ 3.0"

    # Validate radius
    assert np.isclose(r, 5.0, rtol=0.01), f"Radius: {r:.3f} ≠ 5.0"

    # Validate error
    assert error < 0.05, f"Error too large: {error:.3f}mm"


def test_arc_to_polyline_deviation_perfect_arc():
    """Test deviation for a perfect 90-degree arc."""
    center = (7, 3)
    radius = 5.0
    angles = np.linspace(np.pi / 2, np.pi, 10)
    points = [
        (center[0] + radius * np.cos(t), center[1] + radius * np.sin(t), 0.0)
        for t in angles
    ]
    deviation = arc_to_polyline_deviation(points, center, radius)
    assert deviation < 0.05, f"Deviation too large: {deviation}"


def test_arc_to_polyline_deviation_too_large():
    """Test deviation for a perfect 90-degree arc."""
    center = (7, 3)
    radius = 5.0
    angles = np.linspace(np.pi / 2, np.pi, 5)
    points = [
        (center[0] + radius * np.cos(t), center[1] + radius * np.sin(t), 0.0)
        for t in angles
    ]
    deviation = arc_to_polyline_deviation(points, center, radius)
    assert deviation > 0.05, f"Expected larger deviation: {deviation}"


def test_arc_to_polyline_deviation_straight_line():
    """Test deviation for a straight line with a large-radius arc."""
    points = [(0, 0, 0), (1, 0.01, 0), (2, -0.01, 0), (3, 0, 0)]
    center = (1.5, 10)  # Large radius arc
    radius = 100.0
    deviation = arc_to_polyline_deviation(points, center, radius)
    assert deviation > 0.05, f"Deviation too small: {deviation}"


def test_arc_to_polyline_deviation_single_segment():
    """Test deviation for a single line segment."""
    points = [(0, 0, 0), (1, 1, 0)]
    center = (0.5, 0.5)
    radius = math.sqrt(0.5)
    deviation = arc_to_polyline_deviation(points, center, radius)
    assert deviation < 1.0, f"Deviation too large: {deviation}"


def test_arc_welder_converts_semicircle():
    """
    Test if ArcWelder correctly converts line segments into a semicircular arc.
    """
    # Create a semicircle using line segments
    ops = Ops()
    center = 0, 0
    radius = 10.0
    num_points = 20  # Generate 20 points along the semicircle

    # Generate points on a semicircle (180 degrees)
    angles = np.linspace(0, np.pi, num_points)
    points = [
        (
            center[0] + radius * np.cos(theta),
            center[1] + radius * np.sin(theta),
            0.0,
        )
        for theta in angles
    ]

    # Build path
    ops.move_to(*points[0])
    for p in points[1:]:
        ops.line_to(*p)

    # Process with ArcWelder
    welder = ArcWeld(tolerance=0.1, min_points=3, max_points=25)
    welder.run(ops)

    # Verify arc generation
    arc_commands = [cmd for cmd in ops if cmd.__class__ == ArcToCommand]
    assert len(arc_commands) == 1, (
        "Should replace the entire segment with one arc"
    )

    # Validate arc parameters
    arc = arc_commands[0]
    end_x, end_y, end_z = arc.end
    i, j = arc.center_offset

    # Check center offsets (I/J relative to start point (10,0))
    assert np.isclose(i, -10.0, atol=0.1), f"Expected I ≈ -10, got {i}"
    assert np.isclose(j, 0.0, atol=0.1), f"Expected J ≈ 0, got {j}"

    # Validate endpoint (-10,0)
    assert np.isclose(end_x, -10.0, atol=0.1), f"Expected X ≈ -10, got {end_x}"
    assert np.isclose(end_y, 0.0, atol=0.1), f"Expected Y ≈ 0, got {end_y}"
    assert np.isclose(end_z, 0.0, atol=0.1), f"Expected Z = 0, got {end_z}"

    # Validate direction (should be counter-clockwise for 180-degree arc)
    assert not arc.clockwise, "Expected CCW arc, got clockwise"


def test_arc_welder_ignores_straight_lines():
    """Test if ArcWelder leaves straight line segments unchanged."""
    ops = Ops()
    ops.move_to(10, 10, 0)
    ops.line_to(11, 11, 0)
    ops.line_to(12, 12, 0)
    ops.line_to(13, 13, 0)
    ops.line_to(14, 14, 0)
    ops.line_to(15, 15, 0)
    ops.line_to(16, 16, 0)
    ops.line_to(17, 17, 0)

    welder = ArcWeld(tolerance=0.1, min_points=3)
    welder.run(ops)

    # No arcs should be generated for colinear points
    assert all((not isinstance(cmd, ArcToCommand)) for cmd in ops), (
        "Arcs incorrectly generated"
    )


def test_arc_with_trailing_straight_lines():
    ops = Ops()
    radius = 5.0
    center = 0, 0

    # Single segment containing:
    # 1. Semicircle (10 points)
    # 2. Straight line (5 points)
    ops.move_to(radius, 0, 0.0)  # Start at (5,0,0)

    # Semicircle points
    angles = np.linspace(0, np.pi, 20)
    for theta in angles[1:]:  # Skip first point (5,0)
        x = center[0] + radius * np.cos(theta)
        y = center[1] + radius * np.sin(theta)
        ops.line_to(x, y, 0.0)

    # Straight line points
    for x in np.linspace(-5, -10, 5):
        ops.line_to(x, 0, 0.0)

    welder = ArcWeld(tolerance=0.1, min_points=5, max_points=50)
    welder.run(ops)

    # Validate results
    commands = ops.commands
    cmd_types = [cmd.__class__ for cmd in commands]
    assert cmd_types == [
        MoveToCommand,
        ArcToCommand,
        LineToCommand,
    ], f"Unexpected command sequence: {cmd_types}"

    # Validate arc parameters
    arc = commands[1]
    end_x, end_y, _ = arc.end
    assert np.isclose(end_x, -5.0, atol=0.1), "Arc ends at wrong X"
    assert np.isclose(end_y, 0.0, atol=0.1), "Arc ends at wrong Y"

    # Validate straight lines after arc
    assert commands[2].end == (-10.0, 0.0, 0.0)


def test_find_longest_valid_arc():
    welder = ArcWeld(tolerance=0.1)

    # Valid arc segment
    theta = np.linspace(0, np.pi / 2, 10)
    arc_points = [(5 * np.cos(t), 5 * np.sin(t), 0.0) for t in theta]
    arc, end = welder._find_longest_valid_arc(arc_points, 0)
    assert arc is not None
    assert end == 10

    # Straight line segment
    line_points = [(x, 0, 0) for x in range(10)]
    arc, end = welder._find_longest_valid_arc(line_points, 0)
    assert arc is None


def test_is_valid_arc():
    welder = ArcWeld(tolerance=0.1)

    # Test 1: Valid arc
    valid_arc = (10.0, 10.0), 50.0, 0.05
    subsegment = [
        (
            10.0 + 50.0 * np.cos(np.deg2rad(0)),
            10.0 + 50.0 * np.sin(np.deg2rad(0)),
            0.0,
        ),
        (
            10.0 + 50.0 * np.cos(np.deg2rad(30)),
            10.0 + 50.0 * np.sin(np.deg2rad(30)),
            0.0,
        ),
        (
            10.0 + 50.0 * np.cos(np.deg2rad(60)),
            10.0 + 50.0 * np.sin(np.deg2rad(60)),
            0.0,
        ),
    ]
    assert welder._is_valid_arc(subsegment, valid_arc) is True

    # Test 2: Error exceeds tolerance
    high_error_arc = *valid_arc[:2], 0.2
    assert welder._is_valid_arc(subsegment, high_error_arc) is False

    # Test 3: Radius too small
    small_radius_arc = valid_arc[0], 0.5, valid_arc[2]
    assert welder._is_valid_arc(subsegment, small_radius_arc) is False

    # Test 4: Radius too large
    large_radius_arc = valid_arc[0], 20000.0, valid_arc[2]
    assert welder._is_valid_arc(subsegment, large_radius_arc) is False

    # Test 5: Colinear points
    colinear_subsegment = [(0.0, 0.0, 0.0), (2.0, 0.0, 0.0), (4.0, 0.0, 0.0)]
    colinear_arc = fit_circle(colinear_subsegment)
    assert welder._is_valid_arc(colinear_subsegment, colinear_arc) is False

    # Test 6: Invalid arc (None)
    assert welder._is_valid_arc(subsegment, None) is False


def test_is_valid_arc_angular_continuity():
    welder = ArcWeld(tolerance=0.1)

    # Exact points on a circle (radius 5, center (0,0))
    angles = np.deg2rad([0, 30, 60, 90])
    valid_points = [
        (5 * np.cos(theta), 5 * np.sin(theta), 0.0) for theta in angles
    ]
    valid_arc = fit_circle(valid_points)
    assert welder._is_valid_arc(valid_points, valid_arc) is True

    # Add point diametrically opposite to start (180° total span)
    invalid_points = valid_points + [
        (5 * np.cos(np.pi), 5 * np.sin(np.pi), 0.0)
    ]
    invalid_arc = fit_circle(invalid_points)
    assert welder._is_valid_arc(invalid_points, invalid_arc) is False, (
        "Should reject arc due to large angular step"
    )


def test_arc_processing_flow():
    welder = ArcWeld(tolerance=0.1, min_points=3, max_points=50)
    ops = Ops()

    # Exact semicircle points on a circle with radius 5, center (0,0)
    angles = np.linspace(0, np.pi, 40)
    segment = [(5 * np.cos(theta), 5 * np.sin(theta), 0.0) for theta in angles]

    welder.process_segment(segment, ops)

    # Validate command sequence
    cmd_types = [cmd.__class__ for cmd in ops]
    assert cmd_types == [MoveToCommand, ArcToCommand], (
        "Should replace entire segment with one arc"
    )

    # Validate arc parameters
    arc = ops.commands[1]
    end_x, end_y, end_z = arc.end
    i, j = arc.center_offset

    assert np.isclose(end_x, -5.0, atol=0.1)
    assert np.isclose(end_y, 0.0, atol=0.1)
    assert np.isclose(end_z, 0.0, atol=0.1)
    assert np.isclose(i, -5.0, atol=0.1)  # Center (0,0) - start (5,0) → I=-5
    assert np.isclose(j, 0.0, atol=0.1)
    assert not arc.clockwise  # CCW for ascending angles


def test_process_segment_structure():
    # Set min_points <= segment length to ensure consolidation logic is tested.
    # The default is 6, but our segment only has 5 points.
    welder = ArcWeld(tolerance=0.1, min_points=5)
    ops = Ops()

    # Single straight line segment
    segment = [(0, 0, 0), (1, 0, 0), (2, 0, 0), (3, 0, 0), (4, 0, 0)]
    welder.process_segment(segment, ops)
    # Colinear points should be consolidated into a single LineTo command
    assert [cmd.__class__ for cmd in ops] == [
        MoveToCommand,
        LineToCommand,
    ]


def test_move_to_handling():
    welder = ArcWeld()
    ops = Ops()

    # Manually build command sequence
    ops.commands = [
        MoveToCommand((0, 0, 0)),
        LineToCommand((1, 0, 0)),
        MoveToCommand((2, 0, 0)),
        LineToCommand((3, 0, 0)),
    ]

    # Process commands
    welder.run(ops)

    # Verify output
    assert len(ops.commands) == 4
    assert [cmd.__class__ for cmd in ops] == [
        MoveToCommand,
        LineToCommand,
        MoveToCommand,
        LineToCommand,
    ]
