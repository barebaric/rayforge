import numpy as np
from typing import cast
from rayforge.core.ops import Ops, MoveToCommand, LineToCommand, ArcToCommand
from rayforge.pipeline.transformer.arcwelder import ArcWeld


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
    arc = cast(ArcToCommand, arc_commands[0])
    assert arc.end is not None
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
    arc_cmd = cast(ArcToCommand, commands[1])
    assert arc_cmd.end is not None
    end_x, end_y, _ = arc_cmd.end
    assert np.isclose(end_x, -5.0, atol=0.1), "Arc ends at wrong X"
    assert np.isclose(end_y, 0.0, atol=0.1), "Arc ends at wrong Y"

    # Validate straight lines after arc
    line_cmd = cast(LineToCommand, commands[2])
    assert line_cmd.end == (-10.0, 0.0, 0.0)


def test_find_longest_valid_arc():
    welder = ArcWeld(tolerance=0.1)

    # Valid arc segment
    theta = np.linspace(0, np.pi / 2, 10)
    arc_points = [(5 * np.cos(t), 5 * np.sin(t), 0.0) for t in theta]
    arc, end = welder._find_longest_valid_arc(arc_points, 0)
    assert arc is not None
    assert end == 10

    # Straight line segment
    line_points = [(float(x), 0.0, 0.0) for x in range(10)]
    arc, end = welder._find_longest_valid_arc(line_points, 0)
    assert arc is None


def test_is_valid_arc():
    welder = ArcWeld(tolerance=0.1)
    from rayforge.core.geo.analysis import fit_circle_to_points

    # Test 1: Valid arc
    valid_arc = ((10.0, 10.0), 50.0, 0.05)
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
    high_error_arc = (valid_arc[0], valid_arc[1], 0.2)
    assert welder._is_valid_arc(subsegment, high_error_arc) is False

    # Test 3: Radius too small
    small_radius_arc = (valid_arc[0], 0.5, valid_arc[2])
    assert welder._is_valid_arc(subsegment, small_radius_arc) is False

    # Test 4: Radius too large
    large_radius_arc = (valid_arc[0], 20000.0, valid_arc[2])
    assert welder._is_valid_arc(subsegment, large_radius_arc) is False

    # Test 5: Colinear points
    colinear_subsegment = [(0.0, 0.0, 0.0), (2.0, 0.0, 0.0), (4.0, 0.0, 0.0)]
    colinear_arc = fit_circle_to_points(colinear_subsegment)
    assert welder._is_valid_arc(colinear_subsegment, colinear_arc) is False

    # Test 6: Invalid arc (None)
    assert welder._is_valid_arc(subsegment, None) is False


def test_is_valid_arc_angular_continuity():
    welder = ArcWeld(tolerance=0.1)
    from rayforge.core.geo.analysis import fit_circle_to_points

    # Exact points on a circle (radius 5, center (0,0))
    angles = np.deg2rad([0, 30, 60, 90])
    valid_points = [
        (5 * np.cos(theta), 5 * np.sin(theta), 0.0) for theta in angles
    ]
    valid_arc = fit_circle_to_points(valid_points)
    assert welder._is_valid_arc(valid_points, valid_arc) is True

    # Add point diametrically opposite to start (180° total span)
    invalid_points = valid_points + [
        (5 * np.cos(np.pi), 5 * np.sin(np.pi), 0.0)
    ]
    invalid_arc = fit_circle_to_points(invalid_points)
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
    arc = cast(ArcToCommand, ops.commands[1])
    assert arc.end is not None
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
    segment = [
        (0.0, 0.0, 0.0),
        (1.0, 0.0, 0.0),
        (2.0, 0.0, 0.0),
        (3.0, 0.0, 0.0),
        (4.0, 0.0, 0.0),
    ]
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


def test_arc_welder_passes_through_non_weldable_segments():
    """Verify that ArcWeld does not modify segments with arcs."""
    ops = Ops()
    ops.move_to(0, 0)
    ops.arc_to(10, 10, 5, 0)
    ops.move_to(20, 20)  # Separate segment

    original_ops_copy = ops.copy()

    welder = ArcWeld()
    welder.run(ops)

    assert len(ops.commands) == len(original_ops_copy.commands)
    for cmd_new, cmd_orig in zip(ops.commands, original_ops_copy.commands):
        assert type(cmd_new) is type(cmd_orig)
        assert cmd_new.end == cmd_orig.end
