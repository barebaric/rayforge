import numpy as np
from rayforge.pipeline.transformer.arcwelder import ArcWeld


def generate_arc_points(center, radius, start_angle, end_angle, num_points):
    """Helper to generate points along an arc"""
    angles = np.linspace(start_angle, end_angle, num_points)
    return [
        (
            center[0] + radius * np.cos(theta),
            center[1] + radius * np.sin(theta),
        )
        for theta in angles
    ]


def test_find_perfect_arc():
    """Test ideal semicircle with sufficient points"""
    welder = ArcWeld(tolerance=0.1, min_points=5, max_points=25)
    center = (0, 0)
    radius = 10.0
    points = generate_arc_points(center, radius, 0, np.pi, 20)

    best_arc, best_end = welder._find_longest_valid_arc(points, 0)

    assert best_arc is not None
    assert best_end == len(points)
    assert np.isclose(best_arc[0][0], center[0], atol=0.1)
    assert np.isclose(best_arc[0][1], center[1], atol=0.1)
    assert np.isclose(best_arc[1], radius, rtol=0.01)


def test_reject_straight_line():
    """Test colinear points should return None"""
    welder = ArcWeld(tolerance=1.0, min_points=3)  # Loose tolerance
    points = [(x, 0.0) for x in np.linspace(0, 10, 10)]

    best_arc, best_end = welder._find_longest_valid_arc(points, 0)

    assert best_arc is None
    assert best_end == 0


def test_find_partial_arc():
    """Test segment with valid arc followed by invalid points"""
    welder = ArcWeld(tolerance=0.1, min_points=5)

    # First 15 points: valid arc
    valid_points = generate_arc_points((0, 0), 5, 0, np.pi / 2, 15)
    # Next 5 points: straight line
    invalid_points = [(x, 5.0) for x in np.linspace(5, 10, 5)]
    combined = valid_points + invalid_points

    best_arc, best_end = welder._find_longest_valid_arc(combined, 0)

    assert best_arc is not None
    assert best_end == 15
    assert np.isclose(best_arc[1], 5.0, rtol=0.05)


def test_min_points_threshold():
    """Test exact minimum points requirement"""
    welder = ArcWeld(tolerance=0.1, min_points=5)
    points = generate_arc_points(
        (2, 2), 3, 0, np.pi / 4, 5
    )  # Exactly min_points

    best_arc, best_end = welder._find_longest_valid_arc(points, 0)

    assert best_arc is not None
    assert best_end == 5


def test_ignore_short_segments():
    """Test segments shorter than min_points are ignored"""
    welder = ArcWeld(tolerance=0.1, min_points=5)
    points = generate_arc_points(
        (0, 0), 5, 0, np.pi / 4, 4
    )  # 1 less than min_points

    best_arc, best_end = welder._find_longest_valid_arc(points, 0)

    assert best_arc is None
    assert best_end == 0


def test_find_latest_valid_arc():
    """Test finding valid arc starting later in the segment"""
    welder = ArcWeld(tolerance=0.2, min_points=5)

    # First 10 points: noisy line
    invalid_points = [
        (x, 0.1 * np.random.rand()) for x in np.linspace(0, 10, 10)
    ]
    # Last 8 points: valid arc
    valid_points = generate_arc_points((5, 0), 2, 0, np.pi, 8)
    combined = invalid_points + valid_points  # Total 18 points

    # Start search at beginning of valid arc (index 10)
    best_arc, best_end = welder._find_longest_valid_arc(combined, 10)

    assert best_arc is not None
    assert best_end == 18  # Arc spans from index 10 to 18
    assert np.isclose(best_arc[1], 2.0, rtol=0.1)


def test_error_threshold_enforcement():
    """Test tolerance threshold rejects arcs with high fitting errors"""
    welder = ArcWeld(tolerance=0.05, min_points=5, max_points=25)

    # Generate points with guaranteed high error
    noisy_points = generate_arc_points((0, 0), 5, 0, np.pi / 2, 20)
    best_arc, best_end = welder._find_longest_valid_arc(noisy_points, 0)
    assert best_end == 20

    # Shift all points outward by 0.06mm (exceeding tolerance)
    point = noisy_points[10]
    noisy_points[10] = (point[0] + 0.25, point[1])

    best_arc, best_end = welder._find_longest_valid_arc(noisy_points, 0)
    assert best_end == 10


def test_radius_constraints():
    """Test radius range enforcement"""
    # Small radius test
    welder_small = ArcWeld(tolerance=0.1, min_points=5)
    small_points = generate_arc_points((0, 0), 0.5, 0, np.pi, 10)
    arc_small, _ = welder_small._find_longest_valid_arc(small_points, 0)
    assert arc_small is None  # Reject radius < 1mm

    # Large radius test
    welder_large = ArcWeld(tolerance=1.0, min_points=5)
    large_points = generate_arc_points((0, 0), 150, 0, 0.1, 10)  # Huge radius
    arc_large, _ = welder_large._find_longest_valid_arc(large_points, 0)
    assert arc_large is None  # Reject radius > 100mm


def test_find_longest_valid_arc_line_crosses_circle():
    welder = ArcWeld(tolerance=0.1, min_points=5)

    segment = [  # 5 points in a semi circle
        (5 * np.cos(theta), 5 * np.sin(theta))
        for theta in np.linspace(0, 0.7 * np.pi, 5)
    ]
    assert len(segment) == 5

    # 5 straight lines cross the circle in such a way that point 6
    # hits the circle on the other side. That point should not
    # be removed just because it happens to hit the new arc.
    segment += [(x, x * 0.1) for x in np.linspace(5, 10, 5)]
    assert len(segment) == 10

    arc, end = welder._find_longest_valid_arc(segment, 0)
    assert end == 5
