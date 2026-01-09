import pytest
from typing import NamedTuple
import numpy as np

from rayforge.core.geo import Geometry
from rayforge.core.geo.constants import (
    CMD_TYPE_MOVE,
    CMD_TYPE_LINE,
    CMD_TYPE_ARC,
    COL_TYPE,
)
from rayforge.core.geo.linearize import (
    linearize_arc,
    linearize_bezier,
    linearize_bezier_adaptive,
    resample_polyline,
    flatten_to_points,
    linearize_geometry,
)


class MockArc(NamedTuple):
    end: tuple[float, float, float]
    center_offset: tuple[float, float]
    clockwise: bool


@pytest.fixture
def sample_geometry():
    geo = Geometry()
    geo.move_to(0, 0)
    geo.line_to(10, 10)
    geo.arc_to(20, 0, i=5, j=-10)
    return geo


def test_linearize_arc(sample_geometry):
    """Tests the external linearize_arc function."""
    assert sample_geometry.data is not None
    # The second command is a line_to(10,10), which is the start of the arc
    start_point = tuple(sample_geometry.data[1, 1:4])
    # The third command is the arc
    arc_row = sample_geometry.data[2]
    arc_cmd = MockArc(
        end=tuple(arc_row[1:4]),
        center_offset=(arc_row[4], arc_row[5]),
        clockwise=bool(arc_row[6]),
    )

    segments = linearize_arc(arc_cmd, start_point)

    # Check that linearization produces a reasonable number of segments
    assert len(segments) >= 2

    # Check that the start and end points of the chain of segments match
    # the original arc's start and end points.
    first_segment_start, _ = segments[0]
    _, last_segment_end = segments[-1]

    assert first_segment_start == pytest.approx(start_point)
    assert last_segment_end == pytest.approx(arc_cmd.end)


def test_linearize_arc_full_circle():
    """Tests that linearizing a full circle (coincident start/end) works."""
    start_point = (10.0, 0.0, 5.0)
    # For a full circle, end is the same as start
    arc_cmd = MockArc(
        end=start_point,
        center_offset=(-10.0, 0.0),  # Center is at (0,0,z)
        clockwise=False,  # CCW
    )

    segments = linearize_arc(arc_cmd, start_point, resolution=1.0)

    # For a circle of radius 10, circumference is ~62.8.
    # With resolution 1.0, expect ~62 segments.
    assert len(segments) > 50

    # Check start and end points of the chain
    first_segment_start, _ = segments[0]
    _, last_segment_end = segments[-1]

    # Both should be very close to the original start/end point
    assert first_segment_start == pytest.approx(start_point)
    assert last_segment_end == pytest.approx(start_point, abs=1e-6)

    # Check the point halfway through the linearization
    mid_segment_idx = len(segments) // 2
    _, mid_point = segments[mid_segment_idx - 1]

    # Halfway around a CCW circle from (10,0) is (-10,0). Z should be the same.
    expected_mid_point = (-10.0, 0.0, 5.0)
    assert mid_point == pytest.approx(expected_mid_point, abs=0.1)


def test_linearize_bezier_3d():
    """Tests linearization of a 3D Bézier curve."""
    p0 = (0.0, 0.0, 0.0)
    c1 = (1.0, 1.0, 5.0)
    c2 = (2.0, 1.0, 5.0)
    p1 = (3.0, 0.0, 10.0)
    num_steps = 10

    segments = linearize_bezier(p0, c1, c2, p1, num_steps)
    assert len(segments) == num_steps

    start_of_chain = segments[0][0]
    end_of_chain = segments[-1][1]

    assert start_of_chain == pytest.approx(p0)
    assert end_of_chain == pytest.approx(p1)

    # Check midpoint (t=0.5), which is the start of the 6th segment
    # (or end of the 5th)
    midpoint = segments[5][0]
    # B(0.5) = 0.125*p0 + 0.375*c1 + 0.375*c2 + 0.125*p1
    expected_x = 0.125 * 0 + 0.375 * 1 + 0.375 * 2 + 0.125 * 3  # 1.5
    expected_y = 0.125 * 0 + 0.375 * 1 + 0.375 * 1 + 0.125 * 0  # 0.75
    expected_z = 0.125 * 0 + 0.375 * 5 + 0.375 * 5 + 0.125 * 10  # 5.0
    assert midpoint == pytest.approx((expected_x, expected_y, expected_z))


def test_linearize_bezier_adaptive_flat():
    """Tests adaptive linearization on a perfectly flat line."""
    p0 = (0.0, 0.0)
    c1 = (2.5, 0.0)  # Collinear control points
    c2 = (7.5, 0.0)
    p1 = (10.0, 0.0)

    # With a reasonable tolerance, this should result in exactly 1 segment
    points = linearize_bezier_adaptive(p0, c1, c2, p1, tolerance_sq=0.01)

    # Should contain only the end point
    assert len(points) == 1
    assert points[0] == p1


def test_linearize_bezier_adaptive_curved():
    """Tests adaptive linearization on a curve."""
    p0 = (0.0, 0.0)
    c1 = (0.0, 10.0)
    c2 = (10.0, 10.0)
    p1 = (10.0, 0.0)

    # High tolerance = fewer points
    points_coarse = linearize_bezier_adaptive(p0, c1, c2, p1, tolerance_sq=1.0)
    # Low tolerance = more points
    points_fine = linearize_bezier_adaptive(
        p0, c1, c2, p1, tolerance_sq=0.0001
    )

    assert len(points_fine) > len(points_coarse)
    assert points_fine[-1] == p1
    assert points_coarse[-1] == p1


def test_resample_polyline_open_path():
    points = [(0.0, 0.0, 1.0), (10.0, 0.0, 1.0)]
    resampled = resample_polyline(points, 2.0, is_closed=False)
    assert len(resampled) == 6  # 1 start + 4 new + 1 end
    assert resampled[0] == (0.0, 0.0, 1.0)
    assert resampled[-1] == (10.0, 0.0, 1.0)
    assert resampled[1] == pytest.approx((2.0, 0.0, 1.0))


def test_resample_polyline_closed_path():
    points = [
        (0.0, 0.0, 2.0),
        (10.0, 0.0, 2.0),
        (10.0, 10.0, 2.0),
        (0.0, 10.0, 2.0),
    ]
    resampled = resample_polyline(points, 5.0, is_closed=True)
    # 4 segments of length 10. Each needs 1 new point. 4 original + 4 new = 8
    assert len(resampled) == 8
    assert resampled[0] == (0.0, 0.0, 2.0)
    # The path should not have the duplicated end point that a closed geo has
    assert resampled[-1] != resampled[0]
    # Check that one of the new points is correct
    assert (5.0, 0.0, 2.0) in resampled


def test_flatten_to_points():
    """Tests flatten_to_points function."""
    geo = Geometry()
    geo.move_to(0, 0)
    geo.line_to(10, 0)
    geo.arc_to(10, 10, i=5, j=-5, clockwise=False)
    geo.bezier_to(5, 15, c1x=2, c1y=5, c2x=8, c2y=10)

    geo._sync_to_numpy()
    data = geo.data

    result = flatten_to_points(data, 0.1)

    # Should return 1 subpath (one for the move command)
    assert len(result) == 1

    # First subpath should have many points due to bezier linearization
    assert len(result[0]) > 4

    # Check some point values
    assert result[0][0] == (0.0, 0.0, 0.0)
    assert result[0][1] == (10.0, 0.0, 0.0)


def test_flatten_to_points_empty():
    """Tests flatten_to_points with empty geometry."""
    result = flatten_to_points(None, 0.1)
    assert result == []


def test_linearize_geometry():
    """Tests the linearize_geometry function."""
    geo = Geometry()
    geo.move_to(0, 0)
    geo.arc_to(10, 10, i=10, j=0, clockwise=False)

    geo._sync_to_numpy()
    data = geo.data

    result = linearize_geometry(data, tolerance=0.1)

    # Should contain only MOVE and LINE commands
    cmd_types = result[:, COL_TYPE]
    assert CMD_TYPE_ARC not in cmd_types
    assert CMD_TYPE_MOVE in cmd_types
    assert CMD_TYPE_LINE in cmd_types

    # The end point should still be (10, 10)
    end_point = result[-1, 1:4]
    np.testing.assert_allclose(end_point, (10.0, 10.0, 0.0), atol=1e-6)


def test_linearize_geometry_empty():
    """Tests linearize_geometry with empty data."""
    result = linearize_geometry(None, 0.1)
    assert result.shape == (0, 8)
