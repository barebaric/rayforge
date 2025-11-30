import pytest
from typing import cast

from rayforge.core.geo import Geometry, LineToCommand, ArcToCommand
from rayforge.core.geo.linearize import (
    linearize_arc,
    linearize_bezier,
    resample_polyline,
)


@pytest.fixture
def sample_geometry():
    geo = Geometry()
    geo.move_to(0, 0)
    geo.line_to(10, 10)
    geo.arc_to(20, 0, i=5, j=-10)
    return geo


def test_linearize_arc(sample_geometry):
    """Tests the external linearize_arc function."""
    # The second command is a line_to(10,10), which is the start of the arc
    start_point = cast(LineToCommand, sample_geometry.commands[1]).end
    # The third command is the arc
    arc_cmd = cast(ArcToCommand, sample_geometry.commands[2])

    segments = linearize_arc(arc_cmd, start_point)

    # Check that linearization produces a reasonable number of segments
    assert len(segments) >= 2

    # Check that the start and end points of the chain of segments match
    # the original arc's start and end points.
    first_segment_start, _ = segments[0]
    _, last_segment_end = segments[-1]

    assert first_segment_start == pytest.approx(start_point)
    assert last_segment_end == pytest.approx(arc_cmd.end)


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
