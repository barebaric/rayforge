import pytest
from typing import cast

from rayforge.core.geo import Geometry, LineToCommand, ArcToCommand
from rayforge.core.geo.linearize import linearize_arc


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
