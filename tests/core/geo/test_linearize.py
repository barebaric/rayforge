import numpy as np

from rayforge.core.geo import Geometry
from rayforge.core.geo.constants import (
    CMD_TYPE_MOVE,
    CMD_TYPE_LINE,
    CMD_TYPE_ARC,
    COL_TYPE,
)
from rayforge.core.geo.linearize import (
    flatten_to_points,
    linearize_geometry,
)


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
