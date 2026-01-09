import numpy as np
from rayforge.core.geo import Geometry
from rayforge.core.geo.constants import (
    CMD_TYPE_MOVE,
    CMD_TYPE_LINE,
    CMD_TYPE_ARC,
    COL_TYPE,
)


def test_simplify_straight_line():
    """Tests that collinear points on a straight line are removed."""
    geo = Geometry()
    geo.move_to(0, 0)
    geo.line_to(1, 1)
    geo.line_to(2, 2)
    geo.line_to(3, 3)
    geo.line_to(10, 10)

    # With a tiny tolerance, all intermediate points on a perfect line
    # should go
    simplified = geo.simplify(tolerance=0.001)
    simplified._sync_to_numpy()
    assert simplified.data is not None

    assert len(simplified) == 2
    assert simplified.data[0, COL_TYPE] == CMD_TYPE_MOVE
    assert np.all(simplified.data[0, 1:4] == (0, 0, 0))
    assert simplified.data[1, COL_TYPE] == CMD_TYPE_LINE
    assert np.all(simplified.data[1, 1:4] == (10, 10, 0))


def test_simplify_significant_corner():
    """Tests that points forming a corner > tolerance are kept."""
    geo = Geometry()
    geo.move_to(0, 0)
    geo.line_to(5, 5)  # Corner
    geo.line_to(10, 0)

    # Tolerance is smaller than the height of the triangle (5)
    simplified = geo.simplify(tolerance=1.0)
    assert simplified.data is not None

    assert len(simplified) == 3
    assert np.all(simplified.data[1, 1:4] == (5, 5, 0))


def test_simplify_insignificant_bump():
    """Tests that a small bump within tolerance is removed."""
    geo = Geometry()
    geo.move_to(0, 0)
    geo.line_to(5, 0.1)  # Small deviation
    geo.line_to(10, 0)

    # Tolerance (0.5) > Deviation (0.1), so middle point is removed
    simplified = geo.simplify(tolerance=0.5)
    assert simplified.data is not None

    assert len(simplified) == 2
    assert np.all(simplified.data[1, 1:4] == (10, 0, 0))


def test_simplify_preserves_arcs():
    """Tests that simplification does not decimate arcs."""
    geo = Geometry()
    geo.move_to(0, 0)
    geo.line_to(1, 0)
    geo.line_to(2, 0)  # Collinear, should be removed
    geo.arc_to(4, 0, 1, 0)  # Arc from 2,0 to 4,0
    geo.line_to(5, 0)
    geo.line_to(6, 0)  # Collinear, should be removed

    simplified = geo.simplify(tolerance=0.1)
    assert simplified.data is not None

    # Expected:
    # 1. MoveTo(0,0)
    # 2. LineTo(2,0)  <- The end of the first linear chain
    # 3. ArcTo(...)   <- Preserved
    # 4. LineTo(6,0)  <- The end of the second linear chain

    assert len(simplified) == 4
    assert simplified.data[2, COL_TYPE] == CMD_TYPE_ARC
    assert np.all(simplified.data[1, 1:4] == (2, 0, 0))
    assert np.all(simplified.data[3, 1:4] == (6, 0, 0))


def test_simplify_preserves_moveto_breaks():
    """Tests that a MoveTo command breaks the simplification chain."""
    geo = Geometry()
    # Chain 1
    geo.move_to(0, 0)
    geo.line_to(5, 0)
    geo.line_to(10, 0)  # Simplifies to (0,0)->(10,0)

    # Chain 2 (disjoint)
    geo.move_to(20, 0)
    geo.line_to(25, 0)
    geo.line_to(30, 0)  # Simplifies to (20,0)->(30,0)

    simplified = geo.simplify(tolerance=0.1)
    assert simplified.data is not None

    assert len(simplified) == 4
    assert simplified.data[0, COL_TYPE] == CMD_TYPE_MOVE
    assert simplified.data[1, COL_TYPE] == CMD_TYPE_LINE
    assert simplified.data[2, COL_TYPE] == CMD_TYPE_MOVE
    assert simplified.data[3, COL_TYPE] == CMD_TYPE_LINE

    assert np.all(simplified.data[1, 1:4] == (10, 0, 0))
    assert np.all(simplified.data[3, 1:4] == (30, 0, 0))


def test_simplify_closed_shape():
    """Tests simplification on a closed rectangle with redundant points."""
    geo = Geometry()
    geo.move_to(0, 0)
    geo.line_to(5, 0)  # Redundant
    geo.line_to(10, 0)  # Corner
    geo.line_to(10, 5)  # Redundant
    geo.line_to(10, 10)  # Corner
    geo.line_to(5, 10)  # Redundant
    geo.line_to(0, 10)  # Corner
    geo.close_path()  # LineTo(0,0)

    simplified = geo.simplify(tolerance=0.1)
    assert simplified.data is not None

    assert len(simplified) == 5  # Move + 4 Lines
    points = [tuple(row[1:4]) for row in simplified.data]
    assert (10, 0, 0) in points
    assert (10, 10, 0) in points
    assert (0, 10, 0) in points
    assert (0, 0, 0) in points


def test_simplify_empty_geometry():
    """Tests that an empty geometry is handled gracefully."""
    geo = Geometry()
    simplified = geo.simplify(tolerance=0.1)
    assert len(simplified) == 0


def test_simplify_single_segment():
    """Tests that a single segment (2 points) is not reduced."""
    geo = Geometry()
    geo.move_to(0, 0)
    geo.line_to(10, 10)

    # Even with huge tolerance, start and end must be preserved
    simplified = geo.simplify(tolerance=100.0)
    assert simplified.data is not None

    assert len(simplified) == 2
    assert np.all(simplified.data[0, 1:4] == (0, 0, 0))
    assert np.all(simplified.data[1, 1:4] == (10, 10, 0))


def test_simplify_z_axis_preservation():
    """Tests that Z coordinates are preserved even if RDP is 2D based."""
    geo = Geometry()
    geo.move_to(0, 0, 1)
    geo.line_to(5, 0, 2)  # Collinear in XY, but has Z
    geo.line_to(10, 0, 3)

    simplified = geo.simplify(tolerance=0.1)
    assert simplified.data is not None

    assert len(simplified) == 2
    assert np.all(simplified.data[0, 1:4] == (0, 0, 1))
    assert np.all(simplified.data[1, 1:4] == (10, 0, 3))


def test_simplify_zigzag_removal():
    """Tests removal of high frequency zigzag noise within tolerance."""
    geo = Geometry()
    geo.move_to(0, 0)
    # Zigzag with amplitude 0.05
    for x in range(1, 10):
        y = 0.05 if x % 2 else -0.05
        geo.line_to(x, y)
    geo.line_to(10, 0)

    # Tolerance 0.1 > Amplitude 0.05 -> should result in straight line
    simplified = geo.simplify(tolerance=0.1)
    assert simplified.data is not None

    assert len(simplified) == 2
    assert np.all(simplified.data[0, 1:4] == (0, 0, 0))
    assert np.all(simplified.data[1, 1:4] == (10, 0, 0))


def test_simplify_duplicate_points():
    """Tests that consecutive duplicate points are removed/handled."""
    geo = Geometry()
    geo.move_to(0, 0)
    geo.line_to(0, 0)  # Duplicate
    geo.line_to(10, 10)
    geo.line_to(10, 10)  # Duplicate

    simplified = geo.simplify(tolerance=0.001)
    assert simplified.data is not None

    assert len(simplified) == 2
    assert np.all(simplified.data[0, 1:4] == (0, 0, 0))
    assert np.all(simplified.data[1, 1:4] == (10, 10, 0))
