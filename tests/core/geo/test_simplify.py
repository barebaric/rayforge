from rayforge.core.geo import (
    Geometry,
    MoveToCommand,
    LineToCommand,
    ArcToCommand,
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

    assert len(simplified.commands) == 2
    assert isinstance(simplified.commands[0], MoveToCommand)
    assert simplified.commands[0].end == (0, 0, 0)
    assert isinstance(simplified.commands[1], LineToCommand)
    assert simplified.commands[1].end == (10, 10, 0)


def test_simplify_significant_corner():
    """Tests that points forming a corner > tolerance are kept."""
    geo = Geometry()
    geo.move_to(0, 0)
    geo.line_to(5, 5)  # Corner
    geo.line_to(10, 0)

    # Tolerance is smaller than the height of the triangle (5)
    simplified = geo.simplify(tolerance=1.0)

    assert len(simplified.commands) == 3
    assert simplified.commands[1].end == (5, 5, 0)


def test_simplify_insignificant_bump():
    """Tests that a small bump within tolerance is removed."""
    geo = Geometry()
    geo.move_to(0, 0)
    geo.line_to(5, 0.1)  # Small deviation
    geo.line_to(10, 0)

    # Tolerance (0.5) > Deviation (0.1), so middle point is removed
    simplified = geo.simplify(tolerance=0.5)

    assert len(simplified.commands) == 2
    assert simplified.commands[1].end == (10, 0, 0)


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

    # Expected:
    # 1. MoveTo(0,0)
    # 2. LineTo(2,0)  <- The end of the first linear chain
    # 3. ArcTo(...)   <- Preserved
    # 4. LineTo(6,0)  <- The end of the second linear chain

    assert len(simplified.commands) == 4
    assert isinstance(simplified.commands[2], ArcToCommand)
    assert simplified.commands[1].end == (2, 0, 0)
    assert simplified.commands[3].end == (6, 0, 0)


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

    assert len(simplified.commands) == 4
    assert isinstance(simplified.commands[0], MoveToCommand)
    assert isinstance(simplified.commands[1], LineToCommand)
    assert isinstance(simplified.commands[2], MoveToCommand)
    assert isinstance(simplified.commands[3], LineToCommand)

    assert simplified.commands[1].end == (10, 0, 0)
    assert simplified.commands[3].end == (30, 0, 0)


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

    # RDP on a closed loop treats it as a polyline from start to end.
    # For a square (0,0)..(10,0)..(10,10)..(0,10)..(0,0):
    # The max deviation from the baseline (0,0)->(0,0) is at (10,10), so it
    # splits there.
    # Recursion 1: (0,0)..(10,10). Baseline is diagonal. Corner (10,0) is far.
    # Recursion 2: (10,10)..(0,0). Baseline is diagonal. Corner (0,10) is far.
    # Result: All 4 corners preserved.

    assert len(simplified.commands) == 5  # Move + 4 Lines
    points = [cmd.end for cmd in simplified.commands]
    assert (10, 0, 0) in points
    assert (10, 10, 0) in points
    assert (0, 10, 0) in points
    assert (0, 0, 0) in points
