import pytest
from rayforge.core.sketcher.sketch import Sketch
from rayforge.core.geo import ArcToCommand, Geometry
from rayforge.core.sketcher.constraints import (
    EqualDistanceConstraint,
    PointOnLineConstraint,
    PerpendicularConstraint,
)


def test_sketch_workflow():
    s = Sketch()

    # 1. Define params
    s.set_param("side", 10.0)

    # 2. Add geometry (approximate square)
    p1 = s.add_point(0, 0, fixed=True)
    p2 = s.add_point(5, 0)
    p3 = s.add_point(5, 5)
    p4 = s.add_point(0, 5)

    s.add_line(p1, p2)
    s.add_line(p2, p3)
    s.add_line(p3, p4)
    s.add_line(p4, p1)

    # 3. Constrain
    s.constrain_horizontal(p1, p2)
    s.constrain_vertical(p2, p3)
    s.constrain_horizontal(p4, p3)
    s.constrain_vertical(p1, p4)

    s.constrain_distance(p1, p2, "side")
    s.constrain_distance(p2, p3, "side")

    # 4. Solve
    assert s.solve() is True

    # 5. Check geometry export
    geo = s.to_geometry()
    assert isinstance(geo, Geometry)
    assert len(geo.commands) == 8  # 4 moves + 4 lines (simple export)

    # Check bounding box is approx 10x10
    min_x, min_y, max_x, max_y = geo.rect()
    assert min_x == pytest.approx(0.0)
    assert min_y == pytest.approx(0.0)
    assert max_x == pytest.approx(10.0, abs=1e-4)
    assert max_y == pytest.approx(10.0, abs=1e-4)


def test_sketch_arc_export():
    s = Sketch()
    # Simple quarter circle arc
    p1 = s.add_point(10, 0)  # Start
    p2 = s.add_point(0, 10)  # End
    c = s.add_point(0, 0)  # Center

    s.add_arc(p1, p2, c, clockwise=False)

    geo = s.to_geometry()

    # Should contain a MoveTo(10,0) and ArcTo(0,10)
    # Filter for ArcTo
    arcs = [cmd for cmd in geo.commands if isinstance(cmd, ArcToCommand)]
    assert len(arcs) == 1

    arc = arcs[0]
    assert arc.end == (0.0, 10.0, 0.0)
    # Check offsets. Center (0,0) relative to Start (10,0) is (-10, 0)
    assert arc.center_offset == (-10.0, 0.0)
    assert arc.clockwise is False


def test_sketch_parameter_updates():
    """Test that changing a parameter and re-solving updates geometry."""
    s = Sketch()
    s.set_param("len", 10.0)

    p1 = s.add_point(0, 0, fixed=True)
    p2 = s.add_point(5, 0)
    s.constrain_distance(p1, p2, "len")

    # Solver should now work even with 1 constraint vs 2 variables (trf method)
    assert s.solve() is True
    assert s.registry.get_point(p2).x == pytest.approx(10.0)

    # Change param
    s.set_param("len", 20.0)
    s.solve()
    assert s.registry.get_point(p2).x == pytest.approx(20.0)


def test_sketch_constraint_shortcuts():
    """Verify all constraint shortcut methods properly register constraints."""
    s = Sketch()
    p1 = s.add_point(0, 0)
    p2 = s.add_point(10, 0)
    p3 = s.add_point(0, 10)
    p4 = s.add_point(0, 14)
    l1 = s.add_line(p1, p2)
    l2 = s.add_line(p3, p4)

    # Call shortcuts not covered in main workflow test
    s.constrain_equal_distance(p1, p2, p3, p4)
    s.constrain_coincident(p1, p3)
    s.constrain_point_on_line(p3, l1)
    s.constrain_perpendicular(l1, l2)

    # Check if they were added to the list (4 added here)
    # Since this is a new sketch, constraints list should be length 4
    assert len(s.constraints) == 4
    assert isinstance(s.constraints[0], EqualDistanceConstraint)
    assert isinstance(s.constraints[2], PointOnLineConstraint)
    assert isinstance(s.constraints[3], PerpendicularConstraint)
