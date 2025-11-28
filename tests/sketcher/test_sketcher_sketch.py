import pytest
from rayforge.core.sketcher.sketch import Sketch
from rayforge.core.geo import ArcToCommand, Geometry
from rayforge.core.sketcher.constraints import (
    EqualDistanceConstraint,
    PointOnLineConstraint,
    PerpendicularConstraint,
    EqualLengthConstraint,
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


def test_sketch_circle_workflow():
    s = Sketch()
    s.set_param("diam", 20.0)

    center = s.add_point(10, 10, fixed=True)
    radius_pt = s.add_point(15, 10)  # Initial radius is 5
    circ_id = s.add_circle(center, radius_pt)

    s.constrain_diameter(circ_id, "diam")

    assert s.solve() is True

    # After solve, radius should be 10, diameter 20.
    # The radius point should be 10 units away from center.
    p = s.registry.get_point(radius_pt)
    c = s.registry.get_point(center)
    dist = ((p.x - c.x) ** 2 + (p.y - c.y) ** 2) ** 0.5
    assert dist == pytest.approx(10.0)

    geo = s.to_geometry()
    # Should export as two semi-circles -> 2 ArcToCommands
    assert isinstance(geo, Geometry)
    arcs = [cmd for cmd in geo.commands if isinstance(cmd, ArcToCommand)]
    assert len(arcs) == 2


def test_sketch_equal_length_workflow():
    """Test a full workflow using an equal length constraint."""
    s = Sketch()

    # Line 1 will be fixed at length 10
    p1 = s.add_point(0, 0, fixed=True)
    p2 = s.add_point(10, 0)
    l1 = s.add_line(p1, p2)
    s.constrain_horizontal(p1, p2)
    s.constrain_distance(p1, p2, 10.0)

    # Line 2 will start at length 5 and should be solved to 10
    p3 = s.add_point(20, 0, fixed=True)
    p4 = s.add_point(25, 0)
    l2 = s.add_line(p3, p4)
    s.constrain_horizontal(p3, p4)

    # Apply the Equal Length constraint
    s.constrain_equal_length([l1, l2])

    assert s.solve() is True

    # Check that p4 has moved to make line 2 have length 10
    pt4 = s.registry.get_point(p4)
    pt3 = s.registry.get_point(p3)
    dist = ((pt4.x - pt3.x) ** 2 + (pt4.y - pt3.y) ** 2) ** 0.5
    assert dist == pytest.approx(10.0)
    assert pt4.x == pytest.approx(30.0)
    assert pt4.y == pytest.approx(0.0)


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
    c = s.add_point(5, 5)
    l1 = s.add_line(p1, p2)
    l2 = s.add_line(p3, p4)
    circ = s.add_circle(c, p1)

    # Call shortcuts not covered in main workflow test
    s.constrain_equal_distance(p1, p2, p3, p4)
    s.constrain_coincident(p1, p3)
    s.constrain_point_on_line(p3, l1)
    s.constrain_perpendicular(l1, l2)
    s.constrain_diameter(circ, 20.0)
    s.constrain_equal_length([l1, circ])

    assert len(s.constraints) == 6
    assert isinstance(s.constraints[0], EqualDistanceConstraint)
    assert isinstance(s.constraints[2], PointOnLineConstraint)
    assert isinstance(s.constraints[3], PerpendicularConstraint)
    assert isinstance(s.constraints[5], EqualLengthConstraint)


def test_sketch_serialization_round_trip():
    """Tests to_dict and from_dict for a complete Sketch."""
    s = Sketch()
    s.set_param("width", 50.0)
    p1 = s.origin_id
    p2 = s.add_point(40, 0)
    p3 = s.add_point(40, 20)
    l1 = s.add_line(p1, p2)
    l2 = s.add_line(p2, p3)
    circ = s.add_circle(p1, p3)

    s.constrain_horizontal(p1, p2)
    s.constrain_distance(p1, p2, "width")
    s.constrain_perpendicular(l1, l2)
    # Since p3.x will be 50, the radius must be
    # >= 50, so the diameter must be >= 100.
    s.constrain_diameter(circ, 100.0)
    s.solve()

    # Serialize
    data = s.to_dict()

    # Deserialize
    new_sketch = Sketch.from_dict(data)

    # Validate integrity
    assert new_sketch.params.get("width") == 50.0
    assert len(new_sketch.registry.points) == 3
    assert len(new_sketch.registry.entities) == 3
    assert len(new_sketch.constraints) == 4
    assert new_sketch.origin_id == p1

    # Validate functional equivalence
    assert new_sketch.solve() is True
    pt2 = new_sketch.registry.get_point(p2)
    pt3 = new_sketch.registry.get_point(p3)
    radius = (pt3.x**2 + pt3.y**2) ** 0.5
    diameter = radius * 2

    abs_tol = 1e-7
    assert pt2.x == pytest.approx(50.0, abs=abs_tol)
    assert pt2.y == pytest.approx(0.0, abs=abs_tol)
    # Perpendicular constraint should make p3.x == p2.x
    assert pt3.x == pytest.approx(50.0, abs=abs_tol)
    assert diameter == pytest.approx(100.0, abs=abs_tol)


@pytest.fixture
def setup_sketch_for_validation():
    """Provides a sketch with a variety of geometry for validation tests."""
    s = Sketch()
    # Points
    p1 = s.add_point(0, 0)
    p2 = s.add_point(10, 0)
    p3 = s.add_point(0, 10)
    p_ext = s.add_point(5, 5)
    # Entities
    l1 = s.add_line(p1, p2)
    l2 = s.add_line(p1, p3)
    # Arc points are separate to avoid endpoint conflicts
    arc_s = s.add_point(20, 0)
    arc_e = s.add_point(0, 20)
    arc_c = s.add_point(0, 0)
    a1 = s.add_arc(arc_s, arc_e, arc_c)
    c1 = s.add_circle(p1, p2)

    return s, {
        "p1": p1,
        "p2": p2,
        "p3": p3,
        "p_ext": p_ext,
        "l1": l1,
        "l2": l2,
        "a1": a1,
        "c1": c1,
    }


def test_sketch_supports_constraint(setup_sketch_for_validation):
    """Tests the logic of the `supports_constraint` method."""
    s, ids = setup_sketch_for_validation
    p1, p2, p3, p_ext = ids["p1"], ids["p2"], ids["p3"], ids["p_ext"]
    l1, l2, a1, c1 = ids["l1"], ids["l2"], ids["a1"], ids["c1"]

    # Test "dist", "horiz", "vert"
    for c_type in ("dist", "horiz", "vert"):
        # Valid cases
        assert s.supports_constraint(c_type, [p1, p2], []) is True
        assert s.supports_constraint(c_type, [], [l1]) is True
        # Invalid cases
        assert s.supports_constraint(c_type, [p1], []) is False
        assert s.supports_constraint(c_type, [p1, p2, p3], []) is False
        assert s.supports_constraint(c_type, [p1], [l1]) is False
        assert s.supports_constraint(c_type, [], [a1]) is False
        assert s.supports_constraint(c_type, [], [l1, l2]) is False

    # Test "radius"
    assert s.supports_constraint("radius", [], [a1]) is True
    assert s.supports_constraint("radius", [], [c1]) is True
    assert s.supports_constraint("radius", [], [l1]) is False
    assert s.supports_constraint("radius", [p1], [a1]) is False
    assert s.supports_constraint("radius", [], [a1, c1]) is False

    # Test "diameter"
    assert s.supports_constraint("diameter", [], [c1]) is True
    assert s.supports_constraint("diameter", [], [a1]) is False
    assert s.supports_constraint("diameter", [], [l1]) is False
    assert s.supports_constraint("diameter", [p1], [c1]) is False

    # Test "perp"
    assert s.supports_constraint("perp", [], [l1, l2]) is True
    assert s.supports_constraint("perp", [], [l1]) is False
    assert s.supports_constraint("perp", [], [l1, a1]) is False
    assert s.supports_constraint("perp", [], [l1, l2, l1]) is False

    # Test "tangent"
    assert s.supports_constraint("tangent", [], [l1, a1]) is True
    assert s.supports_constraint("tangent", [], [l1, c1]) is True
    assert s.supports_constraint("tangent", [], [l1, l2]) is False
    assert s.supports_constraint("tangent", [], [a1, c1]) is False
    assert s.supports_constraint("tangent", [], [l1]) is False
    assert s.supports_constraint("tangent", [], [a1]) is False

    # Test "equal"
    assert s.supports_constraint("equal", [], [l1, l2]) is True
    assert s.supports_constraint("equal", [], [l1, a1]) is True
    assert s.supports_constraint("equal", [], [l1, c1, a1]) is True
    assert s.supports_constraint("equal", [], [l1]) is False  # Needs >= 2
    assert s.supports_constraint("equal", [p1], [l1, l2]) is False
    assert s.supports_constraint("equal", [], []) is False

    # Test "align" (covers coincident and point-on-line)
    # Coincident (2 points)
    assert s.supports_constraint("align", [p1, p_ext], []) is True
    # Point-on-Shape (1 point, 1 shape)
    assert s.supports_constraint("align", [p_ext], [l1]) is True
    assert s.supports_constraint("align", [p_ext], [a1]) is True
    assert s.supports_constraint("align", [p_ext], [c1]) is True

    # Invalid: Endpoint on its own line
    assert s.supports_constraint("align", [p1], [l1]) is False
    # Invalid: Control point on its own shape
    arc_start_id = s.registry.get_entity(a1).start_idx
    assert s.supports_constraint("align", [arc_start_id], [a1]) is False
    circle_center_id = s.registry.get_entity(c1).center_idx
    assert s.supports_constraint("align", [circle_center_id], [c1]) is False

    # Invalid: Other combos
    assert s.supports_constraint("align", [p1, p2, p3], []) is False
    assert s.supports_constraint("align", [p_ext], [l1, l2]) is False

    # Test "coincident" (internal use)
    assert s.supports_constraint("coincident", [p1, p2], []) is True
    assert s.supports_constraint("coincident", [p1], []) is False
    assert s.supports_constraint("coincident", [p1], [l1]) is False

    # Test "point_on_line" (internal use, now means point-on-shape)
    assert s.supports_constraint("point_on_line", [p_ext], [l1]) is True
    assert s.supports_constraint("point_on_line", [p_ext], [a1]) is True
    assert s.supports_constraint("point_on_line", [p_ext], [c1]) is True
    # Invalid: Endpoint on its own line
    assert s.supports_constraint("point_on_line", [p1], [l1]) is False
    # Invalid: wrong number of items
    assert s.supports_constraint("point_on_line", [p1, p_ext], [l1]) is False
    assert s.supports_constraint("point_on_line", [p_ext], [l1, l2]) is False
