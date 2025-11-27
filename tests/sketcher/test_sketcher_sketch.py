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


def test_sketch_serialization_round_trip():
    """Tests to_dict and from_dict for a complete Sketch."""
    s = Sketch()
    s.set_param("width", 50.0)
    p1 = s.origin_id
    p2 = s.add_point(40, 0)
    p3 = s.add_point(40, 20)
    l1 = s.add_line(p1, p2)
    l2 = s.add_line(p2, p3)
    s.constrain_horizontal(p1, p2)
    s.constrain_distance(p1, p2, "width")
    s.constrain_perpendicular(l1, l2)
    s.solve()

    # Serialize
    data = s.to_dict()

    # Deserialize
    new_sketch = Sketch.from_dict(data)

    # Validate integrity
    assert new_sketch.params.get("width") == 50.0
    assert len(new_sketch.registry.points) == 3
    assert len(new_sketch.registry.entities) == 2
    assert len(new_sketch.constraints) == 3  # Now has 3 constraints
    assert new_sketch.origin_id == p1

    # Validate functional equivalence
    assert new_sketch.solve() is True
    pt2 = new_sketch.registry.get_point(p2)
    pt3 = new_sketch.registry.get_point(p3)

    abs_tol = 1e-7
    assert pt2.x == pytest.approx(50.0, abs=abs_tol)
    assert pt2.y == pytest.approx(0.0, abs=abs_tol)
    # Perpendicular constraint should make p3.x == p2.x
    assert pt3.x == pytest.approx(50.0, abs=abs_tol)


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

    return s, {
        "p1": p1,
        "p2": p2,
        "p3": p3,
        "p_ext": p_ext,
        "l1": l1,
        "l2": l2,
        "a1": a1,
    }


def test_sketch_supports_constraint(setup_sketch_for_validation):
    """Tests the logic of the `supports_constraint` method."""
    s, ids = setup_sketch_for_validation
    p1, p2, p3, p_ext = ids["p1"], ids["p2"], ids["p3"], ids["p_ext"]
    l1, l2, a1 = ids["l1"], ids["l2"], ids["a1"]

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
    assert s.supports_constraint("radius", [], [l1]) is False
    assert s.supports_constraint("radius", [p1], [a1]) is False
    assert s.supports_constraint("radius", [], [a1, a1]) is False

    # Test "perp"
    assert s.supports_constraint("perp", [], [l1, l2]) is True
    assert s.supports_constraint("perp", [], [l1]) is False
    assert s.supports_constraint("perp", [], [l1, a1]) is False
    assert s.supports_constraint("perp", [], [l1, l2, l1]) is False

    # Test "tangent"
    assert s.supports_constraint("tangent", [], [l1, a1]) is True
    assert s.supports_constraint("tangent", [], [l1, l2]) is False
    assert s.supports_constraint("tangent", [], [a1, a1]) is False
    assert s.supports_constraint("tangent", [], [l1]) is False
    assert s.supports_constraint("tangent", [], [a1]) is False

    # Test "align" (covers coincident and point-on-line)
    # Coincident (2 points)
    assert s.supports_constraint("align", [p1, p_ext], []) is True
    # Point-on-Line (1 point, 1 line)
    assert s.supports_constraint("align", [p_ext], [l1]) is True
    # Invalid: Endpoint on its own line
    assert s.supports_constraint("align", [p1], [l1]) is False
    # Invalid: Other combos
    assert s.supports_constraint("align", [p1, p2, p3], []) is False
    assert s.supports_constraint("align", [p_ext], [l1, l2]) is False
    assert s.supports_constraint("align", [p_ext], [a1]) is False

    # Test "coincident" (internal use)
    assert s.supports_constraint("coincident", [p1, p2], []) is True
    assert s.supports_constraint("coincident", [p1], []) is False
    assert s.supports_constraint("coincident", [p1], [l1]) is False

    # Test "point_on_line" (internal use)
    assert s.supports_constraint("point_on_line", [p_ext], [l1]) is True
    # Invalid: Endpoint on its own line
    assert s.supports_constraint("point_on_line", [p1], [l1]) is False
    # Invalid: wrong number of items
    assert s.supports_constraint("point_on_line", [p1, p_ext], [l1]) is False
    assert s.supports_constraint("point_on_line", [p_ext], [l1, l2]) is False
