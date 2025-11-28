import pytest
from rayforge.core.sketcher.params import ParameterContext
from rayforge.core.sketcher.entities import EntityRegistry
from rayforge.core.sketcher.constraints import (
    DistanceConstraint,
    HorizontalConstraint,
    VerticalConstraint,
    CoincidentConstraint,
    RadiusConstraint,
    DiameterConstraint,
    PerpendicularConstraint,
    TangentConstraint,
    DragConstraint,
    EqualDistanceConstraint,
    PointOnLineConstraint,
    EqualLengthConstraint,
)


@pytest.fixture
def setup_env():
    reg = EntityRegistry()
    params = ParameterContext()
    return reg, params


def test_distance_constraint(setup_env):
    reg, params = setup_env
    p1 = reg.add_point(0, 0)
    p2 = reg.add_point(10, 0)

    # Target is 10, actual is 10. Error should be 10^2 - 10^2 = 0.
    c = DistanceConstraint(p1, p2, 10.0)
    assert c.error(reg, params) == pytest.approx(0.0)

    # Target is 5, actual is 10. Error should be 10^2 - 5^2 = 75.
    c2 = DistanceConstraint(p1, p2, 5.0)
    assert c2.error(reg, params) == pytest.approx(75.0)


def test_distance_constraint_with_expression(setup_env):
    reg, params = setup_env
    params.set("width", 20.0)
    p1 = reg.add_point(0, 0)
    p2 = reg.add_point(10, 0)

    c = DistanceConstraint(p1, p2, "width")
    # Actual dist^2 is 100, Target dist^2 is 400. Error is 100 - 400 = -300.
    assert c.error(reg, params) == pytest.approx(-300.0)


def test_equal_distance_constraint(setup_env):
    reg, params = setup_env
    # Segment 1: Length 10 -> dist^2 = 100
    p1 = reg.add_point(0, 0)
    p2 = reg.add_point(10, 0)

    # Segment 2: Length 4 -> dist^2 = 16
    p3 = reg.add_point(0, 10)
    p4 = reg.add_point(0, 14)

    c = EqualDistanceConstraint(p1, p2, p3, p4)

    # Error should be 100 - 16 = 84
    assert c.error(reg, params) == pytest.approx(84.0)


def test_equal_length_constraint(setup_env):
    """Tests the error calculation for EqualLengthConstraint."""
    reg, params = setup_env
    # Line 1: length 10 -> len^2 = 100
    p1 = reg.add_point(0, 0)
    p2 = reg.add_point(10, 0)
    l1 = reg.add_line(p1, p2)

    # Line 2: length 5 -> len^2 = 25
    p3 = reg.add_point(0, 10)
    p4 = reg.add_point(5, 10)
    l2 = reg.add_line(p3, p4)

    # Arc 1: radius 4 -> rad^2 = 16
    c1_p = reg.add_point(20, 0)
    s1 = reg.add_point(24, 0)
    e1 = reg.add_point(20, 4)
    a1 = reg.add_arc(s1, e1, c1_p)

    # Circle 1: radius 3 -> rad^2 = 9
    c2_p = reg.add_point(30, 0)
    r2 = reg.add_point(33, 0)
    circ1 = reg.add_circle(c2_p, r2)

    # Test Line-Line (error is [len2^2 - len1^2])
    c = EqualLengthConstraint([l1, l2])
    assert c.error(reg, params) == pytest.approx([-75.0])

    # Test Line-Arc (error is [rad1^2 - len1^2])
    c2 = EqualLengthConstraint([l1, a1])
    assert c2.error(reg, params) == pytest.approx([-84.0])

    # Test Arc-Circle (error is [rad_circ^2 - rad_arc^2])
    c3 = EqualLengthConstraint([a1, circ1])
    assert c3.error(reg, params) == pytest.approx([-7.0])

    # Test multi-entity constraint
    c_multi = EqualLengthConstraint([l1, l2, a1, circ1])
    # Errors are [len2^2-len1^2, rad_arc^2-len1^2, rad_circ^2-len1^2]
    assert c_multi.error(reg, params) == pytest.approx([-75.0, -84.0, -91.0])

    # Test edge cases (no error for < 2 entities)
    assert EqualLengthConstraint([]).error(reg, params) == []
    assert EqualLengthConstraint([l1]).error(reg, params) == []


def test_horizontal_constraint(setup_env):
    reg, params = setup_env
    # p1 y=0, p2 y=5. Error should be -5 (p1.y - p2.y)
    p1 = reg.add_point(0, 0)
    p2 = reg.add_point(10, 5)

    c = HorizontalConstraint(p1, p2)
    assert c.error(reg, params) == pytest.approx(-5.0)


def test_vertical_constraint(setup_env):
    reg, params = setup_env
    # p1 x=0, p2 x=5. Error should be -5
    p1 = reg.add_point(0, 0)
    p2 = reg.add_point(5, 10)

    c = VerticalConstraint(p1, p2)
    assert c.error(reg, params) == pytest.approx(-5.0)


def test_coincident_constraint(setup_env):
    reg, params = setup_env
    p1 = reg.add_point(0, 0)
    p2 = reg.add_point(3, 4)

    c = CoincidentConstraint(p1, p2)
    # Error is a tuple (dx, dy)
    assert c.error(reg, params) == (-3.0, -4.0)


def test_point_on_line_constraint(setup_env):
    reg, params = setup_env

    # Line along X-axis: (0,0) -> (10,0)
    p1 = reg.add_point(0, 0)
    p2 = reg.add_point(10, 0)
    line_id = reg.add_line(p1, p2)

    # Point at (5, 5).
    # Error is twice the signed area of the triangle:
    # (10-0)*(5-0) - (5-0)*(0-0) = 50
    p3 = reg.add_point(5, 5)

    c = PointOnLineConstraint(p3, line_id)
    assert c.error(reg, params) == pytest.approx(50.0)

    # Move point to (5, 0). Error should be 0.
    reg.get_point(p3).y = 0.0
    assert c.error(reg, params) == pytest.approx(0.0)


def test_point_on_line_invalid_entity(setup_env):
    """Ensure it doesn't crash if passed a non-line ID."""
    reg, params = setup_env
    p1 = reg.add_point(0, 0)
    p2 = reg.add_point(10, 0)
    # Pass a Point ID instead of Line ID
    c = PointOnLineConstraint(p2, p1)
    assert c.error(reg, params) == 0.0


def test_radius_constraint(setup_env):
    reg, params = setup_env
    start = reg.add_point(10, 0)
    end = reg.add_point(0, 10)
    center = reg.add_point(0, 0)
    arc_id = reg.add_arc(start, end, center)

    # Current radius is 10. Target is 10. Error = 10^2 - 10^2 = 0
    c = RadiusConstraint(arc_id, 10.0)
    assert c.error(reg, params) == pytest.approx(0.0)

    # Target is 5. Error = 10^2 - 5^2 = 75
    c2 = RadiusConstraint(arc_id, 5.0)
    assert c2.error(reg, params) == pytest.approx(75.0)


def test_radius_constraint_on_circle(setup_env):
    reg, params = setup_env
    center = reg.add_point(0, 0)
    radius_pt = reg.add_point(10, 0)
    circ_id = reg.add_circle(center, radius_pt)

    # Current radius is 10. Target is 10. Error = 10^2 - 10^2 = 0
    c = RadiusConstraint(circ_id, 10.0)
    assert c.error(reg, params) == pytest.approx(0.0)

    # Target is 5. Error = 10^2 - 5^2 = 75
    c2 = RadiusConstraint(circ_id, 5.0)
    assert c2.error(reg, params) == pytest.approx(75.0)


def test_radius_constraint_invalid_entity(setup_env):
    reg, params = setup_env
    # Add a line, try to constrain radius (should fail gracefully/return 0)
    p1 = reg.add_point(0, 0)
    p2 = reg.add_point(10, 10)
    line_id = reg.add_line(p1, p2)

    c = RadiusConstraint(line_id, 5.0)
    assert c.error(reg, params) == 0.0


def test_diameter_constraint(setup_env):
    reg, params = setup_env
    center = reg.add_point(0, 0)
    # Radius is 5, so diameter is 10
    radius_pt = reg.add_point(5, 0)
    circ_id = reg.add_circle(center, radius_pt)

    # Target diam is 10, actual is 10. Error = 4*5^2 - 10^2 = 100 - 100 = 0
    c = DiameterConstraint(circ_id, 10.0)
    assert c.error(reg, params) == pytest.approx(0.0)

    # Target diam is 20, actual is 10. Error = 4*5^2 - 20^2 = 100 - 400 = -300
    c2 = DiameterConstraint(circ_id, 20.0)
    assert c2.error(reg, params) == pytest.approx(-300.0)


def test_perpendicular_constraint(setup_env):
    reg, params = setup_env

    # Line 1: Horizontal (0,0) -> (10,0)
    p1 = reg.add_point(0, 0)
    p2 = reg.add_point(10, 0)
    l1 = reg.add_line(p1, p2)

    # Line 2: Vertical (5,5) -> (5,15). Vector is (0, 10)
    p3 = reg.add_point(5, 5)
    p4 = reg.add_point(5, 15)
    l2 = reg.add_line(p3, p4)

    c = PerpendicularConstraint(l1, l2)
    # Dot product: (10, 0) . (0, 10) = 0
    assert c.error(reg, params) == pytest.approx(0.0)

    # Make Line 2 NOT perpendicular (Slope 1)
    # Move p4 to (15, 15). Vector (10, 10)
    reg.get_point(p4).x = 15

    # Dot product: (10, 0) . (10, 10) = 100
    assert c.error(reg, params) == pytest.approx(100.0)


def test_perpendicular_constraint_invalid_type(setup_env):
    reg, params = setup_env
    p1 = reg.add_point(0, 0)
    l1 = reg.add_line(p1, p1)  # Dummy line

    # Pass Point ID instead of Line ID
    c = PerpendicularConstraint(l1, p1)
    assert c.error(reg, params) == 0.0


def test_tangent_constraint(setup_env):
    reg, params = setup_env

    # Arc: Center at (0,0), Radius 10
    start = reg.add_point(10, 0)
    end = reg.add_point(0, 10)
    center = reg.add_point(0, 0)
    arc_id = reg.add_arc(start, end, center)

    # Line: Horizontal at y=10, from x=-5 to x=5
    # dist_to_line_sq = 100. radius_sq = 100. Error = 0.
    lp1 = reg.add_point(-5, 10)
    lp2 = reg.add_point(5, 10)
    line_id = reg.add_line(lp1, lp2)

    c = TangentConstraint(line_id, arc_id)
    assert c.error(reg, params) == pytest.approx(0.0)

    # Move line to y=20
    # dist_to_line_sq = 400. radius_sq = 100. Error = 300.
    reg.get_point(lp1).y = 20
    reg.get_point(lp2).y = 20

    assert c.error(reg, params) == pytest.approx(300.0)


def test_tangent_constraint_on_circle(setup_env):
    reg, params = setup_env

    # Circle: Center at (0,0), Radius 10
    center = reg.add_point(0, 0)
    radius_pt = reg.add_point(10, 0)
    circ_id = reg.add_circle(center, radius_pt)

    # Line: Horizontal at y=10, from x=-5 to x=5
    # dist_to_line_sq = 100. radius_sq = 100. Error = 0.
    lp1 = reg.add_point(-5, 10)
    lp2 = reg.add_point(5, 10)
    line_id = reg.add_line(lp1, lp2)

    c = TangentConstraint(line_id, circ_id)
    assert c.error(reg, params) == pytest.approx(0.0)

    # Move line to y=20
    # dist_to_line_sq = 400. radius_sq = 100. Error = 300.
    reg.get_point(lp1).y = 20
    reg.get_point(lp2).y = 20

    assert c.error(reg, params) == pytest.approx(300.0)


def test_drag_constraint(setup_env):
    reg, params = setup_env
    p1 = reg.add_point(0, 0)

    # Target is (100, 0). Current is (0, 0).
    # err_x = (0 - 100) * 0.01 = -1.0
    # err_y = (0 - 0) * 0.01 = 0.0
    c = DragConstraint(p1, 100.0, 0.0, weight=0.01)
    err_x, err_y = c.error(reg, params)
    assert err_x == pytest.approx(-1.0)
    assert err_y == pytest.approx(0.0)


def test_constraint_zero_length_protection(setup_env):
    """Test denominator protection for PointOnLine and Tangent constraints."""
    reg, params = setup_env

    # Create a 'Line' that is just a point (0,0) -> (0,0)
    p1 = reg.add_point(0, 0)
    p2 = reg.add_point(0, 0)
    line_id = reg.add_line(p1, p2)
    p3 = reg.add_point(5, 5)

    # PointOnLine
    c_pol = PointOnLineConstraint(p3, line_id)
    # Cross product is (0-0)*... = 0. Should be 0.0.
    assert c_pol.error(reg, params) == 0.0

    # Tangent (requires line length)
    start = reg.add_point(10, 0)
    end = reg.add_point(0, 10)
    center = reg.add_point(0, 0)
    arc_id = reg.add_arc(start, end, center)

    c_tan = TangentConstraint(line_id, arc_id)
    # Error is dist_to_pt^2 - radius^2. Here dist is 0, radius is 10.
    # Error = 0^2 - 10^2 = -100
    assert c_tan.error(reg, params) == pytest.approx(-100.0)


def test_constraint_serialization_round_trip():
    """Tests to_dict and from_dict for all constraint types."""
    # List all constraint classes and their constructor arguments
    constraints_to_test = [
        DistanceConstraint(p1=0, p2=1, value=10.0),
        DistanceConstraint(p1=0, p2=1, value="width"),
        EqualDistanceConstraint(p1=0, p2=1, p3=2, p4=3),
        HorizontalConstraint(p1=0, p2=1),
        VerticalConstraint(p1=0, p2=1),
        CoincidentConstraint(p1=0, p2=1),
        PointOnLineConstraint(point_id=2, shape_id=4),
        RadiusConstraint(entity_id=5, radius=20.0),
        DiameterConstraint(circle_id=7, diameter=40.0),
        PerpendicularConstraint(l1_id=4, l2_id=6),
        TangentConstraint(line_id=4, shape_id=5),
        EqualLengthConstraint(entity_ids=[4, 5, 6]),
    ]

    for constr in constraints_to_test:
        data = constr.to_dict()
        assert "type" in data
        # Get the class from globals() using the type string
        cls = globals()[data["type"]]
        new_constr = cls.from_dict(data)

        # Check that all attributes were restored correctly
        assert new_constr.__dict__ == constr.__dict__

    # DragConstraint is not serializable
    drag = DragConstraint(0, 1, 2)
    assert drag.to_dict() == {}


def test_equal_length_constraint_serialization_legacy():
    """
    Tests backward compatibility for EqualLengthConstraint deserialization.
    """
    legacy_data = {
        "type": "EqualLengthConstraint",
        "e1_id": 10,
        "e2_id": 12,
    }
    constr = EqualLengthConstraint.from_dict(legacy_data)
    assert isinstance(constr, EqualLengthConstraint)
    assert constr.entity_ids == [10, 12]
