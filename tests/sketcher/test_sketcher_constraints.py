import pytest
import numpy as np
from scipy.optimize import check_grad
from functools import partial

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
    SymmetryConstraint,
)


@pytest.fixture
def setup_env():
    reg = EntityRegistry()
    params = ParameterContext()
    return reg, params


@pytest.fixture
def setup_ui_env(setup_env):
    """Extends setup_env with mock UI elements."""
    reg, params = setup_env

    # Mock to_screen: identity function (model coords == screen coords)
    def to_screen(pos):
        return pos

    # Mock element with a canvas that has a scale
    class MockCanvas:
        def get_view_scale(self):
            return 1.0, 1.0

    class MockElement:
        def __init__(self):
            self.canvas = MockCanvas()
            self.sketch = type("sketch", (), {})()  # Dummy object

    element = MockElement()
    return reg, params, to_screen, element


def test_distance_constraint(setup_env):
    reg, params = setup_env
    p1 = reg.add_point(0, 0)
    p2 = reg.add_point(10, 0)

    # Target is 10, actual is 10. Error should be 10 - 10 = 0.
    c = DistanceConstraint(p1, p2, 10.0)
    assert c.error(reg, params) == pytest.approx(0.0)

    # Target is 5, actual is 10. Error should be 10 - 5 = 5.
    c2 = DistanceConstraint(p1, p2, 5.0)
    assert c2.error(reg, params) == pytest.approx(5.0)


def test_distance_constraint_with_expression(setup_env):
    reg, params = setup_env
    params.set("width", 20.0)
    p1 = reg.add_point(0, 0)
    p2 = reg.add_point(10, 0)

    c = DistanceConstraint(p1, p2, "width")
    # Actual dist is 10, Target dist is 20. Error is 10 - 20 = -10.
    assert c.error(reg, params) == pytest.approx(-10.0)


def test_equal_distance_constraint(setup_env):
    reg, params = setup_env
    # Segment 1: Length 10
    p1 = reg.add_point(0, 0)
    p2 = reg.add_point(10, 0)

    # Segment 2: Length 4
    p3 = reg.add_point(0, 10)
    p4 = reg.add_point(0, 14)

    c = EqualDistanceConstraint(p1, p2, p3, p4)

    # Error should be 10 - 4 = 6
    assert c.error(reg, params) == pytest.approx(6.0)


def test_equal_length_constraint(setup_env):
    """Tests the error calculation for EqualLengthConstraint."""
    reg, params = setup_env
    # Line 1: length 10
    p1 = reg.add_point(0, 0)
    p2 = reg.add_point(10, 0)
    l1 = reg.add_line(p1, p2)

    # Line 2: length 5
    p3 = reg.add_point(0, 10)
    p4 = reg.add_point(5, 10)
    l2 = reg.add_line(p3, p4)

    # Arc 1: radius 4
    c1_p = reg.add_point(20, 0)
    s1 = reg.add_point(24, 0)
    e1 = reg.add_point(20, 4)
    a1 = reg.add_arc(s1, e1, c1_p)

    # Circle 1: radius 3
    c2_p = reg.add_point(30, 0)
    r2 = reg.add_point(33, 0)
    circ1 = reg.add_circle(c2_p, r2)

    # Test Line-Line (error is [len2 - len1])
    c = EqualLengthConstraint([l1, l2])
    assert c.error(reg, params) == pytest.approx([5.0 - 10.0])

    # Test Line-Arc (error is [rad1 - len1])
    c2 = EqualLengthConstraint([l1, a1])
    assert c2.error(reg, params) == pytest.approx([4.0 - 10.0])

    # Test Arc-Circle (error is [rad_circ - rad_arc])
    c3 = EqualLengthConstraint([a1, circ1])
    assert c3.error(reg, params) == pytest.approx([3.0 - 4.0])

    # Test multi-entity constraint
    c_multi = EqualLengthConstraint([l1, l2, a1, circ1])
    # Errors are [len2-len1, rad_arc-len1, rad_circ-len1]
    assert c_multi.error(reg, params) == pytest.approx(
        [5.0 - 10.0, 4.0 - 10.0, 3.0 - 10.0]
    )

    # Test edge cases (no error for < 2 entities)
    assert EqualLengthConstraint([]).error(reg, params) == []
    assert EqualLengthConstraint([l1]).error(reg, params) == []


def test_horizontal_constraint(setup_env):
    reg, params = setup_env
    # p1 y=0, p2 y=5. Error should be 0 - 5 = -5
    p1 = reg.add_point(0, 0)
    p2 = reg.add_point(10, 5)

    c = HorizontalConstraint(p1, p2)
    assert c.error(reg, params) == pytest.approx(-5.0)


def test_vertical_constraint(setup_env):
    reg, params = setup_env
    # p1 x=0, p2 x=5. Error should be 0 - 5 = -5
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
    assert c.error(reg, params) == (0.0 - 3.0, 0.0 - 4.0)


def test_point_on_line_constraint(setup_env):
    reg, params = setup_env

    # Line along X-axis: (0,0) -> (10,0)
    p1 = reg.add_point(0, 0)
    p2 = reg.add_point(10, 0)
    line_id = reg.add_line(p1, p2)

    # Point at (5, 5). Distance is 5.
    # Error is signed distance: cross product / length
    # cross = (10-0)*(5-0) - (5-0)*(0-0) = 50
    # length = 10. Error = 50/10 = 5.
    p3 = reg.add_point(5, 5)

    c = PointOnLineConstraint(p3, line_id)
    assert c.error(reg, params) == pytest.approx(5.0)

    # Move point to (5, 0). Error should be 0.
    reg.get_point(p3).y = 0.0
    assert c.error(reg, params) == pytest.approx(0.0)


def test_point_on_arc_circle_constraint(setup_env):
    """Tests PointOnLine constraint for Arc and Circle types."""
    reg, params = setup_env

    # Circle at (0,0) with radius 10
    center = reg.add_point(0, 0)
    radius_pt = reg.add_point(10, 0)
    circ_id = reg.add_circle(center, radius_pt)

    # Point on the circle circumference
    pt_on = reg.add_point(0, 10)
    c1 = PointOnLineConstraint(pt_on, circ_id)
    # Error: dist(pt, center) - radius = 10 - 10 = 0
    assert c1.error(reg, params) == pytest.approx(0.0)

    # Point outside the circle
    pt_off = reg.add_point(0, 12)
    c2 = PointOnLineConstraint(pt_off, circ_id)
    # Error: dist(pt, center) - radius = 12 - 10 = 2
    assert c2.error(reg, params) == pytest.approx(2.0)


def test_point_on_line_invalid_entity(setup_env):
    """Ensure it doesn't crash if passed a non-line ID."""
    reg, params = setup_env
    p1 = reg.add_point(0, 0)
    p2 = reg.add_point(10, 0)
    # Pass a Point ID instead of a shape ID
    c = PointOnLineConstraint(p2, p1)
    assert c.error(reg, params) == 0.0


def test_radius_constraint(setup_env):
    reg, params = setup_env
    start = reg.add_point(10, 0)
    end = reg.add_point(0, 10)
    center = reg.add_point(0, 0)
    arc_id = reg.add_arc(start, end, center)

    # Current radius is 10. Target is 10. Error = 10 - 10 = 0
    c = RadiusConstraint(arc_id, 10.0)
    assert c.error(reg, params) == pytest.approx(0.0)

    # Target is 5. Error = 10 - 5 = 5
    c2 = RadiusConstraint(arc_id, 5.0)
    assert c2.error(reg, params) == pytest.approx(5.0)


def test_radius_constraint_on_circle(setup_env):
    reg, params = setup_env
    center = reg.add_point(0, 0)
    radius_pt = reg.add_point(10, 0)
    circ_id = reg.add_circle(center, radius_pt)

    # Current radius is 10. Target is 10. Error = 10 - 10 = 0
    c = RadiusConstraint(circ_id, 10.0)
    assert c.error(reg, params) == pytest.approx(0.0)

    # Target is 5. Error = 10 - 5 = 5
    c2 = RadiusConstraint(circ_id, 5.0)
    assert c2.error(reg, params) == pytest.approx(5.0)


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

    # Target diam is 10, actual is 10. Error = 10 - 10 = 0
    c = DiameterConstraint(circ_id, 10.0)
    assert c.error(reg, params) == pytest.approx(0.0)

    # Target diam is 20, actual is 10. Error = 10 - 20 = -10
    c2 = DiameterConstraint(circ_id, 20.0)
    assert c2.error(reg, params) == pytest.approx(-10.0)


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


def test_perpendicular_constraint_extended(setup_env):
    """Tests perpendicular constraint for Line-Circle and Circle-Circle."""
    reg, params = setup_env

    # --- Line-Circle Test ---
    # Circle at (10,10), radius 5
    c1_p = reg.add_point(10, 10)
    c1_r = reg.add_point(15, 10)
    circ1 = reg.add_circle(c1_p, c1_r)

    # Line passing through center (0,10) -> (20,10)
    l1_p1 = reg.add_point(0, 10)
    l1_p2 = reg.add_point(20, 10)
    line1 = reg.add_line(l1_p1, l1_p2)

    # Error is cross product of (L2-L1) and (C-L1)
    # (20,0) x (10-0, 10-10) = 20*0 - 10*0 = 0
    lc_constraint = PerpendicularConstraint(line1, circ1)
    assert lc_constraint.error(reg, params) == pytest.approx(0.0)

    # Move line so it doesn't pass through center
    reg.get_point(l1_p1).y = 0
    reg.get_point(l1_p2).y = 0
    # Line is now (0,0)->(20,0). Center is (10,10).
    # Vector L2-L1: (20, 0)
    # Vector C-L1: (10, 10)
    # Cross product: (20 * 10) - (10 * 0) = 200
    assert lc_constraint.error(reg, params) == pytest.approx(200.0)

    # --- Circle-Circle Test ---
    # C1 at (0,0), radius 3 (r^2=9)
    c2_p = reg.add_point(0, 0)
    c2_r = reg.add_point(3, 0)
    circ2 = reg.add_circle(c2_p, c2_r)

    # C2 at (5,0), radius 4 (r^2=16), distance between centers = 5 (d^2=25)
    c3_p = reg.add_point(5, 0)
    c3_r = reg.add_point(9, 0)  # 5+4
    circ3 = reg.add_circle(c3_p, c3_r)

    # Error is r1^2 + r2^2 - d^2 = 9 + 16 - 25 = 0
    cc_constraint = PerpendicularConstraint(circ2, circ3)
    assert cc_constraint.error(reg, params) == pytest.approx(0.0)

    # Move C2 center to (6,0), d^2 = 36
    reg.get_point(c3_p).x = 6.0
    # Radius of circ3 changes! New radius is (9-6)=3, so r2^2=9.
    # Error = r1^2 + new_r2^2 - d^2 = 9 + 9 - 36 = -18
    assert cc_constraint.error(reg, params) == pytest.approx(-18.0)


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

    # Line: Horizontal at y=10. dist_to_line = 10. radius = 10. Error = 0.
    lp1 = reg.add_point(-5, 10)
    lp2 = reg.add_point(5, 10)
    line_id = reg.add_line(lp1, lp2)

    c = TangentConstraint(line_id, arc_id)
    assert c.error(reg, params) == pytest.approx(0.0)

    # Move line to y=20. dist=20, radius=10. Error = 10.
    reg.get_point(lp1).y = 20
    reg.get_point(lp2).y = 20

    assert c.error(reg, params) == pytest.approx(10.0)


def test_tangent_constraint_on_circle(setup_env):
    reg, params = setup_env

    # Circle: Center at (0,0), Radius 10
    center = reg.add_point(0, 0)
    radius_pt = reg.add_point(10, 0)
    circ_id = reg.add_circle(center, radius_pt)

    # Line: Horizontal at y=10. dist=10, radius=10. Error = 0.
    lp1 = reg.add_point(-5, 10)
    lp2 = reg.add_point(5, 10)
    line_id = reg.add_line(lp1, lp2)

    c = TangentConstraint(line_id, circ_id)
    assert c.error(reg, params) == pytest.approx(0.0)

    # Move line to y=20. dist=20, radius=10, Error = 10.
    reg.get_point(lp1).y = 20
    reg.get_point(lp2).y = 20

    assert c.error(reg, params) == pytest.approx(10.0)


def test_symmetry_constraint_point(setup_env):
    """Test symmetry between two points with respect to a center point."""
    reg, params = setup_env
    # Center at (0,0)
    pc = reg.add_point(0, 0)
    # P1 at (-5, -2)
    p1 = reg.add_point(-5, -2)
    # P2 at (5, 2) (perfectly symmetric)
    p2 = reg.add_point(5, 2)

    c = SymmetryConstraint(p1, p2, center=pc)

    # Error vector: [(x1+x2) - 2xc, (y1+y2) - 2yc]
    # x: (-5 + 5) - 0 = 0
    # y: (-2 + 2) - 0 = 0
    assert c.error(reg, params) == [0.0, 0.0]

    # Move P2 to (6, 2)
    # x: (-5 + 6) - 0 = 1
    reg.get_point(p2).x = 6.0
    assert c.error(reg, params) == [1.0, 0.0]


def test_symmetry_constraint_line(setup_env):
    """Test symmetry between two points with respect to an axis line."""
    reg, params = setup_env
    # Axis on Y-axis: (0, -10) -> (0, 10)
    l1 = reg.add_point(0, -10)
    l2 = reg.add_point(0, 10)
    axis_id = reg.add_line(l1, l2)

    # P1 at (-5, 5), P2 at (5, 5) (perfectly symmetric)
    p1 = reg.add_point(-5, 5)
    p2 = reg.add_point(5, 5)

    c = SymmetryConstraint(p1, p2, axis=axis_id)
    assert c.error(reg, params) == [0.0, 0.0]

    # Move P2 up by 1 -> (5, 6)
    reg.get_point(p2).y = 6.0

    # 1. Perpendicularity check (Dot product)
    # Axis Vector: (0, 20)
    # Point Vector P1->P2: (10, 1)
    # Dot: 0*10 + 20*1 = 20
    expected_perp_err = 20.0

    # 2. Midpoint on line check (Cross product logic)
    # Midpoint of P1(-5, 5) and P2(5, 6) is (0, 5.5)
    # Line Start L1(0, -10). Vector L1->Mid is (0, 15.5)
    # Axis Vector L1->L2 is (0, 20)
    # Cross product 2D: (0 * 20) - (15.5 * 0) = 0
    expected_coll_err = 0.0

    err = c.error(reg, params)
    assert err[0] == pytest.approx(expected_perp_err)
    assert err[1] == pytest.approx(expected_coll_err)


def test_drag_constraint(setup_env):
    reg, params = setup_env
    p1 = reg.add_point(0, 0)

    # Target is (100, 0). Current is (0, 0).
    # err_x = (0 - 100) * 0.1 = -10.0
    # err_y = (0 - 0) * 0.1 = 0.0
    c = DragConstraint(p1, 100.0, 0.0, weight=0.1)
    err_x, err_y = c.error(reg, params)
    assert err_x == pytest.approx(-10.0)
    assert err_y == pytest.approx(0.0)


def test_constraint_zero_length_protection(setup_env):
    """Test denominator protection for PointOnLine and Tangent constraints."""
    reg, params = setup_env

    # Create a 'Line' that is just a point (0,0) -> (0,0)
    p1 = reg.add_point(0, 0)
    p2 = reg.add_point(0, 0)
    line_id = reg.add_line(p1, p2)
    p3 = reg.add_point(5, 5)

    # PointOnLine should fall back to point-point distance
    c_pol = PointOnLineConstraint(p3, line_id)
    expected_dist = (5**2 + 5**2) ** 0.5
    assert c_pol.error(reg, params) == pytest.approx(expected_dist)

    # Tangent (requires line length)
    start = reg.add_point(10, 0)
    end = reg.add_point(0, 10)
    center = reg.add_point(0, 0)
    arc_id = reg.add_arc(start, end, center)

    c_tan = TangentConstraint(line_id, arc_id)
    # Error falls back to dist(center, line_pt) - radius
    # dist = 0. radius = 10. Error = 0 - 10 = -10.
    assert c_tan.error(reg, params) == pytest.approx(-10.0)


# =============================================================================
# GRADIENT CHECKING
# =============================================================================

# Define test cases for various constraints
# Each case defines:
# - 'cls': The constraint class to test.
# - 'args': The arguments to initialize the constraint.
# - 'geometries': A list of tuples describing the initial sketch setup.
# - 'mutable_pids': A list of point IDs that are variables for the check.
constraints_to_test = [
    {
        "cls": DistanceConstraint,
        "args": {"p1": 0, "p2": 1, "value": 10.0},
        "geometries": [("point", 1, 2), ("point", 5, 6)],
        "mutable_pids": [0, 1],
    },
    {
        "cls": HorizontalConstraint,
        "args": {"p1": 0, "p2": 1},
        "geometries": [("point", 1, 2), ("point", 5, 6)],
        "mutable_pids": [0, 1],
    },
    {
        "cls": VerticalConstraint,
        "args": {"p1": 0, "p2": 1},
        "geometries": [("point", 1, 2), ("point", 5, 6)],
        "mutable_pids": [0, 1],
    },
    {
        "cls": CoincidentConstraint,
        "args": {"p1": 0, "p2": 1},
        "geometries": [("point", 1, 2), ("point", 5, 6)],
        "mutable_pids": [0, 1],
    },
    {
        "cls": PointOnLineConstraint,
        "args": {"point_id": 0, "shape_id": 3},
        "geometries": [
            ("point", 5, 6),
            ("point", 0, 0),
            ("point", 10, 2),
            ("line", 1, 2),
        ],
        "mutable_pids": [0, 1, 2],
    },
    {
        "cls": PointOnLineConstraint,
        "args": {"point_id": 0, "shape_id": 3},
        "geometries": [
            ("point", 10, 12),
            ("point", 5, 5),
            ("point", 15, 5),
            ("circle", 1, 2),
        ],
        "mutable_pids": [0, 1, 2],
    },
    {
        "cls": RadiusConstraint,
        "args": {"entity_id": 3, "radius": 12.0},
        "geometries": [
            ("point", 5, 5),
            ("point", 15, 5),
            ("point", 5, 15),
            ("arc", 1, 2, 0),
        ],
        "mutable_pids": [0, 1],
    },
    {
        "cls": PerpendicularConstraint,
        "args": {"e1_id": 4, "e2_id": 5},
        "geometries": [
            ("point", 0, 0),
            ("point", 10, 2),
            ("point", 5, 10),
            ("point", 3, -5),
            ("line", 0, 1),
            ("line", 2, 3),
        ],
        "mutable_pids": [0, 1, 2, 3],
    },
    {
        "cls": TangentConstraint,
        "args": {"line_id": 4, "shape_id": 5},
        "geometries": [
            ("point", 0, 12),
            ("point", 10, 12),  # Line points
            ("point", 5, 0),
            ("point", 15, 0),  # Circle center and radius points
            ("line", 0, 1),
            ("circle", 2, 3),
        ],
        "mutable_pids": [0, 1, 2, 3],
    },
    {
        "cls": EqualLengthConstraint,
        "args": {"entity_ids": [4, 5]},
        "geometries": [
            ("point", 0, 0),
            ("point", 10, 0),
            ("point", 15, 15),
            ("point", 15, 22),
            ("line", 0, 1),
            ("line", 2, 3),
        ],
        "mutable_pids": [0, 1, 2, 3],
    },
    {
        "cls": SymmetryConstraint,
        "args": {"p1": 0, "p2": 1, "center": 2},
        "geometries": [("point", -5, 2), ("point", 5, -3), ("point", 1, -1)],
        "mutable_pids": [0, 1, 2],
    },
    {
        "cls": SymmetryConstraint,
        "args": {"p1": 0, "p2": 1, "axis": 4},
        "geometries": [
            ("point", -5, 10),
            ("point", 6, 11),  # Symmetric points
            ("point", 0, 0),
            ("point", 0, 20),  # Axis line points
            ("line", 2, 3),
        ],
        "mutable_pids": [0, 1, 2, 3],
    },
]


@pytest.mark.parametrize("test_case", constraints_to_test)
def test_constraint_gradients(setup_env, test_case):
    """
    Uses scipy.optimize.check_grad to numerically verify the analytical
    gradients of each constraint type.
    """
    reg, params = setup_env
    constr_class = test_case["cls"]
    constr_args = test_case["args"]
    geometries = test_case["geometries"]
    mutable_pids = test_case["mutable_pids"]

    # 1. Setup the geometry and constraint instance
    entity_ids = {}
    for geo_type, *geo_args in geometries:
        if geo_type == "point":
            reg.add_point(*geo_args)
        elif geo_type == "line":
            eid = reg.add_line(*geo_args)
            entity_ids[eid] = "line"
        elif geo_type == "arc":
            eid = reg.add_arc(*geo_args)
            entity_ids[eid] = "arc"
        elif geo_type == "circle":
            eid = reg.add_circle(*geo_args)
            entity_ids[eid] = "circle"

    constraint = constr_class(**constr_args)

    # 2. Define wrappers for check_grad
    pid_to_idx_map = {pid: i for i, pid in enumerate(mutable_pids)}

    def update_state_from_vec(x_vec):
        for pid, i in pid_to_idx_map.items():
            pt = reg.get_point(pid)
            pt.x = x_vec[i * 2]
            pt.y = x_vec[i * 2 + 1]

    def func_wrapper(x_vec, error_index=0):
        update_state_from_vec(x_vec)
        err = constraint.error(reg, params)
        if isinstance(err, (list, tuple)):
            return err[error_index]
        return err

    def grad_wrapper(x_vec, error_index=0):
        update_state_from_vec(x_vec)
        grad_map = constraint.gradient(reg, params)

        grad_vec = np.zeros_like(x_vec)
        for pid, grads in grad_map.items():
            if pid in pid_to_idx_map:
                idx = pid_to_idx_map[pid] * 2
                # Ensure grads has enough elements
                if error_index < len(grads):
                    dx, dy = grads[error_index]
                    grad_vec[idx] = dx
                    grad_vec[idx + 1] = dy
        return grad_vec

    # 3. Get initial state and run the check
    x0 = np.array(
        [coord for pid in mutable_pids for coord in reg.get_point(pid).pos()],
        dtype=float,
    )

    # Determine how many error values the constraint produces
    num_errors = 1
    initial_error = constraint.error(reg, params)
    if isinstance(initial_error, (list, tuple)):
        num_errors = len(initial_error)

    # Check gradient for each error component
    for i in range(num_errors):
        # Use partial to create functions with the fixed error_index
        func = partial(func_wrapper, error_index=i)
        grad = partial(grad_wrapper, error_index=i)

        # Using a slightly larger epsilon can help with numerical stability
        # for functions with square roots.
        diff = check_grad(func, grad, x0, epsilon=1e-6)

        # A small difference (e.g., < 1e-5) indicates the gradient is correct
        assert diff < 1e-5, (
            f"Gradient mismatch for {constr_class.__name__} (error #{i}). "
            f"Difference: {diff}"
        )


# =============================================================================
# GRADIENT CHECKING (SHARED POINTS)
# These tests specifically check for the gradient accumulation bug where one
# point has multiple roles in a constraint and its gradient contributions
# were being overwritten instead of added.
# =============================================================================

constraints_with_shared_points = [
    {
        "name": "TangentConstraint: line endpoint is also circle radius point",
        "cls": TangentConstraint,
        "args": {"line_id": 3, "shape_id": 4},
        "geometries": [
            ("point", 5, 0),  # p0: circle center
            ("point", 0, 10),  # p1: line start
            ("point", 10, 10),  # p2: line end AND circle radius point
            ("line", 1, 2),
            ("circle", 0, 2),
        ],
        "mutable_pids": [0, 1, 2],
    },
    {
        "name": (
            "PerpendicularConstraint: circle center is other circle's "
            "radius point",
        ),
        "cls": PerpendicularConstraint,
        "args": {"e1_id": 3, "e2_id": 4},
        "geometries": [
            ("point", 0, 0),  # p0: C1 center AND C2 radius point
            ("point", 3, 0),  # p1: C1 radius point
            ("point", 5, 0),  # p2: C2 center
            ("circle", 0, 1),
            ("circle", 2, 0),
        ],
        "mutable_pids": [0, 1, 2],
    },
]


@pytest.mark.parametrize("test_case", constraints_with_shared_points)
def test_constraint_gradients_with_shared_points(setup_env, test_case):
    """
    Verifies the analytical gradients for complex constraints where a single
    point has multiple geometric roles.
    """
    reg, params = setup_env
    constr_class = test_case["cls"]
    constr_args = test_case["args"]
    geometries = test_case["geometries"]
    mutable_pids = test_case["mutable_pids"]
    test_name = test_case["name"]

    # Setup is identical to the main gradient test
    for geo_type, *geo_args in geometries:
        if geo_type == "point":
            reg.add_point(*geo_args)
        elif geo_type == "line":
            reg.add_line(*geo_args)
        elif geo_type == "circle":
            reg.add_circle(*geo_args)

    constraint = constr_class(**constr_args)
    pid_to_idx_map = {pid: i for i, pid in enumerate(mutable_pids)}

    def update_state_from_vec(x_vec):
        for pid, i in pid_to_idx_map.items():
            pt = reg.get_point(pid)
            pt.x = x_vec[i * 2]
            pt.y = x_vec[i * 2 + 1]

    def func_wrapper(x_vec, error_index=0):
        update_state_from_vec(x_vec)
        err = constraint.error(reg, params)
        if isinstance(err, (list, tuple)):
            return err[error_index]
        return err

    def grad_wrapper(x_vec, error_index=0):
        update_state_from_vec(x_vec)
        grad_map = constraint.gradient(reg, params)
        grad_vec = np.zeros_like(x_vec)
        for pid, grads in grad_map.items():
            if pid in pid_to_idx_map:
                idx = pid_to_idx_map[pid] * 2
                if error_index < len(grads):
                    dx, dy = grads[error_index]
                    grad_vec[idx] = dx
                    grad_vec[idx + 1] = dy
        return grad_vec

    x0 = np.array(
        [coord for pid in mutable_pids for coord in reg.get_point(pid).pos()],
        dtype=float,
    )
    num_errors = 1
    initial_error = constraint.error(reg, params)
    if isinstance(initial_error, (list, tuple)):
        num_errors = len(initial_error)

    for i in range(num_errors):
        func = partial(func_wrapper, error_index=i)
        grad = partial(grad_wrapper, error_index=i)
        diff = check_grad(func, grad, x0, epsilon=1e-6)

        assert diff < 1e-5, (
            f"Gradient mismatch for SHARED POINT test '{test_name}' "
            f"({constr_class.__name__}, error #{i}). Difference: {diff}"
        )
