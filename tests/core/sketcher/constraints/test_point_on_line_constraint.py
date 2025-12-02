import pytest
import numpy as np
from scipy.optimize import check_grad
from types import SimpleNamespace

from rayforge.core.sketcher.params import ParameterContext
from rayforge.core.sketcher.entities import EntityRegistry
from rayforge.core.sketcher.constraints import PointOnLineConstraint


@pytest.fixture
def setup_env():
    reg = EntityRegistry()
    params = ParameterContext()
    return reg, params


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


def test_point_on_line_constrains_radius(setup_env):
    """
    Test that PointOnLineConstraint correctly reports constraining the radius
    when the point on the circle is itself constrained.
    """
    reg, params = setup_env

    # Setup Circle
    c = reg.add_point(0, 0, fixed=True)
    r_pt = reg.add_point(10, 0, fixed=False)  # Not fixed initially
    circ_id = reg.add_circle(c, r_pt)

    # Setup Constrained Point on Circle
    p_constrained = reg.add_point(0, 10, fixed=False)
    reg.get_point(p_constrained).constrained = True  # Simulate solver result

    # Setup Unconstrained Point on Circle
    p_unconstrained = reg.add_point(0, -10, fixed=False)
    reg.get_point(p_unconstrained).constrained = False

    # Constraint 1: Constrained Point -> Circle
    c1 = PointOnLineConstraint(p_constrained, circ_id)
    assert c1.constrains_radius(reg, circ_id) is True

    # Constraint 2: Unconstrained Point -> Circle
    c2 = PointOnLineConstraint(p_unconstrained, circ_id)
    assert c2.constrains_radius(reg, circ_id) is False

    # Constraint 3: Wrong Entity ID
    assert c1.constrains_radius(reg, 999) is False


def test_point_on_line_invalid_entity(setup_env):
    """Ensure it doesn't crash if passed a non-line ID."""
    reg, params = setup_env
    p1 = reg.add_point(0, 0)
    p2 = reg.add_point(10, 0)
    # Pass a Point ID instead of a shape ID
    c = PointOnLineConstraint(p2, p1)
    assert c.error(reg, params) == 0.0


def test_point_on_line_zero_length(setup_env):
    """Test denominator protection for PointOnLine with a zero-length line."""
    reg, params = setup_env
    p1 = reg.add_point(0, 0)
    p2 = reg.add_point(0, 0)
    line_id = reg.add_line(p1, p2)
    p3 = reg.add_point(5, 5)
    c_pol = PointOnLineConstraint(p3, line_id)
    expected_dist = (5**2 + 5**2) ** 0.5
    assert c_pol.error(reg, params) == pytest.approx(expected_dist)


def test_point_on_line_gradient(setup_env):
    reg, params = setup_env
    pt_id = reg.add_point(5, 6)
    l1_id = reg.add_point(0, 0)
    l2_id = reg.add_point(10, 2)
    line_id = reg.add_line(l1_id, l2_id)
    mutable_pids = [pt_id, l1_id, l2_id]

    constraint = PointOnLineConstraint(point_id=pt_id, shape_id=line_id)

    pid_to_idx_map = {pid: i for i, pid in enumerate(mutable_pids)}

    def update_state_from_vec(x_vec):
        for pid, i in pid_to_idx_map.items():
            pt = reg.get_point(pid)
            pt.x = x_vec[i * 2]
            pt.y = x_vec[i * 2 + 1]

    def func_wrapper(x_vec):
        update_state_from_vec(x_vec)
        return constraint.error(reg, params)

    def grad_wrapper(x_vec):
        update_state_from_vec(x_vec)
        grad_map = constraint.gradient(reg, params)
        grad_vec = np.zeros_like(x_vec)
        for pid, grads in grad_map.items():
            if pid in pid_to_idx_map:
                idx = pid_to_idx_map[pid] * 2
                dx, dy = grads[0]
                grad_vec[idx] = dx
                grad_vec[idx + 1] = dy
        return grad_vec

    x0 = np.array([5, 6, 0, 0, 10, 2], dtype=float)
    diff = check_grad(func_wrapper, grad_wrapper, x0, epsilon=1e-6)
    assert diff < 1e-5


def test_point_on_circle_gradient(setup_env):
    reg, params = setup_env
    pt_id = reg.add_point(10, 12)
    c_id = reg.add_point(5, 5)
    r_id = reg.add_point(15, 5)
    circle_id = reg.add_circle(c_id, r_id)
    mutable_pids = [pt_id, c_id, r_id]

    constraint = PointOnLineConstraint(point_id=pt_id, shape_id=circle_id)
    pid_to_idx_map = {pid: i for i, pid in enumerate(mutable_pids)}

    def update_state_from_vec(x_vec):
        for pid, i in pid_to_idx_map.items():
            pt = reg.get_point(pid)
            pt.x = x_vec[i * 2]
            pt.y = x_vec[i * 2 + 1]

    def func_wrapper(x_vec):
        update_state_from_vec(x_vec)
        return constraint.error(reg, params)

    def grad_wrapper(x_vec):
        update_state_from_vec(x_vec)
        grad_map = constraint.gradient(reg, params)
        grad_vec = np.zeros_like(x_vec)
        for pid, grads in grad_map.items():
            if pid in pid_to_idx_map:
                idx = pid_to_idx_map[pid] * 2
                dx, dy = grads[0]
                grad_vec[idx] = dx
                grad_vec[idx + 1] = dy
        return grad_vec

    x0 = np.array([10, 12, 5, 5, 15, 5], dtype=float)
    diff = check_grad(func_wrapper, grad_wrapper, x0, epsilon=1e-6)
    assert diff < 1e-5


def test_point_on_line_constraint_serialization_round_trip(setup_env):
    reg, params = setup_env

    # Line along X-axis: (0,0) -> (10,0)
    p1 = reg.add_point(0, 0)
    p2 = reg.add_point(10, 0)
    line_id = reg.add_line(p1, p2)

    # Point at (5, 5)
    p3 = reg.add_point(5, 5)

    # Create original constraint
    original = PointOnLineConstraint(p3, line_id)

    # Serialize to dict
    serialized = original.to_dict()

    # Deserialize from dict
    restored = PointOnLineConstraint.from_dict(serialized)

    # Check that the restored constraint has the same error
    assert original.error(reg, params) == restored.error(reg, params)


def test_point_on_line_is_hit(setup_env):
    reg, params = setup_env
    p1 = reg.add_point(10, 20)
    l1 = reg.add_point(0, 0)
    l2 = reg.add_point(100, 100)
    line = reg.add_line(l1, l2)
    c = PointOnLineConstraint(p1, line)

    def to_screen(pos):
        return pos

    mock_element = SimpleNamespace()
    threshold = 15.0

    # Hit the point
    assert c.is_hit(10, 20, reg, to_screen, mock_element, threshold) is True
    # Miss the point
    assert c.is_hit(30, 20, reg, to_screen, mock_element, threshold) is False
