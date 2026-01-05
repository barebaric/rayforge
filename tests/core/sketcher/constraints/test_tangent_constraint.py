import pytest
import numpy as np
from scipy.optimize import check_grad
from types import SimpleNamespace

from rayforge.core.sketcher.constraints import TangentConstraint
from rayforge.core.sketcher.params import ParameterContext
from rayforge.core.sketcher.registry import EntityRegistry


@pytest.fixture
def setup_env():
    reg = EntityRegistry()
    params = ParameterContext()
    return reg, params


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


def test_tangent_constraint_zero_length_line(setup_env):
    """Test denominator protection for Tangent constraint."""
    reg, params = setup_env
    p1 = reg.add_point(0, 0)
    line_id = reg.add_line(p1, p1)

    start = reg.add_point(10, 0)
    end = reg.add_point(0, 10)
    center = reg.add_point(0, 0)
    arc_id = reg.add_arc(start, end, center)

    c_tan = TangentConstraint(line_id, arc_id)
    # Error falls back to dist(center, line_pt) - radius
    # dist = 0. radius = 10. Error = 0 - 10 = -10.
    assert c_tan.error(reg, params) == pytest.approx(-10.0)


def test_tangent_constraint_gradient(setup_env):
    reg, params = setup_env
    lp1 = reg.add_point(0, 12)
    lp2 = reg.add_point(10, 12)
    cp = reg.add_point(5, 0)
    rp = reg.add_point(15, 0)
    line = reg.add_line(lp1, lp2)
    circle = reg.add_circle(cp, rp)
    mutable_pids = [lp1, lp2, cp, rp]

    constraint = TangentConstraint(line_id=line, shape_id=circle)
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

    x0 = np.array([0, 12, 10, 12, 5, 0, 15, 0], dtype=float)
    diff = check_grad(func_wrapper, grad_wrapper, x0, epsilon=1e-6)
    assert diff < 1e-5


def test_tangent_gradient_with_shared_points(setup_env):
    reg, params = setup_env
    p0 = reg.add_point(5, 0)  # circle center
    p1 = reg.add_point(0, 10)  # line start
    p2 = reg.add_point(10, 10)  # line end AND circle radius point
    line = reg.add_line(p1, p2)
    circle = reg.add_circle(p0, p2)
    mutable_pids = [p0, p1, p2]

    constraint = TangentConstraint(line_id=line, shape_id=circle)
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

    x0 = np.array([5, 0, 0, 10, 10, 10], dtype=float)
    diff = check_grad(func_wrapper, grad_wrapper, x0, epsilon=1e-6)
    assert diff < 1e-5


def test_tangent_constraint_serialization_round_trip(setup_env):
    reg, params = setup_env

    # Circle: Center at (0,0), Radius 10
    center = reg.add_point(0, 0)
    radius_pt = reg.add_point(10, 0)
    circ_id = reg.add_circle(center, radius_pt)

    # Line: Horizontal at y=10
    lp1 = reg.add_point(-5, 10)
    lp2 = reg.add_point(5, 10)
    line_id = reg.add_line(lp1, lp2)

    # Create original constraint
    original = TangentConstraint(line_id, circ_id)

    # Serialize to dict
    serialized = original.to_dict()

    # Deserialize from dict
    restored = TangentConstraint.from_dict(serialized)

    # Check that the restored constraint has the same error
    assert original.error(reg, params) == restored.error(reg, params)


def test_tangent_is_hit(setup_env):
    reg, params = setup_env
    center = reg.add_point(50, 50)
    radius_pt = reg.add_point(50, 60)  # radius=10
    circ_id = reg.add_circle(center, radius_pt)

    lp1 = reg.add_point(0, 60)
    lp2 = reg.add_point(100, 60)
    line_id = reg.add_line(lp1, lp2)
    c = TangentConstraint(line_id, circ_id)

    def to_screen(pos):
        return pos

    mock_element = SimpleNamespace()
    threshold = 15.0

    # Tangent point is (50, 60). Normal angle is PI/2. Offset is 12.
    # Symbol pos: (50, 60 + 12) = (50, 72)
    symbol_x, symbol_y = 50, 72

    # Hit
    assert (
        c.is_hit(symbol_x, symbol_y, reg, to_screen, mock_element, threshold)
        is True
    )
    # Miss
    assert c.is_hit(0, 0, reg, to_screen, mock_element, threshold) is False
