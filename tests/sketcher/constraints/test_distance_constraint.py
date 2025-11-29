import pytest
import numpy as np
from scipy.optimize import check_grad
from rayforge.core.sketcher.params import ParameterContext
from rayforge.core.sketcher.entities import EntityRegistry
from rayforge.core.sketcher.constraints import DistanceConstraint


@pytest.fixture
def setup_env():
    reg = EntityRegistry()
    params = ParameterContext()
    return reg, params


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


def test_distance_constraint_gradient(setup_env):
    """
    Uses scipy.optimize.check_grad to numerically verify the analytical
    gradient of the DistanceConstraint.
    """
    reg, params = setup_env
    p1_id = reg.add_point(1, 2)
    p2_id = reg.add_point(5, 6)
    mutable_pids = [p1_id, p2_id]

    constraint = DistanceConstraint(p1=p1_id, p2=p2_id, value=10.0)

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

    x0 = np.array([1, 2, 5, 6], dtype=float)
    diff = check_grad(func_wrapper, grad_wrapper, x0, epsilon=1e-6)
    assert diff < 1e-5
