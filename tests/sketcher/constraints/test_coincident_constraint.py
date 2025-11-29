import pytest
import numpy as np
from scipy.optimize import check_grad
from functools import partial

from rayforge.core.sketcher.params import ParameterContext
from rayforge.core.sketcher.entities import EntityRegistry
from rayforge.core.sketcher.constraints import CoincidentConstraint


@pytest.fixture
def setup_env():
    reg = EntityRegistry()
    params = ParameterContext()
    return reg, params


def test_coincident_constraint(setup_env):
    reg, params = setup_env
    p1 = reg.add_point(0, 0)
    p2 = reg.add_point(3, 4)

    c = CoincidentConstraint(p1, p2)
    # Error is a tuple (dx, dy)
    assert c.error(reg, params) == (0.0 - 3.0, 0.0 - 4.0)


def test_coincident_constraint_gradient(setup_env):
    reg, params = setup_env
    p1_id = reg.add_point(1, 2)
    p2_id = reg.add_point(5, 6)
    mutable_pids = [p1_id, p2_id]

    constraint = CoincidentConstraint(p1=p1_id, p2=p2_id)

    pid_to_idx_map = {pid: i for i, pid in enumerate(mutable_pids)}

    def update_state_from_vec(x_vec):
        for pid, i in pid_to_idx_map.items():
            pt = reg.get_point(pid)
            pt.x = x_vec[i * 2]
            pt.y = x_vec[i * 2 + 1]

    def func_wrapper(x_vec, error_index=0):
        update_state_from_vec(x_vec)
        err = constraint.error(reg, params)
        return err[error_index]

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

    x0 = np.array([1, 2, 5, 6], dtype=float)

    for i in range(2):  # Two error components for this constraint
        func = partial(func_wrapper, error_index=i)
        grad = partial(grad_wrapper, error_index=i)
        diff = check_grad(func, grad, x0, epsilon=1e-6)
        assert diff < 1e-5
