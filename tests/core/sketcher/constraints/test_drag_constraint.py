import pytest
import numpy as np
from scipy.optimize import check_grad
from functools import partial

from rayforge.core.sketcher.params import ParameterContext
from rayforge.core.sketcher.entities import EntityRegistry
from rayforge.core.sketcher.constraints import DragConstraint


@pytest.fixture
def setup_env():
    reg = EntityRegistry()
    params = ParameterContext()
    return reg, params


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


def test_drag_constraint_gradient(setup_env):
    reg, params = setup_env
    p1_id = reg.add_point(1, 2)
    mutable_pids = [p1_id]

    constraint = DragConstraint(
        point_id=p1_id, target_x=10, target_y=10, weight=0.5
    )
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

    x0 = np.array([1, 2], dtype=float)
    for i in range(2):
        func = partial(func_wrapper, error_index=i)
        grad = partial(grad_wrapper, error_index=i)
        diff = check_grad(func, grad, x0, epsilon=1e-6)
        assert diff < 1e-5


def test_drag_constraint_serialization_round_trip(setup_env):
    reg, params = setup_env
    p1 = reg.add_point(0, 0)

    # Create original constraint
    original = DragConstraint(p1, 100.0, 0.0, weight=0.1)

    # Serialize to dict
    serialized = original.to_dict()

    # DragConstraint returns empty dict from to_dict() as it's not meant to be
    # serialized
    assert serialized == {}
