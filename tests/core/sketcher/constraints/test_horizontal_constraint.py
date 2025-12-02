import pytest
import numpy as np
from scipy.optimize import check_grad
from types import SimpleNamespace

from rayforge.core.sketcher.params import ParameterContext
from rayforge.core.sketcher.entities import EntityRegistry
from rayforge.core.sketcher.constraints import HorizontalConstraint


@pytest.fixture
def setup_env():
    reg = EntityRegistry()
    params = ParameterContext()
    return reg, params


def test_horizontal_constraint(setup_env):
    reg, params = setup_env
    # p1 y=0, p2 y=5. Error should be 0 - 5 = -5
    p1 = reg.add_point(0, 0)
    p2 = reg.add_point(10, 5)

    c = HorizontalConstraint(p1, p2)
    assert c.error(reg, params) == pytest.approx(-5.0)


def test_horizontal_constraint_gradient(setup_env):
    reg, params = setup_env
    p1_id = reg.add_point(1, 2)
    p2_id = reg.add_point(5, 6)
    mutable_pids = [p1_id, p2_id]

    constraint = HorizontalConstraint(p1=p1_id, p2=p2_id)

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


def test_horizontal_constraint_serialization_round_trip(setup_env):
    reg, params = setup_env
    p1 = reg.add_point(0, 0)
    p2 = reg.add_point(10, 5)

    # Create original constraint
    original = HorizontalConstraint(p1, p2)

    # Serialize to dict
    serialized = original.to_dict()

    # Deserialize from dict
    restored = HorizontalConstraint.from_dict(serialized)

    # Check that the restored constraint has the same error
    assert original.error(reg, params) == restored.error(reg, params)


def test_horizontal_is_hit(setup_env):
    reg, params = setup_env
    p1 = reg.add_point(0, 50)
    p2 = reg.add_point(100, 50)
    c = HorizontalConstraint(p1, p2)

    def to_screen(pos):
        return pos

    mock_element = SimpleNamespace()
    threshold = 15.0

    # Symbol is at t=0.2 along line, offset by -10 in Y
    # Point is at x=20, y=50. Symbol at x=20, y=40.
    symbol_x, symbol_y = 20, 40

    # Hit
    assert (
        c.is_hit(symbol_x, symbol_y, reg, to_screen, mock_element, threshold)
        is True
    )
    # Miss
    assert c.is_hit(0, 0, reg, to_screen, mock_element, threshold) is False
