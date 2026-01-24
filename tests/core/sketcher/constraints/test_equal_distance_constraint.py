import pytest
import numpy as np
from scipy.optimize import check_grad
from unittest.mock import MagicMock
from rayforge.core.sketcher.constraints import EqualDistanceConstraint
from rayforge.core.sketcher.params import ParameterContext
from rayforge.core.sketcher.registry import EntityRegistry


@pytest.fixture
def setup_env():
    reg = EntityRegistry()
    params = ParameterContext()
    return reg, params


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
    assert c.user_visible is True


def test_equal_distance_constraint_user_visible(setup_env):
    reg, params = setup_env
    p1 = reg.add_point(0, 0)
    p2 = reg.add_point(10, 0)
    p3 = reg.add_point(0, 10)
    p4 = reg.add_point(0, 14)

    c = EqualDistanceConstraint(p1, p2, p3, p4, user_visible=False)
    assert c.user_visible is False

    c2 = EqualDistanceConstraint(p1, p2, p3, p4, user_visible=True)
    assert c2.user_visible is True


def test_equal_distance_targets_segment(setup_env):
    reg, params = setup_env
    p1 = reg.add_point(0, 0)
    p2 = reg.add_point(10, 0)
    p3 = reg.add_point(0, 10)
    p4 = reg.add_point(0, 14)
    p5 = reg.add_point(5, 5)

    c = EqualDistanceConstraint(p1, p2, p3, p4)

    # Matches pair 1 (order independent)
    assert c.targets_segment(p1, p2, None) is True
    assert c.targets_segment(p2, p1, None) is True

    # Matches pair 2 (order independent)
    assert c.targets_segment(p3, p4, None) is True
    assert c.targets_segment(p4, p3, None) is True

    # No match
    assert c.targets_segment(p1, p3, None) is False
    assert c.targets_segment(p1, p5, None) is False


def test_equal_distance_constraint_gradient(setup_env):
    reg, params = setup_env
    p1 = reg.add_point(0, 0)
    p2 = reg.add_point(10, 0)
    p3 = reg.add_point(15, 15)
    p4 = reg.add_point(15, 22)
    mutable_pids = [p1, p2, p3, p4]

    constraint = EqualDistanceConstraint(p1, p2, p3, p4)
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

    x0 = np.array([0, 0, 10, 0, 15, 15, 15, 22], dtype=float)
    diff = check_grad(func_wrapper, grad_wrapper, x0, epsilon=1e-6)
    assert diff < 1e-5


def test_equal_distance_constraint_serialization_round_trip(setup_env):
    reg, params = setup_env
    # Segment 1: Length 10
    p1 = reg.add_point(0, 0)
    p2 = reg.add_point(10, 0)

    # Segment 2: Length 4
    p3 = reg.add_point(0, 10)
    p4 = reg.add_point(0, 14)

    # Create original constraint
    original = EqualDistanceConstraint(p1, p2, p3, p4)

    # Serialize to dict
    serialized = original.to_dict()

    # Deserialize from dict
    restored = EqualDistanceConstraint.from_dict(serialized)

    # Check that the restored constraint has the same error
    assert original.error(reg, params) == restored.error(reg, params)
    assert original.user_visible == restored.user_visible


def test_equal_distance_draw(setup_env):
    reg, params = setup_env
    p1 = reg.add_point(0, 0)
    p2 = reg.add_point(10, 0)
    p3 = reg.add_point(0, 10)
    p4 = reg.add_point(0, 14)

    c = EqualDistanceConstraint(p1, p2, p3, p4)

    ctx = MagicMock()

    def to_screen(pos):
        return pos

    c.draw(ctx, reg, to_screen)
    c.draw(ctx, reg, to_screen, is_selected=True)
    c.draw(ctx, reg, to_screen, is_hovered=True)
    c.draw(ctx, reg, to_screen, point_radius=10.0)
