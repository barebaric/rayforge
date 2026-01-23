import pytest
import numpy as np
from scipy.optimize import check_grad
from rayforge.core.sketcher.constraints import ParallelogramConstraint
from rayforge.core.sketcher.params import ParameterContext
from rayforge.core.sketcher.registry import EntityRegistry


@pytest.fixture
def setup_env():
    reg = EntityRegistry()
    params = ParameterContext()
    return reg, params


def test_parallelogram_constraint_error(setup_env):
    reg, params = setup_env
    # Create a perfect parallelogram: (0,0), (10,0), (0,5), (10,5)
    p_origin = reg.add_point(0, 0)
    p_width = reg.add_point(10, 0)
    p_height = reg.add_point(0, 5)
    p4 = reg.add_point(10, 5)

    c = ParallelogramConstraint(p_origin, p_width, p_height, p4)
    # Vector (p_width-p_origin) = (10, 0)
    # Vector (p4-p_height) = (10, 0)
    # Error = (10-10, 0-0) = (0, 0)
    assert c.error(reg, params) == pytest.approx((0.0, 0.0))


def test_parallelogram_constraint_error_non_parallelogram(setup_env):
    reg, params = setup_env
    # Create a non-parallelogram: (0,0), (10,0), (0,5), (12,6)
    p_origin = reg.add_point(0, 0)
    p_width = reg.add_point(10, 0)
    p_height = reg.add_point(0, 5)
    p4 = reg.add_point(12, 6)

    c = ParallelogramConstraint(p_origin, p_width, p_height, p4)
    # Vector (p_width-p_origin) = (10, 0)
    # Vector (p4-p_height) = (12, 1)
    # Error = (10-12, 0-1) = (-2, -1)
    assert c.error(reg, params) == pytest.approx((-2.0, -1.0))


def test_parallelogram_constraint_gradient(setup_env):
    reg, params = setup_env
    p_origin = reg.add_point(0, 0)
    p_width = reg.add_point(10, 0)
    p_height = reg.add_point(0, 5)
    p4 = reg.add_point(12, 6)

    constraint = ParallelogramConstraint(p_origin, p_width, p_height, p4)
    mutable_pids = [p_origin, p_width, p_height, p4]

    pid_to_idx_map = {pid: i for i, pid in enumerate(mutable_pids)}

    def update_state_from_vec(x_vec):
        for pid, i in pid_to_idx_map.items():
            pt = reg.get_point(pid)
            pt.x = x_vec[i * 2]
            pt.y = x_vec[i * 2 + 1]

    def func_wrapper(x_vec):
        update_state_from_vec(x_vec)
        error = constraint.error(reg, params)
        # Return sum of squared errors for gradient checking
        return error[0] ** 2 + error[1] ** 2

    def grad_wrapper(x_vec):
        update_state_from_vec(x_vec)
        grad_map = constraint.gradient(reg, params)
        grad_vec = np.zeros_like(x_vec)
        for pid, grads in grad_map.items():
            if pid in pid_to_idx_map:
                idx = pid_to_idx_map[pid] * 2
                error = constraint.error(reg, params)
                # grads[0] is (dE_x/dx, dE_x/dy)
                # grads[1] is (dE_y/dx, dE_y/dy)
                # Chain rule: d(E_x^2+E_y^2)/dx = 2*E_x*dE_x/dx + 2*E_y*dE_y/dx
                grad_vec[idx] = (
                    2 * error[0] * grads[0][0] + 2 * error[1] * grads[1][0]
                )
                grad_vec[idx + 1] = (
                    2 * error[0] * grads[0][1] + 2 * error[1] * grads[1][1]
                )
        return grad_vec

    x0 = np.array([0, 0, 10, 0, 0, 5, 12, 6], dtype=float)
    diff = check_grad(func_wrapper, grad_wrapper, x0, epsilon=1e-6)
    assert diff < 1e-5


def test_parallelogram_constraint_gradient_direct(setup_env):
    reg, params = setup_env
    p_origin = reg.add_point(1, 2)
    p_width = reg.add_point(5, 3)
    p_height = reg.add_point(2, 6)
    p4 = reg.add_point(7, 7)

    constraint = ParallelogramConstraint(p_origin, p_width, p_height, p4)

    grad = constraint.gradient(reg, params)

    # Check that all four points are in the gradient
    assert p_origin in grad
    assert p_width in grad
    assert p_height in grad
    assert p4 in grad

    # Check gradient values
    # Error = (v1_x - v2_x, v1_y - v2_y)
    # where v1 = p_width - p_origin, v2 = p4 - p_height
    # For E_x: dE_x/d(p_origin.x) = -1, dE_x/d(p_origin.y) = 0
    #         dE_x/d(p_width.x) = 1, dE_x/d(p_width.y) = 0
    #         dE_x/d(p_height.x) = 1, dE_x/d(p_height.y) = 0
    #         dE_x/d(p4.x) = -1, dE_x/d(p4.y) = 0
    # For E_y: dE_y/d(p_origin.x) = 0, dE_y/d(p_origin.y) = -1
    #         dE_y/d(p_width.x) = 0, dE_y/d(p_width.y) = 1
    #         dE_y/d(p_height.x) = 0, dE_y/d(p_height.y) = 1
    #         dE_y/d(p4.x) = 0, dE_y/d(p4.y) = -1

    assert grad[p_origin] == [(-1.0, 0.0), (0.0, -1.0)]
    assert grad[p_width] == [(1.0, 0.0), (0.0, 1.0)]
    assert grad[p_height] == [(1.0, 0.0), (0.0, 1.0)]
    assert grad[p4] == [(-1.0, 0.0), (0.0, -1.0)]


def test_parallelogram_constraint_user_visible_false(setup_env):
    reg, params = setup_env
    p_origin = reg.add_point(0, 0)
    p_width = reg.add_point(10, 0)
    p_height = reg.add_point(0, 5)
    p4 = reg.add_point(10, 5)

    c = ParallelogramConstraint(p_origin, p_width, p_height, p4)
    assert c.user_visible is False


def test_parallelogram_constraint_user_visible_true(setup_env):
    reg, params = setup_env
    p_origin = reg.add_point(0, 0)
    p_width = reg.add_point(10, 0)
    p_height = reg.add_point(0, 5)
    p4 = reg.add_point(10, 5)

    c = ParallelogramConstraint(
        p_origin, p_width, p_height, p4, user_visible=True
    )
    assert c.user_visible is True


def test_parallelogram_constraint_serialization_round_trip(setup_env):
    reg, params = setup_env
    p_origin = reg.add_point(0, 0)
    p_width = reg.add_point(10, 0)
    p_height = reg.add_point(0, 5)
    p4 = reg.add_point(10, 5)

    original = ParallelogramConstraint(p_origin, p_width, p_height, p4)

    serialized = original.to_dict()

    restored = ParallelogramConstraint.from_dict(serialized)

    assert original.user_visible == restored.user_visible
    assert original.p_origin == restored.p_origin
    assert original.p_width == restored.p_width
    assert original.p_height == restored.p_height
    assert original.p4 == restored.p4

    assert original.error(reg, params) == restored.error(reg, params)


def test_parallelogram_constraint_serialization_includes_user_visible(
    setup_env,
):
    reg, params = setup_env
    p_origin = reg.add_point(0, 0)
    p_width = reg.add_point(10, 0)
    p_height = reg.add_point(0, 5)
    p4 = reg.add_point(10, 5)

    c = ParallelogramConstraint(p_origin, p_width, p_height, p4)
    serialized = c.to_dict()

    assert "user_visible" in serialized
    assert serialized["user_visible"] is False
