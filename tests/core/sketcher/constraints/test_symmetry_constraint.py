import pytest
import numpy as np
from scipy.optimize import check_grad
from functools import partial
from types import SimpleNamespace
from rayforge.core.sketcher.params import ParameterContext
from rayforge.core.sketcher.entities import EntityRegistry
from rayforge.core.sketcher.constraints import SymmetryConstraint


@pytest.fixture
def setup_env():
    reg = EntityRegistry()
    params = ParameterContext()
    return reg, params


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


def test_symmetry_constraint_point_gradient(setup_env):
    reg, params = setup_env
    p1 = reg.add_point(-5, 2)
    p2 = reg.add_point(5, -3)
    c = reg.add_point(1, -1)
    mutable_pids = [p1, p2, c]

    constraint = SymmetryConstraint(p1=p1, p2=p2, center=c)
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

    x0 = np.array([-5, 2, 5, -3, 1, -1], dtype=float)
    for i in range(2):
        func = partial(func_wrapper, error_index=i)
        grad = partial(grad_wrapper, error_index=i)
        diff = check_grad(func, grad, x0, epsilon=1e-6)
        assert diff < 1e-5


def test_symmetry_constraint_line_gradient(setup_env):
    reg, params = setup_env
    p1 = reg.add_point(-5, 10)
    p2 = reg.add_point(6, 11)
    l1 = reg.add_point(0, 0)
    l2 = reg.add_point(0, 20)
    axis = reg.add_line(l1, l2)
    mutable_pids = [p1, p2, l1, l2]

    constraint = SymmetryConstraint(p1=p1, p2=p2, axis=axis)
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

    x0 = np.array([-5, 10, 6, 11, 0, 0, 0, 20], dtype=float)
    for i in range(2):
        func = partial(func_wrapper, error_index=i)
        grad = partial(grad_wrapper, error_index=i)
        diff = check_grad(func, grad, x0, epsilon=1e-6)
        assert diff < 1e-5


def test_symmetry_constraint_serialization_round_trip(setup_env):
    reg, params = setup_env
    # Center at (0,0)
    pc = reg.add_point(0, 0)
    # P1 at (-5, -2)
    p1 = reg.add_point(-5, -2)
    # P2 at (5, 2)
    p2 = reg.add_point(5, 2)

    # Create original constraint
    original = SymmetryConstraint(p1, p2, center=pc)

    # Serialize to dict
    serialized = original.to_dict()

    # Deserialize from dict
    restored = SymmetryConstraint.from_dict(serialized)

    # Check that the restored constraint has the same error
    assert original.error(reg, params) == restored.error(reg, params)


def test_symmetry_constraint_line_serialization_round_trip(setup_env):
    reg, params = setup_env
    l1 = reg.add_point(0, -10)
    l2 = reg.add_point(0, 10)
    axis_id = reg.add_line(l1, l2)
    p1 = reg.add_point(-5, 5)
    p2 = reg.add_point(5, 5)
    original = SymmetryConstraint(p1, p2, axis=axis_id)
    serialized = original.to_dict()
    restored = SymmetryConstraint.from_dict(serialized)
    assert original.error(reg, params) == restored.error(reg, params)


def test_symmetry_is_hit(setup_env):
    reg, params = setup_env
    p1 = reg.add_point(-50, 0)
    p2 = reg.add_point(50, 0)
    c = SymmetryConstraint(p1, p2)

    def to_screen(pos):
        return pos

    mock_element = SimpleNamespace()
    threshold = 15.0

    # Midpoint is (0,0). Angle is 0. Offset is 12.
    # Symbol points are at (-12, 0) and (12, 0)
    # Hit left symbol
    assert c.is_hit(-12, 0, reg, to_screen, mock_element, threshold) is True
    # Hit right symbol
    assert c.is_hit(12, 0, reg, to_screen, mock_element, threshold) is True
    # Miss
    assert c.is_hit(50, 50, reg, to_screen, mock_element, threshold) is False
