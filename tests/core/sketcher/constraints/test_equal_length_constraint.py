import pytest
import numpy as np
from functools import partial
from scipy.optimize import check_grad
from types import SimpleNamespace
from unittest.mock import MagicMock
from rayforge.core.sketcher.constraints import EqualLengthConstraint
from rayforge.core.sketcher.params import ParameterContext
from rayforge.core.sketcher.registry import EntityRegistry


@pytest.fixture
def setup_env():
    reg = EntityRegistry()
    params = ParameterContext()
    return reg, params


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


def test_equal_length_constrains_radius_method(setup_env):
    reg, params = setup_env
    p1 = reg.add_point(0, 0)
    p2 = reg.add_point(10, 0)
    l1 = reg.add_line(p1, p2)

    p3 = reg.add_point(0, 10)
    p4 = reg.add_point(5, 10)
    l2 = reg.add_line(p3, p4)

    # Entity 999 (not in list)

    c = EqualLengthConstraint([l1, l2])

    # Should return True for entities in the list
    assert c.constrains_radius(reg, l1) is True
    assert c.constrains_radius(reg, l2) is True

    # Should return False for entities not in the list
    assert c.constrains_radius(reg, 999) is False


def test_equal_length_constraint_gradient(setup_env):
    reg, params = setup_env
    p0 = reg.add_point(0, 0)
    p1 = reg.add_point(10, 0)
    p2 = reg.add_point(15, 15)
    p3 = reg.add_point(15, 22)
    l1 = reg.add_line(p0, p1)
    l2 = reg.add_line(p2, p3)
    mutable_pids = [p0, p1, p2, p3]

    constraint = EqualLengthConstraint(entity_ids=[l1, l2])

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

    x0 = np.array([0, 0, 10, 0, 15, 15, 15, 22], dtype=float)
    func = partial(func_wrapper, error_index=0)
    grad = partial(grad_wrapper, error_index=0)
    diff = check_grad(func, grad, x0, epsilon=1e-6)
    assert diff < 1e-5


def test_equal_length_constraint_serialization_round_trip(setup_env):
    reg, params = setup_env
    # Line 1: length 10
    p1 = reg.add_point(0, 0)
    p2 = reg.add_point(10, 0)
    l1 = reg.add_line(p1, p2)

    # Line 2: length 5
    p3 = reg.add_point(0, 10)
    p4 = reg.add_point(5, 10)
    l2 = reg.add_line(p3, p4)

    # Create original constraint
    original = EqualLengthConstraint([l1, l2])

    # Serialize to dict
    serialized = original.to_dict()

    # Deserialize from dict
    restored = EqualLengthConstraint.from_dict(serialized)

    # Check that the restored constraint has the same error
    assert original.error(reg, params) == restored.error(reg, params)


def test_equal_length_is_hit(setup_env):
    reg, params = setup_env
    p1 = reg.add_point(0, 0)
    p2 = reg.add_point(100, 0)
    l1 = reg.add_line(p1, p2)

    p3 = reg.add_point(0, 50)
    p4 = reg.add_point(0, 150)
    l2 = reg.add_line(p3, p4)

    c = EqualLengthConstraint([l1, l2])

    # Mock canvas and element
    mock_canvas = MagicMock()
    mock_canvas.get_view_scale.return_value = (1.0, 1.0)
    mock_element = SimpleNamespace(canvas=mock_canvas)

    def to_screen(pos):
        return pos

    threshold = 15.0

    # Symbol pos for line 1: midpoint (50, 0), normal is (0, -1), offset 15
    # -> (50, -15)
    l1_symbol_x, l1_symbol_y = 50, -15
    # Symbol pos for line 2: midpoint (0, 100), tangent angle pi/2, normal
    # angle 0 -> (15, 100)
    l2_symbol_x, l2_symbol_y = 15, 100

    # Hit line 1
    assert (
        c.is_hit(
            l1_symbol_x, l1_symbol_y, reg, to_screen, mock_element, threshold
        )
        is True
    )
    # Hit line 2
    assert (
        c.is_hit(
            l2_symbol_x, l2_symbol_y, reg, to_screen, mock_element, threshold
        )
        is True
    )
    # Miss both
    assert c.is_hit(0, 0, reg, to_screen, mock_element, threshold) is False
