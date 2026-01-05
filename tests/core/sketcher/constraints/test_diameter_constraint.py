import pytest
import numpy as np
from scipy.optimize import check_grad
from types import SimpleNamespace
from unittest.mock import MagicMock
from rayforge.core.sketcher.constraints import DiameterConstraint
from rayforge.core.sketcher.params import ParameterContext
from rayforge.core.sketcher.registry import EntityRegistry


@pytest.fixture
def setup_env():
    reg = EntityRegistry()
    params = ParameterContext()
    return reg, params


def test_diameter_constraint(setup_env):
    reg, params = setup_env
    center = reg.add_point(0, 0)
    # Radius is 5, so diameter is 10
    radius_pt = reg.add_point(5, 0)
    circ_id = reg.add_circle(center, radius_pt)

    # Target diam is 10, actual is 10. Error = 10 - 10 = 0
    c = DiameterConstraint(circ_id, 10.0)
    assert c.error(reg, params) == pytest.approx(0.0)

    # Target diam is 20, actual is 10. Error = 10 - 20 = -10
    c2 = DiameterConstraint(circ_id, 20.0)
    assert c2.error(reg, params) == pytest.approx(-10.0)


def test_diameter_constrains_radius_method(setup_env):
    reg, params = setup_env
    center = reg.add_point(0, 0)
    radius_pt = reg.add_point(5, 0)
    circ_id = reg.add_circle(center, radius_pt)
    other_circ_id = reg.add_circle(center, radius_pt)

    c = DiameterConstraint(circ_id, 10.0)

    # Should return True for the constrained circle
    assert c.constrains_radius(reg, circ_id) is True
    # Should return False for any other entity
    assert c.constrains_radius(reg, other_circ_id) is False
    assert c.constrains_radius(reg, 999) is False


def test_diameter_constraint_serialization_round_trip(setup_env):
    reg, params = setup_env
    center = reg.add_point(0, 0)
    radius_pt = reg.add_point(5, 0)
    circ_id = reg.add_circle(center, radius_pt)

    # Create original constraint
    original = DiameterConstraint(circ_id, 10.0)

    # Serialize to dict
    serialized = original.to_dict()

    # Deserialize from dict
    restored = DiameterConstraint.from_dict(serialized)

    # Check that the restored constraint has the same error
    assert original.error(reg, params) == restored.error(reg, params)


def test_diameter_constraint_with_expression(setup_env):
    reg, params = setup_env
    center = reg.add_point(0, 0)
    # Radius is 5, so diameter is 10
    radius_pt = reg.add_point(5, 0)
    circ_id = reg.add_circle(center, radius_pt)

    # Define context
    ctx = {"target_diam": 20.0}

    # Create constraint with expression
    c = DiameterConstraint(circ_id, "target_diam")

    # Verify initial state (value is 0.0 before update)
    assert c.value == 0.0

    # Update from context
    c.update_from_context(ctx)
    assert c.value == 20.0

    # Actual diam is 10, Target is 20. Error = 10 - 20 = -10
    assert c.error(reg, params) == pytest.approx(-10.0)


def test_diameter_constraint_gradient(setup_env):
    reg, params = setup_env
    c_id = reg.add_point(0, 0)
    r_id = reg.add_point(5, 0)
    circ_id = reg.add_circle(c_id, r_id)
    mutable_pids = [c_id, r_id]

    constraint = DiameterConstraint(circle_id=circ_id, value=12.0)
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

    x0 = np.array([0, 0, 5, 0], dtype=float)
    diff = check_grad(func_wrapper, grad_wrapper, x0, epsilon=1e-6)
    assert diff < 1e-5


def test_diameter_is_hit(setup_env):
    reg, params = setup_env
    center = reg.add_point(0, 0)
    radius_pt = reg.add_point(100, 0)  # radius=100
    circ_id = reg.add_circle(center, radius_pt)

    c = DiameterConstraint(circ_id, 200)

    # Mock the canvas and element
    mock_canvas = MagicMock()
    mock_canvas.get_view_scale.return_value = (1.0, 1.0)
    mock_element = SimpleNamespace(canvas=mock_canvas)

    def to_screen(pos):
        return pos

    threshold = 15.0

    # Label pos logic places it at (radius + 20) along radius vector
    # (100 + 20, 0) = (120, 0)
    label_pos_x, label_pos_y = 120, 0

    # Hit
    assert (
        c.is_hit(
            label_pos_x, label_pos_y, reg, to_screen, mock_element, threshold
        )
        is True
    )
    # Miss
    assert c.is_hit(0, 0, reg, to_screen, mock_element, threshold) is False
