import pytest
import numpy as np
from scipy.optimize import check_grad
import math
from types import SimpleNamespace
from unittest.mock import MagicMock

from rayforge.core.sketcher.constraints import RadiusConstraint
from rayforge.core.sketcher.params import ParameterContext
from rayforge.core.sketcher.registry import EntityRegistry


@pytest.fixture
def setup_env():
    reg = EntityRegistry()
    params = ParameterContext()
    return reg, params


def test_radius_constraint(setup_env):
    reg, params = setup_env
    start = reg.add_point(10, 0)
    end = reg.add_point(0, 10)
    center = reg.add_point(0, 0)
    arc_id = reg.add_arc(start, end, center)

    # Current radius is 10. Target is 10. Error = 10 - 10 = 0
    c = RadiusConstraint(arc_id, 10.0)
    assert c.error(reg, params) == pytest.approx(0.0)

    # Target is 5. Error = 10 - 5 = 5
    c2 = RadiusConstraint(arc_id, 5.0)
    assert c2.error(reg, params) == pytest.approx(5.0)


def test_radius_constraint_on_circle(setup_env):
    reg, params = setup_env
    center = reg.add_point(0, 0)
    radius_pt = reg.add_point(10, 0)
    circ_id = reg.add_circle(center, radius_pt)

    # Current radius is 10. Target is 10. Error = 10 - 10 = 0
    c = RadiusConstraint(circ_id, 10.0)
    assert c.error(reg, params) == pytest.approx(0.0)

    # Target is 5. Error = 10 - 5 = 5
    c2 = RadiusConstraint(circ_id, 5.0)
    assert c2.error(reg, params) == pytest.approx(5.0)


def test_radius_constrains_radius_method(setup_env):
    reg, params = setup_env
    start = reg.add_point(10, 0)
    end = reg.add_point(0, 10)
    center = reg.add_point(0, 0)
    arc_id = reg.add_arc(start, end, center)
    other_id = reg.add_arc(start, end, center)

    c = RadiusConstraint(arc_id, 10.0)

    # Should return True for the constrained entity
    assert c.constrains_radius(reg, arc_id) is True
    # Should return False for others
    assert c.constrains_radius(reg, other_id) is False
    assert c.constrains_radius(reg, 999) is False


def test_radius_constraint_invalid_entity(setup_env):
    reg, params = setup_env
    # Add a line, try to constrain radius (should fail gracefully/return 0)
    p1 = reg.add_point(0, 0)
    p2 = reg.add_point(10, 10)
    line_id = reg.add_line(p1, p2)

    c = RadiusConstraint(line_id, 5.0)
    assert c.error(reg, params) == 0.0


def test_radius_constraint_gradient(setup_env):
    reg, params = setup_env
    c_id = reg.add_point(5, 5)
    s_id = reg.add_point(15, 5)
    e_id = reg.add_point(5, 15)
    arc_id = reg.add_arc(s_id, e_id, c_id)
    mutable_pids = [c_id, s_id]

    constraint = RadiusConstraint(entity_id=arc_id, value=12.0)
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

    x0 = np.array([5, 5, 15, 5], dtype=float)
    diff = check_grad(func_wrapper, grad_wrapper, x0, epsilon=1e-6)
    assert diff < 1e-5


def test_radius_constraint_serialization_round_trip(setup_env):
    reg, params = setup_env
    center = reg.add_point(0, 0)
    radius_pt = reg.add_point(10, 0)
    circ_id = reg.add_circle(center, radius_pt)

    # Create original constraint
    original = RadiusConstraint(circ_id, 10.0)

    # Serialize to dict
    serialized = original.to_dict()

    # Deserialize from dict
    restored = RadiusConstraint.from_dict(serialized)

    # Check that the restored constraint has the same error
    assert original.error(reg, params) == restored.error(reg, params)


def test_radius_constraint_with_expression(setup_env):
    reg, params = setup_env
    center = reg.add_point(0, 0)
    radius_pt = reg.add_point(5, 0)
    circ_id = reg.add_circle(center, radius_pt)

    ctx = {"r_val": 10.0}
    c = RadiusConstraint(circ_id, "r_val")
    c.update_from_context(ctx)

    # Current r=5, target=10. Error = 5 - 10 = -5
    assert c.error(reg, params) == pytest.approx(-5.0)


def test_radius_is_hit(setup_env):
    reg, params = setup_env
    start = reg.add_point(100, 0)
    end = reg.add_point(0, 100)
    center = reg.add_point(0, 0)
    arc_id = reg.add_arc(start, end, center)

    c = RadiusConstraint(arc_id, 100)

    # Mock the canvas and element
    mock_canvas = MagicMock()
    mock_canvas.get_view_scale.return_value = (1.0, 1.0)
    mock_element = SimpleNamespace(canvas=mock_canvas)

    def to_screen(pos):
        return pos

    threshold = 15.0

    # Label pos for arc is at (radius + 20) along midpoint vector
    # Mid-angle is 45deg.
    # pos = (120 * cos(45), 120 * sin(45)) ~ (84.85, 84.85)
    dist = 120
    angle = math.pi / 4
    label_pos_x, label_pos_y = dist * math.cos(angle), dist * math.sin(angle)

    # Hit
    assert (
        c.is_hit(
            label_pos_x, label_pos_y, reg, to_screen, mock_element, threshold
        )
        is True
    )
    # Miss
    assert c.is_hit(0, 0, reg, to_screen, mock_element, threshold) is False
