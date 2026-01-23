import pytest
import numpy as np
from scipy.optimize import check_grad
from rayforge.core.sketcher.constraints import AspectRatioConstraint
from rayforge.core.sketcher.params import ParameterContext
from rayforge.core.sketcher.registry import EntityRegistry


@pytest.fixture
def setup_env():
    reg = EntityRegistry()
    params = ParameterContext()
    return reg, params


def test_aspect_ratio_constraint(setup_env):
    reg, params = setup_env
    p1 = reg.add_point(0, 0)
    p2 = reg.add_point(10, 0)
    p3 = reg.add_point(0, 10)
    p4 = reg.add_point(5, 10)

    c = AspectRatioConstraint(p1, p2, p3, p4, 2.0)
    dist1 = 10.0
    dist2 = 5.0
    expected_error = dist1 - dist2 * 2.0
    assert c.error(reg, params) == pytest.approx(expected_error)


def test_aspect_ratio_constraint_perfect_match(setup_env):
    reg, params = setup_env
    p1 = reg.add_point(0, 0)
    p2 = reg.add_point(10, 0)
    p3 = reg.add_point(0, 10)
    p4 = reg.add_point(5, 10)

    c = AspectRatioConstraint(p1, p2, p3, p4, 2.0)
    assert c.error(reg, params) == pytest.approx(0.0)


def test_aspect_ratio_constraint_diagonal_lines(setup_env):
    reg, params = setup_env
    p1 = reg.add_point(0, 0)
    p2 = reg.add_point(6, 8)
    p3 = reg.add_point(10, 10)
    p4 = reg.add_point(13, 14)

    c = AspectRatioConstraint(p1, p2, p3, p4, 1.0)
    dist1 = 10.0
    dist2 = 5.0
    expected_error = dist1 - dist2 * 1.0
    assert c.error(reg, params) == pytest.approx(expected_error)


def test_aspect_ratio_constraint_zero_ratio(setup_env):
    reg, params = setup_env
    p1 = reg.add_point(0, 0)
    p2 = reg.add_point(10, 0)
    p3 = reg.add_point(0, 10)
    p4 = reg.add_point(5, 10)

    c = AspectRatioConstraint(p1, p2, p3, p4, 0.0)
    dist1 = 10.0
    expected_error = dist1
    assert c.error(reg, params) == pytest.approx(expected_error)


def test_aspect_ratio_constraint_zero_second_distance(setup_env):
    reg, params = setup_env
    p1 = reg.add_point(0, 0)
    p2 = reg.add_point(10, 0)
    p3 = reg.add_point(0, 10)
    p4 = reg.add_point(0, 10)

    c = AspectRatioConstraint(p1, p2, p3, p4, 2.0)
    dist1 = 10.0
    expected_error = dist1
    assert c.error(reg, params) == pytest.approx(expected_error)


def test_aspect_ratio_constrains_radius_method(setup_env):
    reg, params = setup_env
    p1 = reg.add_point(0, 0)
    p2 = reg.add_point(10, 0)
    p3 = reg.add_point(0, 10)
    p4 = reg.add_point(5, 10)
    c = AspectRatioConstraint(p1, p2, p3, p4, 2.0)

    assert c.constrains_radius(reg, 999) is False
    assert c.constrains_radius(reg, p1) is False


def test_aspect_ratio_constraint_gradient(setup_env):
    reg, params = setup_env
    p1_id = reg.add_point(1, 2)
    p2_id = reg.add_point(5, 6)
    p3_id = reg.add_point(0, 0)
    p4_id = reg.add_point(3, 0)
    mutable_pids = [p1_id, p2_id, p3_id, p4_id]

    constraint = AspectRatioConstraint(
        p1=p1_id, p2=p2_id, p3=p3_id, p4=p4_id, ratio=2.0
    )

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

    x0 = np.array([1, 2, 5, 6, 0, 0, 3, 0], dtype=float)
    diff = check_grad(func_wrapper, grad_wrapper, x0, epsilon=1e-6)
    assert diff < 1e-5


def test_aspect_ratio_constraint_gradient_zero_first_distance(setup_env):
    reg, params = setup_env
    p1_id = reg.add_point(0, 0)
    p2_id = reg.add_point(0, 0)
    p3_id = reg.add_point(0, 0)
    p4_id = reg.add_point(3, 0)

    constraint = AspectRatioConstraint(
        p1=p1_id, p2=p2_id, p3=p3_id, p4=p4_id, ratio=2.0
    )

    grad_map = constraint.gradient(reg, params)
    assert p1_id not in grad_map
    assert p2_id not in grad_map


def test_aspect_ratio_constraint_gradient_zero_second_distance(setup_env):
    reg, params = setup_env
    p1_id = reg.add_point(0, 0)
    p2_id = reg.add_point(3, 0)
    p3_id = reg.add_point(0, 0)
    p4_id = reg.add_point(0, 0)

    constraint = AspectRatioConstraint(
        p1=p1_id, p2=p2_id, p3=p3_id, p4=p4_id, ratio=2.0
    )

    grad_map = constraint.gradient(reg, params)
    assert p3_id not in grad_map
    assert p4_id not in grad_map


def test_aspect_ratio_constraint_serialization_round_trip(setup_env):
    reg, params = setup_env
    p1 = reg.add_point(0, 0)
    p2 = reg.add_point(10, 0)
    p3 = reg.add_point(0, 10)
    p4 = reg.add_point(5, 10)

    original = AspectRatioConstraint(p1, p2, p3, p4, 2.5)

    serialized = original.to_dict()

    restored = AspectRatioConstraint.from_dict(serialized)

    assert original.error(reg, params) == restored.error(reg, params)
    assert original.p1 == restored.p1
    assert original.p2 == restored.p2
    assert original.p3 == restored.p3
    assert original.p4 == restored.p4
    assert original.ratio == restored.ratio


def test_aspect_ratio_constraint_serialization_dict_format(setup_env):
    reg, params = setup_env
    p1 = reg.add_point(0, 0)
    p2 = reg.add_point(10, 0)
    p3 = reg.add_point(0, 10)
    p4 = reg.add_point(5, 10)

    c = AspectRatioConstraint(p1, p2, p3, p4, 2.5)

    data = c.to_dict()

    assert data["type"] == "AspectRatioConstraint"
    assert data["p1"] == p1
    assert data["p2"] == p2
    assert data["p3"] == p3
    assert data["p4"] == p4
    assert data["ratio"] == 2.5


def test_aspect_ratio_depends_on_points(setup_env):
    reg, params = setup_env
    p1 = reg.add_point(0, 0)
    p2 = reg.add_point(10, 0)
    p3 = reg.add_point(0, 10)
    p4 = reg.add_point(5, 10)

    c = AspectRatioConstraint(p1, p2, p3, p4, 2.0)

    assert c.depends_on_points({p1}) is True
    assert c.depends_on_points({p2}) is True
    assert c.depends_on_points({p3}) is True
    assert c.depends_on_points({p4}) is True
    assert c.depends_on_points({p1, p2}) is True
    assert c.depends_on_points({999}) is False
    assert c.depends_on_points(set()) is False


def test_aspect_ratio_depends_on_entities(setup_env):
    reg, params = setup_env
    p1 = reg.add_point(0, 0)
    p2 = reg.add_point(10, 0)
    p3 = reg.add_point(0, 10)
    p4 = reg.add_point(5, 10)

    c = AspectRatioConstraint(p1, p2, p3, p4, 2.0)

    assert c.depends_on_entities({999}) is False
    assert c.depends_on_entities(set()) is False
