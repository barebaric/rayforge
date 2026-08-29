from unittest.mock import MagicMock

import numpy as np
import pytest
from scipy.optimize import check_grad
from sketcher.core import Sketch
from sketcher.core.constraints import PointOnCurveConstraint
from sketcher.core.constraints.point_on_curve import closest_point_on_bezier
from sketcher.core.params import ParameterContext
from sketcher.core.registry import EntityRegistry
from sketcher.core.selection import SketchSelection
from sketcher.core.solver import Solver

# Hump-shaped curve: (0,0) -> (10,0) with both control points pulling
# the curve up to an apex of (5, 7.5).
X = (0.0, 0.0, 10.0, 10.0)
Y = (0.0, 10.0, 10.0, 0.0)


@pytest.fixture
def setup_env():
    reg = EntityRegistry()
    params = ParameterContext()
    return reg, params


def add_hump_bezier(reg, start=(0.0, 0.0), end=(10.0, 0.0)):
    p_start = reg.add_point(*start)
    p_end = reg.add_point(*end)
    bez_id = reg.add_bezier(p_start, p_end, cp1=(0.0, 10.0), cp2=(0.0, 10.0))
    return bez_id, p_start, p_end


def test_closest_point_apex():
    t, bx, by = closest_point_on_bezier(X, Y, 5.0, 8.0)
    assert t == pytest.approx(0.5, abs=1e-6)
    assert bx == pytest.approx(5.0, abs=1e-6)
    assert by == pytest.approx(7.5, abs=1e-6)


def test_closest_point_clamps_to_start():
    t, bx, by = closest_point_on_bezier(X, Y, -3.0, 0.0)
    assert t == pytest.approx(0.0)
    assert bx == pytest.approx(0.0)
    assert by == pytest.approx(0.0)


def test_closest_point_clamps_to_end():
    t, bx, by = closest_point_on_bezier(X, Y, 14.0, 0.0)
    assert t == pytest.approx(1.0)
    assert bx == pytest.approx(10.0)
    assert by == pytest.approx(0.0)


def test_point_on_curve_error(setup_env):
    reg, params = setup_env
    bez_id, _start, _end = add_hump_bezier(reg)
    pt = reg.add_point(5.0, 8.0)

    c = PointOnCurveConstraint(pt, bez_id)
    assert c.error(reg, params) == pytest.approx(0.5, abs=1e-6)


def test_point_on_curve_error_zero_on_curve(setup_env):
    reg, params = setup_env
    bez_id, _start, _end = add_hump_bezier(reg)
    pt = reg.add_point(5.0, 7.5)

    c = PointOnCurveConstraint(pt, bez_id)
    assert c.error(reg, params) < 1e-9


def test_point_on_curve_gradient(setup_env):
    reg, params = setup_env
    bez_id, p_start, p_end = add_hump_bezier(reg)
    pt = reg.add_point(5.0, 8.0)
    mutable_pids = [pt, p_start, p_end]

    constraint = PointOnCurveConstraint(point_id=pt, shape_id=bez_id)
    pid_to_idx = {pid: i for i, pid in enumerate(mutable_pids)}

    def update_state_from_vec(x_vec):
        for pid, i in pid_to_idx.items():
            point = reg.get_point(pid)
            point.x = x_vec[i * 2]
            point.y = x_vec[i * 2 + 1]

    def func_wrapper(x_vec):
        update_state_from_vec(x_vec)
        return constraint.error(reg, params)

    def grad_wrapper(x_vec):
        update_state_from_vec(x_vec)
        grad_map = constraint.gradient(reg, params)
        grad_vec = np.zeros_like(x_vec)
        for pid, grads in grad_map.items():
            if pid in pid_to_idx:
                idx = pid_to_idx[pid] * 2
                dx, dy = grads[0]
                grad_vec[idx] = dx
                grad_vec[idx + 1] = dy
        return grad_vec

    x0 = np.array([5.0, 8.0, 0.0, 0.0, 10.0, 0.0])

    diff = check_grad(func_wrapper, grad_wrapper, x0, epsilon=1e-7)
    assert diff < 1e-4


def test_point_on_curve_gradient_zero_at_solution(setup_env):
    reg, params = setup_env
    bez_id, _start, _end = add_hump_bezier(reg)
    pt = reg.add_point(5.0, 7.5)

    c = PointOnCurveConstraint(pt, bez_id)
    assert c.gradient(reg, params) == {}


def test_point_on_curve_pulls_point_onto_curve(setup_env):
    reg, _params = setup_env
    bez_id, p_start, p_end = add_hump_bezier(reg)
    reg.get_point(p_start).fixed = True
    reg.get_point(p_end).fixed = True
    pt = reg.add_point(5.0, 6.0)

    constraints = [PointOnCurveConstraint(pt, bez_id)]
    solver = Solver(reg, ParameterContext(), constraints)
    solver.solve()

    c = PointOnCurveConstraint(pt, bez_id)
    assert c.error(reg, ParameterContext()) < 1e-4


def test_point_on_curve_pulls_curve_to_anchor(setup_env):
    reg, _params = setup_env
    bez_id, _p_start, _p_end = add_hump_bezier(reg)
    anchor = reg.add_point(5.0, 6.5, fixed=True)

    constraints = [PointOnCurveConstraint(anchor, bez_id)]
    solver = Solver(reg, ParameterContext(), constraints)
    solver.solve()

    c = PointOnCurveConstraint(anchor, bez_id)
    assert c.error(reg, ParameterContext()) < 1e-4


def test_point_on_curve_can_apply_to():
    sketch = Sketch()
    p_start = sketch.add_point(0, 0)
    p_end = sketch.add_point(10, 0)
    bez_id = sketch.registry.add_bezier(
        start_idx=p_start, end_idx=p_end, cp1=(0.0, 5.0)
    )
    free_pt = sketch.add_point(5, 8)
    line_id = sketch.registry.add_line(p_start, p_end)

    sel = SketchSelection()
    sel.point_ids = [free_pt]
    sel.entity_ids = [bez_id]
    assert PointOnCurveConstraint.can_apply_to(sel, sketch) is True

    sel.point_ids = [p_start]
    sel.entity_ids = [bez_id]
    assert PointOnCurveConstraint.can_apply_to(sel, sketch) is False

    sel.point_ids = [free_pt]
    sel.entity_ids = [line_id]
    assert PointOnCurveConstraint.can_apply_to(sel, sketch) is False

    sel.point_ids = [p_start, p_end]
    sel.entity_ids = []
    assert PointOnCurveConstraint.can_apply_to(sel, sketch) is False


def test_point_on_curve_serialization_round_trip(setup_env):
    reg, params = setup_env
    bez_id, _start, _end = add_hump_bezier(reg)
    pt = reg.add_point(5.0, 8.0)

    original = PointOnCurveConstraint(pt, bez_id)
    restored = PointOnCurveConstraint.from_dict(original.to_dict())

    assert original.error(reg, params) == pytest.approx(
        restored.error(reg, params)
    )
    assert restored.user_visible is True


def test_point_on_curve_references(setup_env):
    reg, _params = setup_env
    bez_id, _start, _end = add_hump_bezier(reg)
    pt = reg.add_point(5.0, 8.0)

    c = PointOnCurveConstraint(pt, bez_id)
    assert c.get_referenced_point_ids() == {pt}
    assert c.get_referenced_entity_ids() == {bez_id}
    assert c.get_draggable_point() == pt
    assert c.get_type_key() == "point_on_curve"
    assert c.targets_segment(pt, 0, bez_id) is False


def test_point_on_curve_is_hit(setup_env):
    reg, _params = setup_env
    bez_id, _start, _end = add_hump_bezier(reg)
    pt = reg.add_point(10, 20)

    c = PointOnCurveConstraint(pt, bez_id)

    def to_screen(pos):
        return pos

    mock_element = MagicMock()
    threshold = 15.0

    assert c.is_hit(10, 20, reg, to_screen, mock_element, threshold) is True
    assert c.is_hit(30, 20, reg, to_screen, mock_element, threshold) is False


def test_point_on_curve_draw(setup_env):
    reg, _params = setup_env
    bez_id, _start, _end = add_hump_bezier(reg)
    pt = reg.add_point(0, 0)

    c = PointOnCurveConstraint(pt, bez_id)

    ctx = MagicMock()

    def to_screen(pos):
        return pos

    c.draw(ctx, reg, to_screen)
    c.draw(ctx, reg, to_screen, is_selected=True)
    c.draw(ctx, reg, to_screen, is_hovered=True)
    c.draw(ctx, reg, to_screen, point_radius=10.0)


def test_point_on_curve_sketch_end_to_end():
    sketch = Sketch()
    p_start = sketch.add_point(0, 0, fixed=True)
    p_end = sketch.add_point(10, 0, fixed=True)
    bez_id = sketch.registry.add_bezier(
        start_idx=p_start,
        end_idx=p_end,
        cp1=(0.0, 10.0),
        cp2=(0.0, 10.0),
    )
    pt = sketch.add_point(5, 6)
    sketch.constraints.append(PointOnCurveConstraint(pt, bez_id))

    sketch.solve()

    c = PointOnCurveConstraint(pt, bez_id)
    assert c.error(sketch.registry, ParameterContext()) < 1e-4


def test_point_on_curve_ignores_missing_shape(setup_env):
    reg, params = setup_env
    pt = reg.add_point(1.0, 2.0)

    c = PointOnCurveConstraint(pt, shape_id=999)
    assert c.error(reg, params) == 0.0
    assert c.gradient(reg, params) == {}
