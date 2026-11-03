import math
from types import SimpleNamespace

import cairo
import pytest
from sketcher.core import Sketch
from sketcher.core.constraints import AngleConstraint, DistanceConstraint
from sketcher.core.constraints.angle import ARC_RADIUS
from sketcher.core.params import ParameterContext
from sketcher.core.registry import EntityRegistry
from sketcher.core.selection import SketchSelection
from sketcher.core.solver import Solver


@pytest.fixture
def setup_env():
    reg = EntityRegistry()
    params = ParameterContext()
    return reg, params


def _chained_lines(reg, n_segments, seg_len=10.0, turn_deg=30.0):
    px, py = 0.0, 0.0
    p_first = reg.add_point(px, py, fixed=True)
    p_prev = p_first
    joints = []
    lines = []
    angle = 0.0
    for i in range(n_segments):
        angle += turn_deg if i % 2 == 0 else -turn_deg
        a = math.radians(angle)
        px += seg_len * math.cos(a)
        py += seg_len * math.sin(a)
        p_next = reg.add_point(px, py)
        line = reg.add_line(p_prev, p_next)
        joints.append(p_next)
        lines.append(line)
        p_prev = p_next
    return p_first, joints, lines


def _chain_constraints(reg, p_first, joints, lines, turn_deg=30.0):
    constraints = [DistanceConstraint(p_first, joints[0], 10.0)]
    for i, line in enumerate(lines):
        if i == 0:
            continue
        prev = lines[i - 1]
        constraints.append(DistanceConstraint(joints[i - 1], joints[i], 10.0))
        theta = turn_deg if i % 2 == 0 else -turn_deg
        target = math.degrees(math.pi - math.radians(theta))
        constraints.append(AngleConstraint(prev, line, target))
    return constraints


def test_gradient_matches_error_on_chained_lines(setup_env):
    reg, params = setup_env
    p_first, joints, lines = _chained_lines(reg, 4)
    constraints = _chain_constraints(reg, p_first, joints, lines)

    eps = 1e-6
    for constraint in constraints:
        if not isinstance(constraint, AngleConstraint):
            continue
        grads = constraint.gradient(reg, params)
        for pid, points in grads.items():
            point = reg.get_point(pid)
            for axis, attr in enumerate(("x", "y")):
                original = getattr(point, attr)
                setattr(point, attr, original + eps)
                plus = constraint.error(reg, params)
                setattr(point, attr, original - eps)
                minus = constraint.error(reg, params)
                setattr(point, attr, original)
                finite = (plus - minus) / (2.0 * eps)
                assert points[0][axis] == pytest.approx(finite, abs=1e-4)


def test_solver_converges_after_parameter_change(setup_env):
    reg, params = setup_env
    p_first, joints, lines = _chained_lines(reg, 13)
    constraints = _chain_constraints(reg, p_first, joints, lines)

    solver = Solver(reg, params, constraints)
    assert solver.solve() is True
    assert solver.get_conflicting_constraints() == set()

    for constraint in constraints:
        if isinstance(constraint, AngleConstraint):
            constraint.value -= 3.0

    assert solver.solve() is True
    assert solver.get_conflicting_constraints() == set()

    for constraint in constraints:
        error = constraint.error(reg, params)
        if isinstance(error, (list, tuple)):
            assert max(abs(e) for e in error) < 1e-4
        else:
            assert abs(error) < 1e-4


def test_angle_constraint_90_degrees(setup_env):
    reg, params = setup_env

    p1 = reg.add_point(0, 0)
    p2 = reg.add_point(10, 0)
    l1 = reg.add_line(p1, p2)

    p3 = reg.add_point(0, 0)
    p4 = reg.add_point(0, 10)
    l2 = reg.add_line(p3, p4)

    c = AngleConstraint(l2, l1, 90.0)
    assert c.error(reg, params) == pytest.approx(0.0, abs=1e-6)


def test_angle_constraint_45_degrees(setup_env):
    reg, params = setup_env

    p1 = reg.add_point(0, 0)
    p2 = reg.add_point(10, 0)
    l1 = reg.add_line(p1, p2)

    p3 = reg.add_point(0, 0)
    p4 = reg.add_point(10, 10)
    l2 = reg.add_line(p3, p4)

    c = AngleConstraint(l2, l1, 45.0)
    assert c.error(reg, params) == pytest.approx(0.0, abs=1e-6)


def test_angle_constraint_non_zero_error(setup_env):
    reg, params = setup_env

    p1 = reg.add_point(0, 0)
    p2 = reg.add_point(10, 0)
    l1 = reg.add_line(p1, p2)

    p3 = reg.add_point(0, 0)
    p4 = reg.add_point(0, 10)
    l2 = reg.add_line(p3, p4)

    c = AngleConstraint(l2, l1, 45.0)
    error = c.error(reg, params)
    assert error == pytest.approx(math.pi / 4 * 10, abs=1e-6)


def test_angle_constraint_user_visible(setup_env):
    reg, _params = setup_env
    p1 = reg.add_point(0, 0)
    p2 = reg.add_point(10, 0)
    l1 = reg.add_line(p1, p2)

    p3 = reg.add_point(0, 0)
    p4 = reg.add_point(0, 10)
    l2 = reg.add_line(p3, p4)

    c = AngleConstraint(l1, l2, 90.0, user_visible=False)
    assert c.user_visible is False

    c2 = AngleConstraint(l1, l2, 90.0, user_visible=True)
    assert c2.user_visible is True


def test_angle_constraint_with_expression(setup_env):
    reg, params = setup_env
    ctx = {"angle": 45.0}

    p1 = reg.add_point(0, 0)
    p2 = reg.add_point(10, 0)
    l1 = reg.add_line(p1, p2)

    p3 = reg.add_point(0, 0)
    p4 = reg.add_point(10, 10)
    l2 = reg.add_line(p3, p4)

    c = AngleConstraint(l2, l1, "angle")
    c.update_from_context(ctx)

    assert c.error(reg, params) == pytest.approx(0.0, abs=1e-6)


def test_angle_constraint_invalid_entities(setup_env):
    reg, params = setup_env
    p1 = reg.add_point(0, 0)

    c = AngleConstraint(p1, p1, 90.0)
    assert c.error(reg, params) == 0.0
    assert c.gradient(reg, params) == {}


def test_angle_constraint_serialization_round_trip(setup_env):
    reg, params = setup_env

    p1 = reg.add_point(0, 0)
    p2 = reg.add_point(10, 0)
    l1 = reg.add_line(p1, p2)

    p3 = reg.add_point(0, 0)
    p4 = reg.add_point(0, 10)
    l2 = reg.add_line(p3, p4)

    original = AngleConstraint(l1, l2, 90.0)

    serialized = original.to_dict()

    restored = AngleConstraint.from_dict(serialized)

    assert original.error(reg, params) == pytest.approx(
        restored.error(reg, params), abs=1e-6
    )
    assert original.user_visible == restored.user_visible


def test_angle_constraint_serialization_with_expression(setup_env):
    reg, params = setup_env
    ctx = {"angle": 90.0}

    p1 = reg.add_point(0, 0)
    p2 = reg.add_point(10, 0)
    l1 = reg.add_line(p1, p2)

    p3 = reg.add_point(0, 0)
    p4 = reg.add_point(0, 10)
    l2 = reg.add_line(p3, p4)

    original = AngleConstraint(l1, l2, "angle")
    original.update_from_context(ctx)

    serialized = original.to_dict()
    restored = AngleConstraint.from_dict(serialized)
    restored.update_from_context(ctx)

    assert original.error(reg, params) == pytest.approx(
        restored.error(reg, params), abs=1e-6
    )


def test_angle_is_hit(setup_env):
    reg, _params = setup_env

    def to_screen(pos):
        return pos

    mock_element = SimpleNamespace()
    threshold = 15.0

    l1p1 = reg.add_point(0, 50)
    l1p2 = reg.add_point(100, 50)
    l1 = reg.add_line(l1p1, l1p2)

    l2p1 = reg.add_point(50, 0)
    l2p2 = reg.add_point(50, 100)
    l2 = reg.add_line(l2p1, l2p2)

    c = AngleConstraint(l2, l1, 90.0)

    hit_x, hit_y = (
        50 + ARC_RADIUS * math.cos(math.pi),
        50 + ARC_RADIUS * math.sin(math.pi),
    )
    assert (
        c.is_hit(hit_x, hit_y, reg, to_screen, mock_element, threshold) is True
    )


def test_angle_draw(setup_env):
    reg, _params = setup_env
    p1 = reg.add_point(0, 0)
    p2 = reg.add_point(10, 0)
    l1 = reg.add_line(p1, p2)

    p3 = reg.add_point(0, 0)
    p4 = reg.add_point(0, 10)
    l2 = reg.add_line(p3, p4)

    c = AngleConstraint(l1, l2, 90.0)

    ctx = cairo.Context(cairo.ImageSurface(cairo.FORMAT_ARGB32, 200, 200))

    def to_screen(pos):
        return pos

    c.draw(ctx, reg, to_screen)
    c.draw(ctx, reg, to_screen, is_selected=True)
    c.draw(ctx, reg, to_screen, is_hovered=True)
    c.draw(ctx, reg, to_screen, point_radius=10.0)


def test_angle_constraint_get_type_name():
    assert AngleConstraint.get_type_name() == "Angle"


def test_angle_can_apply_to_two_lines():
    sketch = Sketch()
    p1 = sketch.add_point(0, 0)
    p2 = sketch.add_point(10, 0)
    p3 = sketch.add_point(0, 10)
    line1_id = sketch.add_line(p1, p2)
    line2_id = sketch.add_line(p1, p3)

    selection = SketchSelection()
    selection.point_ids = []
    selection.entity_ids = [line1_id, line2_id]
    assert AngleConstraint.can_apply_to(selection, sketch) is True


def test_angle_can_apply_to_one_line_invalid():
    sketch = Sketch()
    p1 = sketch.add_point(0, 0)
    p2 = sketch.add_point(10, 0)
    line_id = sketch.add_line(p1, p2)

    selection = SketchSelection()
    selection.point_ids = []
    selection.entity_ids = [line_id]
    assert AngleConstraint.can_apply_to(selection, sketch) is False
