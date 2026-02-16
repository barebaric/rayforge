import pytest
import math
from types import SimpleNamespace
from unittest.mock import MagicMock
from rayforge.core.sketcher.constraints import AngleConstraint
from rayforge.core.sketcher.constraints.angle import ARC_RADIUS
from rayforge.core.sketcher.params import ParameterContext
from rayforge.core.sketcher.registry import EntityRegistry


@pytest.fixture
def setup_env():
    reg = EntityRegistry()
    params = ParameterContext()
    return reg, params


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
    reg, params = setup_env
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
    reg, params = setup_env

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
    reg, params = setup_env
    p1 = reg.add_point(0, 0)
    p2 = reg.add_point(10, 0)
    l1 = reg.add_line(p1, p2)

    p3 = reg.add_point(0, 0)
    p4 = reg.add_point(0, 10)
    l2 = reg.add_line(p3, p4)

    c = AngleConstraint(l1, l2, 90.0)

    ctx = MagicMock()

    def to_screen(pos):
        return pos

    c.draw(ctx, reg, to_screen)
    c.draw(ctx, reg, to_screen, is_selected=True)
    c.draw(ctx, reg, to_screen, is_hovered=True)
    c.draw(ctx, reg, to_screen, point_radius=10.0)


def test_angle_constraint_get_type_name():
    assert AngleConstraint.get_type_name() == "Angle"
