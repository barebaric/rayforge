import math

import pytest
from sketcher.core.constraints import RotationalConstraint
from sketcher.core.params import ParameterContext
from sketcher.core.registry import EntityRegistry


@pytest.fixture
def setup_env():
    reg = EntityRegistry()
    params = ParameterContext()
    return reg, params


def test_error_zero_when_satisfied(setup_env):
    reg, params = setup_env
    center = reg.add_point(0, 0)
    source = reg.add_point(10, 0)
    target = reg.add_point(0, 10)

    c = RotationalConstraint(center, source, target, math.pi / 2)
    assert c.error(reg, params) == pytest.approx([0.0, 0.0])


def test_error_nonzero_when_violated(setup_env):
    reg, params = setup_env
    center = reg.add_point(0, 0)
    source = reg.add_point(10, 0)
    target = reg.add_point(5, 5)

    c = RotationalConstraint(center, source, target, math.pi / 2)
    err_x, err_y = c.error(reg, params)
    assert err_x == pytest.approx(5.0)
    assert err_y == pytest.approx(-5.0)


def test_rotation_about_offset_center(setup_env):
    reg, params = setup_env
    center = reg.add_point(100, -50)
    source = reg.add_point(110, -50)
    angle = math.radians(30)
    target = reg.add_point(
        100 + 10 * math.cos(angle), -50 + 10 * math.sin(angle)
    )

    c = RotationalConstraint(center, source, target, angle)
    assert c.error(reg, params) == pytest.approx([0.0, 0.0])


def test_gradient_matches_finite_difference(setup_env):
    reg, _ = setup_env
    params = ParameterContext()
    center = reg.add_point(3, -2)
    source = reg.add_point(12.0, 4.0)
    target = reg.add_point(-6.0, 9.0)

    c = RotationalConstraint(center, source, target, math.radians(37))

    eps = 1e-7
    for row in range(2):
        for pid in (source, target, center):
            pt = reg.get_point(pid)
            for axis in (0, 1):
                attr = "x" if axis == 0 else "y"
                original = getattr(pt, attr)
                setattr(pt, attr, original + eps)
                e_plus = c.error(reg, params)[row]
                setattr(pt, attr, original - eps)
                e_minus = c.error(reg, params)[row]
                setattr(pt, attr, original)
                fd = (e_plus - e_minus) / (2 * eps)
                analytic = c.gradient(reg, params)[pid][row][axis]
                assert fd == pytest.approx(analytic, abs=1e-4)


def test_serialization_round_trip(setup_env):
    _reg, _params = setup_env
    c = RotationalConstraint(
        center=1, p1=2, p2=3, value=math.radians(45), user_visible=False
    )
    data = c.to_dict()
    restored = RotationalConstraint.from_dict(data)
    assert restored.center == 1
    assert restored.p1 == 2
    assert restored.p2 == 3
    assert restored.value == pytest.approx(c.value)
    assert restored.user_visible is False


def test_expression_value_update(setup_env):
    _reg, _params = setup_env
    c = RotationalConstraint(0, 1, 2, value=0.0, expression="pi / 3")
    c.update_from_context({"pi": math.pi})
    assert c.value == pytest.approx(math.pi / 3)


def test_deleted_points_are_depended_on():
    """The generic constraint cleanup must catch this constraint via
    depends_on_points (attribute names p1/p2/center)."""
    c = RotationalConstraint(center=7, p1=8, p2=9, value=1.0)
    assert c.depends_on_points({8})
    assert c.depends_on_points({9})
    assert c.depends_on_points({7})
    assert not c.depends_on_points({10})
