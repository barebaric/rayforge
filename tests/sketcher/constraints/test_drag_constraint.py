import pytest

from rayforge.core.sketcher.params import ParameterContext
from rayforge.core.sketcher.entities import EntityRegistry
from rayforge.core.sketcher.constraints import DragConstraint


@pytest.fixture
def setup_env():
    reg = EntityRegistry()
    params = ParameterContext()
    return reg, params


def test_drag_constraint(setup_env):
    reg, params = setup_env
    p1 = reg.add_point(0, 0)

    # Target is (100, 0). Current is (0, 0).
    # err_x = (0 - 100) * 0.1 = -10.0
    # err_y = (0 - 0) * 0.1 = 0.0
    c = DragConstraint(p1, 100.0, 0.0, weight=0.1)
    err_x, err_y = c.error(reg, params)
    assert err_x == pytest.approx(-10.0)
    assert err_y == pytest.approx(0.0)


def test_drag_constraint_serialization_round_trip(setup_env):
    reg, params = setup_env
    p1 = reg.add_point(0, 0)

    # Create original constraint
    original = DragConstraint(p1, 100.0, 0.0, weight=0.1)

    # Serialize to dict
    serialized = original.to_dict()

    # DragConstraint returns empty dict from to_dict() as it's not meant to be
    # serialized
    assert serialized == {}
