import pytest

from rayforge.core.sketcher.params import ParameterContext
from rayforge.core.sketcher.entities import EntityRegistry
from rayforge.core.sketcher.constraints import EqualDistanceConstraint


@pytest.fixture
def setup_env():
    reg = EntityRegistry()
    params = ParameterContext()
    return reg, params


def test_equal_distance_constraint(setup_env):
    reg, params = setup_env
    # Segment 1: Length 10
    p1 = reg.add_point(0, 0)
    p2 = reg.add_point(10, 0)

    # Segment 2: Length 4
    p3 = reg.add_point(0, 10)
    p4 = reg.add_point(0, 14)

    c = EqualDistanceConstraint(p1, p2, p3, p4)

    # Error should be 10 - 4 = 6
    assert c.error(reg, params) == pytest.approx(6.0)


def test_equal_distance_constraint_serialization_round_trip(setup_env):
    reg, params = setup_env
    # Segment 1: Length 10
    p1 = reg.add_point(0, 0)
    p2 = reg.add_point(10, 0)

    # Segment 2: Length 4
    p3 = reg.add_point(0, 10)
    p4 = reg.add_point(0, 14)

    # Create original constraint
    original = EqualDistanceConstraint(p1, p2, p3, p4)

    # Serialize to dict
    serialized = original.to_dict()

    # Deserialize from dict
    restored = EqualDistanceConstraint.from_dict(serialized)

    # Check that the restored constraint has the same error
    assert original.error(reg, params) == restored.error(reg, params)
