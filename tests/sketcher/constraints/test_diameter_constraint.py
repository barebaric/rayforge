import pytest

from rayforge.core.sketcher.params import ParameterContext
from rayforge.core.sketcher.entities import EntityRegistry
from rayforge.core.sketcher.constraints import DiameterConstraint


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
