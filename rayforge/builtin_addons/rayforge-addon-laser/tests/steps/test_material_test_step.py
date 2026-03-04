from unittest.mock import MagicMock

import pytest

from laser_essentials.steps import MaterialTestStep


@pytest.fixture
def mock_context():
    context = MagicMock()
    machine = MagicMock()
    machine.max_cut_speed = 5000
    machine.max_travel_speed = 10000
    machine.acceleration = 3000
    default_head = MagicMock()
    default_head.uid = "test-laser-uid"
    default_head.spot_size_mm = (0.1, 0.1)
    machine.get_default_head.return_value = default_head
    context.machine = machine
    return context


class TestMaterialTestStep:
    def test_instantiation(self):
        step = MaterialTestStep(name="Test")
        assert step.typelabel == "Material Test Grid"
        assert step.capabilities == set()

    def test_create(self, mock_context):
        step = MaterialTestStep.create(mock_context)
        assert isinstance(step, MaterialTestStep)
        assert step.opsproducer_dict is not None
        assert step.opsproducer_dict["type"] == "MaterialTestGridProducer"

    def test_serialization_includes_step_type(self):
        step = MaterialTestStep(name="Test")
        data = step.to_dict()
        assert data["step_type"] == "MaterialTestStep"
