from unittest.mock import MagicMock

import pytest

from rayforge.core.capability import ENGRAVE
from rayforge.pipeline.steps import EngraveStep, create_engrave_step


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


class TestEngraveStep:
    def test_instantiation(self):
        step = EngraveStep(name="Test")
        assert step.typelabel == "Engrave"
        assert step.capabilities == {ENGRAVE}

    def test_create(self, mock_context):
        step = EngraveStep.create(mock_context, name="Created")
        assert isinstance(step, EngraveStep)
        assert step.opsproducer_dict is not None
        assert step.opsproducer_dict["type"] == "Rasterizer"
        assert len(step.per_workpiece_transformers_dicts) == 2
        assert step.selected_laser_uid == "test-laser-uid"

    def test_serialization_includes_step_type(self):
        step = EngraveStep(name="Test")
        data = step.to_dict()
        assert data["step_type"] == "EngraveStep"

    def test_backward_compat_create_engrave_step(self, mock_context):
        step = create_engrave_step(mock_context, name="Backward")
        assert isinstance(step, EngraveStep)
