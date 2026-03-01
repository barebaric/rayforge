from unittest.mock import MagicMock

import pytest

from rayforge.core.capability import CUT, SCORE
from rayforge.core.step import Step
from rayforge.core.step_registry import step_registry
from laser_essentials.steps import ContourStep


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


class TestContourStep:
    def test_instantiation(self):
        step = ContourStep(name="Test")
        assert step.typelabel == "Contour"
        assert step.name == "Test"
        assert step.capabilities == {CUT, SCORE}

    def test_create(self, mock_context):
        step = ContourStep.create(mock_context, name="Created")
        assert isinstance(step, ContourStep)
        assert step.name == "Created"
        assert step.opsproducer_dict is not None
        assert step.opsproducer_dict["type"] == "ContourProducer"
        assert len(step.per_workpiece_transformers_dicts) == 3
        assert len(step.per_step_transformers_dicts) == 1
        assert step.selected_laser_uid == "test-laser-uid"

    def test_create_without_optimize(self, mock_context):
        step = ContourStep.create(mock_context, optimize=False)
        assert len(step.per_workpiece_transformers_dicts) == 2

    def test_serialization_includes_step_type(self):
        step = ContourStep(name="Test")
        data = step.to_dict()
        assert data["step_type"] == "ContourStep"

    def test_deserialization_returns_contour_step(self):
        step_registry.register(ContourStep)
        step = ContourStep(name="Original")
        data = step.to_dict()

        restored = Step.from_dict(data)
        assert isinstance(restored, ContourStep)
        assert restored.name == "Original"

    def test_registry_create_contour_step(self, mock_context):
        StepClass = step_registry.get("ContourStep")
        assert StepClass is not None
        step = StepClass.create(mock_context, name="FromRegistry")
        assert isinstance(step, ContourStep)
        assert step.name == "FromRegistry"
