from unittest.mock import MagicMock

import pytest

from rayforge.core.capability import CUT, SCORE
from laser_essentials.steps import FrameStep


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


class TestFrameStep:
    def test_instantiation(self):
        step = FrameStep(name="Test")
        assert step.typelabel == "Frame Outline"
        assert step.capabilities == {CUT, SCORE}

    def test_create(self, mock_context):
        step = FrameStep.create(mock_context)
        assert isinstance(step, FrameStep)
        assert step.opsproducer_dict is not None
        assert step.opsproducer_dict["type"] == "FrameProducer"

    def test_serialization_includes_step_type(self):
        step = FrameStep(name="Test")
        data = step.to_dict()
        assert data["step_type"] == "FrameStep"
