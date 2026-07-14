from unittest.mock import MagicMock

import pytest
from laser_essentials.steps import ShrinkWrapStep

from rayforge.core.capability import CUT, SCORE, WITH_KERF
from rayforge.pipeline.stage.assembler_helpers import MachineDefaults


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


@pytest.fixture
def machine_defaults():
    return MachineDefaults(
        kerf_mm=0.1,
        arc_tolerance=0.03,
        allow_arcs=True,
        supports_curves=False,
        line_interval_mm=0.1,
        step_power=1.0,
        tool_radius=0.05,
        step_over=0.1,
        cut_speed=500,
    )


class TestShrinkWrapStep:
    def test_instantiation(self):
        step = ShrinkWrapStep(name="Test")
        assert step.typelabel == "Shrink Wrap"
        assert step.capabilities == (CUT, SCORE, WITH_KERF)

    def test_create(self, mock_context):
        step = ShrinkWrapStep.create(mock_context)
        assert isinstance(step, ShrinkWrapStep)
        assert step.opsproducer_dict is not None
        assert step.opsproducer_dict["type"] == "ShrinkWrapProducer"

    def test_serialization_includes_step_type(self):
        step = ShrinkWrapStep(name="Test")
        data = step.to_dict()
        assert data["step_type"] == "ShrinkWrapStep"

    def test_get_assembler_kwargs(self, machine_defaults):
        step = ShrinkWrapStep(name="Test")
        workpiece = MagicMock(spec=["size"])
        workpiece.size = (100, 100)
        kwargs = step.get_assembler_kwargs(machine_defaults, workpiece)
        assert isinstance(kwargs, dict)
        expected_keys = {
            "cut_side",
            "gravity",
            "path_offset_mm",
            "kerf_mm",
            "arc_tolerance",
            "allow_arcs",
            "supports_curves",
        }
        assert set(kwargs.keys()) == expected_keys

    def test_roundtrip_serialization(self):
        step = ShrinkWrapStep(name="Test")
        step.cut_side = "OUTSIDE"
        step.path_offset_mm = 0.5
        step.gravity = 0.5
        data = step.to_dict()
        restored = ShrinkWrapStep.from_dict(data)
        assert data == restored.to_dict()
