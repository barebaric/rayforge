from unittest.mock import MagicMock

import pytest
from laser_essentials.steps import EngraveStep

from rayforge.core.capability import ENGRAVE
from rayforge.core.step_registry import step_registry
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


class TestEngraveStep:
    def test_instantiation(self):
        step = EngraveStep(name="Test")
        assert step.typelabel == "Engrave"
        assert step.capabilities == (ENGRAVE,)

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

    def test_registry_create_engrave_step(self, mock_context):
        StepClass = step_registry.get("EngraveStep")
        assert StepClass is not None
        step = StepClass.create(mock_context, name="FromRegistry")
        assert type(step).__name__ == "EngraveStep"

    def test_get_assembler_kwargs(self, machine_defaults):
        step = EngraveStep(name="Test")
        workpiece = MagicMock(spec=["size"])
        workpiece.size = (100, 100)
        kwargs = step.get_assembler_kwargs(machine_defaults, workpiece)
        assert isinstance(kwargs, dict)
        expected_keys = {
            "mode",
            "line_interval_mm",
            "sample_interval_mm",
            "min_power",
            "max_power",
            "step_power",
            "num_power_levels",
            "angle",
            "offset_x_mm",
            "offset_y_mm",
            "scan_mode",
            "cross_hatch",
            "num_depth_levels",
            "z_step_down",
            "angle_increment",
        }
        assert set(kwargs.keys()) == expected_keys

    def test_roundtrip_serialization(self):
        step = EngraveStep(name="Test")
        step.scan_angle = 45.0
        step.depth_mode = "MULTI_PASS"
        step.line_interval_mm = 0.2  # type: ignore[assignment]
        data = step.to_dict()
        restored = EngraveStep.from_dict(data)
        assert data == restored.to_dict()
