from unittest.mock import MagicMock

import pytest
from laser_essentials.steps import MaterialTestStep

from rayforge.core.capability import MATERIAL_TEST
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


class TestMaterialTestStep:
    def test_instantiation(self):
        step = MaterialTestStep(name="Test")
        assert step.typelabel == "Material Test Grid"
        assert step.capabilities == (MATERIAL_TEST,)

    def test_create(self, mock_context):
        step = MaterialTestStep.create(mock_context)
        assert isinstance(step, MaterialTestStep)
        assert step.opsproducer_dict is not None
        assert step.opsproducer_dict["type"] == "MaterialTestGridProducer"

    def test_serialization_includes_step_type(self):
        step = MaterialTestStep(name="Test")
        data = step.to_dict()
        assert data["step_type"] == "MaterialTestStep"

    def test_get_assembler_kwargs(self, machine_defaults):
        step = MaterialTestStep(name="Test")
        workpiece = MagicMock(spec=["size"])
        workpiece.size = (100, 100)
        kwargs = step.get_assembler_kwargs(machine_defaults, workpiece)
        assert isinstance(kwargs, dict)
        expected_keys = {
            "size_mm",
            "cols",
            "rows",
            "min_speed",
            "max_speed",
            "min_power",
            "max_power",
            "min_passes",
            "max_passes",
            "mode",
            "grid_mode",
            "fixed_speed",
            "fixed_power",
            "shape_size",
            "spacing",
            "include_labels",
            "line_interval_mm",
        }
        assert set(kwargs.keys()) == expected_keys

    def test_roundtrip_serialization(self):
        step = MaterialTestStep(name="Test")
        step.test_type = "Engrave"
        step.grid_mode = "Power vs Passes"
        step.shape_size = 5.0
        data = step.to_dict()
        restored = MaterialTestStep.from_dict(data)
        assert data == restored.to_dict()
