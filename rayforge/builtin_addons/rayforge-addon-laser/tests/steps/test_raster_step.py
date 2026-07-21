from unittest.mock import MagicMock, patch

import numpy as np
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
        assert len(step.per_workpiece_transformers_dicts) == 3
        transformer_names = {
            t.get("name") for t in step.per_workpiece_transformers_dicts
        }
        assert "BidirScanOffsetTransformer" in transformer_names
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
            "dot_width_correction_mm",
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
        step.dot_width_correction_mm = 0.05  # type: ignore[assignment]
        data = step.to_dict()
        restored = EngraveStep.from_dict(data)
        assert data == restored.to_dict()
        assert restored.dot_width_correction_mm == 0.05


class TestAssembleOnSurfaceAutoDefaults:
    """
    assemble_on_surface must resolve dot_width_correction_mm the same
    way line_interval_mm/sample_interval_mm already do: None falls back
    to a laser-spot-size-derived value actually used at generation
    time, an explicit override is passed through unchanged.
    """

    def _run(self, step, machine_defaults, laser_spot_size_mm):
        laser = MagicMock()
        laser.uid = "laser-1"
        laser.spot_size_mm = laser_spot_size_mm

        surface = MagicMock()
        surface.get_width.return_value = 10
        surface.get_height.return_value = 10

        workpiece = MagicMock()
        workpiece.size = (10.0, 10.0)
        workpiece.bbox = (0.0, 0.0, 10.0, 10.0)
        workpiece.uid = "wp-1"

        fake_image = np.zeros((10, 10), dtype=np.uint8)
        fake_alpha = np.ones((10, 10), dtype=np.float32)

        with (
            patch(
                "laser_essentials.steps.raster_step.preprocess_raster_image",
                return_value=(fake_image, fake_alpha),
            ),
            patch("laser_essentials.steps.raster_step.make_artifact"),
            patch(
                "laser_essentials.steps.raster_step.assembler_registry"
            ) as mock_registry,
        ):
            mock_result = MagicMock()
            mock_result.ops = MagicMock(len=MagicMock(return_value=0))
            mock_registry.assemble.return_value = mock_result

            step.assemble_on_surface(
                workpiece,
                laser,
                generation_id=1,
                surface=surface,
                pixels_per_mm=(1.0, 1.0),
                machine_defaults=machine_defaults,
            )

        return mock_registry.assemble.call_args.kwargs

    def test_auto_default_uses_half_spot_size(self, machine_defaults):
        step = EngraveStep(name="Test")
        kwargs = self._run(step, machine_defaults, (0.2, 0.15))
        assert kwargs["dot_width_correction_mm"] == pytest.approx(0.1)

    def test_explicit_override_is_respected(self, machine_defaults):
        step = EngraveStep(name="Test")
        step.dot_width_correction_mm = 0.5  # type: ignore[assignment]
        kwargs = self._run(step, machine_defaults, (0.2, 0.15))
        assert kwargs["dot_width_correction_mm"] == pytest.approx(0.5)
