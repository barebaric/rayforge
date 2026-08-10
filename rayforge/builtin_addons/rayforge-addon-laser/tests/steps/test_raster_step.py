from unittest.mock import MagicMock, patch

import pytest
from laser_essentials.steps import EngraveStep
from raygeo.cnc.execution.specs import ComputePayload
from raygeo.ops.assembly import Assembler
from raygeo.ops.assembly.raster import RasterSpec
from raygeo.ops.part import Part

from rayforge.core.step_registry import step_registry
from rayforge.core.workpiece import WorkPiece


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
    machine.get_default_laser_head.return_value = default_head
    context.machine = machine
    return context


class TestEngraveStep:
    def test_instantiation(self):
        step = EngraveStep(name="Test")
        assert step.typelabel == "Engrave"

    def test_create(self, mock_context):
        step = EngraveStep.create(mock_context, name="Created")
        assert isinstance(step, EngraveStep)
        assert len(step.per_workpiece_transformers_dicts) == 3
        transformer_names = {
            t.get("name") for t in step.per_workpiece_transformers_dicts
        }
        assert "BidirScanOffsetTransformer" in transformer_names
        assert step.selected_head_uid == "test-laser-uid"

    def test_serialization_includes_step_type(self):
        step = EngraveStep(name="Test")
        data = step.to_dict()
        assert data["step_type"] == "EngraveStep"

    def test_registry_create_engrave_step(self, mock_context):
        StepClass = step_registry.get("EngraveStep")
        assert StepClass is not None
        step = StepClass.create(mock_context, name="FromRegistry")
        assert type(step).__name__ == "EngraveStep"

    def test_get_assembler_kwargs(self, machine):
        step = EngraveStep(name="Test")
        workpiece = MagicMock(spec=["size"])
        workpiece.size = (100, 100)
        kwargs = step.get_assembler_kwargs(machine, workpiece)
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

    def test_legacy_power_keys_migrate(self):
        """Old files keyed the raster power range as min_power/max_power.

        Those must load into min_power_level/max_power_level and must not
        pollute extra. The hardware max_power slot is restored to its
        default rather than inheriting the old raster ceiling.
        """
        step = EngraveStep(name="Test")
        data = step.to_dict()
        data["min_power"] = data.pop("min_power_level")
        data["max_power"] = data.pop("max_power_level")
        data["min_power"] = 0.2
        data["max_power"] = 1.0

        restored = EngraveStep.from_dict(data)

        assert restored.min_power_level == 0.2
        assert restored.max_power_level == 1.0
        assert restored.max_power == 1000
        assert "min_power" not in restored.extra
        assert "max_power" not in restored.extra

    def test_from_dict_migrates_legacy_opsproducer_params(self):
        """True legacy files store raster params in
        ``opsproducer_dict.params``; loading must restore them."""
        step = EngraveStep(name="Test")
        data = step.to_dict()
        for key in (
            "scan_angle",
            "depth_mode",
            "invert",
            "auto_levels",
            "black_point",
            "white_point",
            "threshold",
            "line_interval_mm",
            "sample_interval_mm",
            "min_power_level",
            "max_power_level",
            "num_power_levels",
            "scan_mode",
            "cross_hatch",
            "num_depth_levels",
            "z_step_down",
            "angle_increment",
            "dither_algorithm",
        ):
            data.pop(key, None)
        data["opsproducer_dict"] = {
            "type": "Rasterizer",
            "params": {
                "direction_degrees": 45.0,
                "scan_mode": "FullSweep",
                "threshold": 100,
                "dither_algorithm": "bayer4",
                "cross_hatch": True,
                "min_power": 0.2,
                "max_power": 0.9,
                "num_depth_levels": 3,
                "num_power_levels": 10,
                "z_step_down": 0.5,
                "invert": True,
                "auto_levels": False,
                "black_point": 20,
                "white_point": 200,
                "angle_increment": 30.0,
                "line_interval_mm": 0.4,
            },
        }

        restored = EngraveStep.from_dict(data)

        assert restored.depth_mode == "CONSTANT_POWER"
        assert restored.scan_angle == 45.0
        assert restored.scan_mode == "FULL_SWEEP"
        assert restored.threshold == 100
        assert restored.dither_algorithm is not None
        assert restored.dither_algorithm.name == "BAYER4"
        assert restored.cross_hatch is True
        assert restored.min_power_level == 0.2
        assert restored.max_power_level == 0.9
        assert restored.num_depth_levels == 3
        assert restored.num_power_levels == 10
        assert restored.z_step_down == 0.5
        assert restored.invert is True
        assert restored.auto_levels is False
        assert restored.black_point == 20
        assert restored.white_point == 200
        assert restored.angle_increment == 30.0
        assert restored.line_interval_mm == 0.4
        assert restored.max_power == 1000

    def test_from_dict_dither_rasterizer_uses_dither_mode(self):
        """The legacy ``DitherRasterizer`` type implies DITHER mode."""
        step = EngraveStep(name="Test")
        data = step.to_dict()
        for key in ("depth_mode", "scan_angle", "threshold"):
            data.pop(key, None)
        data["opsproducer_dict"] = {
            "type": "DitherRasterizer",
            "params": {"threshold": 150},
        }

        restored = EngraveStep.from_dict(data)

        assert restored.depth_mode == "DITHER"
        assert restored.threshold == 150


class TestEngraveComputePayload:
    """Verifies EngraveStep's build_compute_payload (B3)."""

    def test_build_compute_payload_returns_raster_spec(self, machine):
        step = EngraveStep(name="engrave")
        step.min_power_level = 0.1
        step.max_power_level = 0.9
        wp = WorkPiece(name="wp")
        wp.set_size(10.0, 10.0)

        with patch.object(WorkPiece, "render_to_pixels", return_value=None):
            part, payload = step.build_compute_payload(machine, wp)

        assert isinstance(part, Part)
        assert isinstance(payload, ComputePayload)
        assert isinstance(payload.assembler, Assembler)
        spec = payload.assembler.spec
        assert isinstance(spec, RasterSpec)
        assert spec.min_power == 0.1
        assert spec.max_power == 0.9
        assert spec.mode == "power_modulated"

    def test_assembler_token_params_mirrors_kwargs(self, machine):
        step = EngraveStep(name="engrave")
        wp = WorkPiece(name="wp")
        wp.set_size(10.0, 10.0)
        token = step.assembler_token_params(machine, wp)
        kwargs = step.get_assembler_kwargs(machine, wp)
        assert token == kwargs
