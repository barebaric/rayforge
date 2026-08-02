from typing import TYPE_CHECKING, Protocol, cast
from unittest.mock import MagicMock

import pytest
from laser_essentials.steps import MaterialTestStep

from rayforge.core.capability import MATERIAL_TEST
from rayforge.core.workpiece import WorkPiece

if TYPE_CHECKING:

    class OverscanTransformerType(Protocol):
        @staticmethod
        def calculate_auto_distance(
            step_speed: int, max_acceleration: int
        ) -> float: ...


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


class TestMaterialTestStep:
    def test_instantiation(self):
        step = MaterialTestStep(name="Test")
        assert step.typelabel == "Material Test Grid"
        assert step.capabilities == (MATERIAL_TEST,)

    def test_create(self, mock_context):
        step = MaterialTestStep.create(mock_context)
        assert isinstance(step, MaterialTestStep)

    def test_serialization_includes_step_type(self):
        step = MaterialTestStep(name="Test")
        data = step.to_dict()
        assert data["step_type"] == "MaterialTestStep"

    def test_get_assembler_kwargs(self, machine):
        step = MaterialTestStep(name="Test")
        workpiece = MagicMock(spec=["size"])
        workpiece.size = (100, 100)
        kwargs = step.get_assembler_kwargs(machine, workpiece)
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
            "min_offset",
            "max_offset",
            "mode",
            "grid_mode",
            "fixed_speed",
            "fixed_power",
            "shape_size",
            "spacing",
            "include_labels",
            "label_power_percent",
            "label_speed",
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

    def test_optimize_present_but_disabled_by_default(self, mock_context):
        """Optimize must be off by default: its nearest-neighbor travel
        reordering has no concept of cell boundaries and can interleave
        lines from different cells instead of engraving each one fully
        before moving to the next. Left toggleable (not removed) so it's
        easy to compare with/without."""
        step = MaterialTestStep.create(mock_context)
        per_wp = {
            t.get("name"): t for t in step.per_workpiece_transformers_dicts
        }
        per_step = {t.get("name"): t for t in step.per_step_transformers_dicts}
        assert "Optimize" in per_wp
        assert per_wp["Optimize"]["enabled"] is False
        assert "Optimize" in per_step
        assert per_step["Optimize"]["enabled"] is False

    def test_overscan_distance_is_doubled(self, mock_context):
        """Individual test blocks get double the usual auto-overscan
        distance, so backlash settling happens outside the visible
        engrave area."""
        from rayforge.pipeline.transformer.registry import (
            transformer_registry,
        )

        OverscanTransformer = cast(
            "OverscanTransformerType",
            transformer_registry.get("OverscanTransformer"),
        )
        assert OverscanTransformer is not None

        step = MaterialTestStep.create(mock_context)
        overscan_dict = next(
            t
            for t in step.per_workpiece_transformers_dicts
            if t.get("name") == "OverscanTransformer"
        )
        expected_base = OverscanTransformer.calculate_auto_distance(
            step.cut_speed, mock_context.machine.acceleration
        )
        assert overscan_dict["distance_mm"] == pytest.approx(expected_base * 2)

    def test_includes_bidir_scan_offset_transformer(self, mock_context):
        step = MaterialTestStep.create(mock_context)
        per_wp_names = {
            t.get("name") for t in step.per_workpiece_transformers_dicts
        }
        assert "BidirScanOffsetTransformer" in per_wp_names


class TestMaterialTestComputePayload:
    def test_build_compute_payload_returns_material_test_spec(self, machine):
        from raygeo.cnc.execution.specs import ComputePayload
        from raygeo.ops.assembly import Assembler
        from raygeo.ops.assembly.material_test_grid import (
            MaterialTestGridSpec,
        )
        from raygeo.ops.part import Part

        step = MaterialTestStep(name="mtg")
        step.test_type = "Cut"
        wp = WorkPiece(name="wp")
        wp.set_size(100.0, 100.0)

        part, payload = step.build_compute_payload(machine, wp)
        assert isinstance(part, Part)
        assert isinstance(payload, ComputePayload)
        assert isinstance(payload.assembler, Assembler)
        spec = payload.assembler.spec
        assert isinstance(spec, MaterialTestGridSpec)
        assert spec.mode == "cut"
        assert spec.size_mm == (100.0, 100.0)

    def test_assembler_token_params_mirrors_kwargs(self, machine):
        step = MaterialTestStep(name="mtg")
        wp = WorkPiece(name="wp")
        wp.set_size(100.0, 100.0)
        token = step.assembler_token_params(machine, wp)
        kwargs = step.get_assembler_kwargs(machine, wp)
        assert token == kwargs
