from typing import TYPE_CHECKING, Protocol, cast
from unittest.mock import MagicMock

import pytest
from laser_essentials.steps import MaterialTestStep

from rayforge.core.capability import MATERIAL_TEST

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
    machine.get_default_head.return_value = default_head
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
        assert step.opsproducer_dict is not None
        assert step.opsproducer_dict["type"] == "MaterialTestGridProducer"

    def test_serialization_includes_step_type(self):
        step = MaterialTestStep(name="Test")
        data = step.to_dict()
        assert data["step_type"] == "MaterialTestStep"

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
        per_step = {
            t.get("name"): t for t in step.per_step_transformers_dicts
        }
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
        assert overscan_dict["distance_mm"] == pytest.approx(
            expected_base * 2
        )

    def test_includes_bidir_scan_offset_transformer(self, mock_context):
        step = MaterialTestStep.create(mock_context)
        per_wp_names = {
            t.get("name") for t in step.per_workpiece_transformers_dicts
        }
        assert "BidirScanOffsetTransformer" in per_wp_names
