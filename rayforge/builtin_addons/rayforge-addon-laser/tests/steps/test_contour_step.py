from unittest.mock import MagicMock

import pytest
from laser_essentials.capabilities import CUT, SCORE
from laser_essentials.steps import ContourStep
from raygeo.cnc.execution.specs import ComputePayload
from raygeo.geo import Matrix
from raygeo.ops.assembly import Assembler
from raygeo.ops.assembly.contour import ContourSpec

from rayforge.core.step import Step
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


class TestContourStep:
    def test_instantiation(self):
        step = ContourStep(name="Test")
        assert step.typelabel == "Contour"
        assert step.name == "Test"
        assert step.capabilities == (CUT, SCORE)

    def test_create(self, mock_context):
        step = ContourStep.create(mock_context, name="Created")
        assert isinstance(step, ContourStep)
        assert step.name == "Created"
        assert len(step.per_workpiece_transformers_dicts) == 5
        assert len(step.per_step_transformers_dicts) == 3
        assert step.selected_head_uid == "test-laser-uid"

    def test_create_without_optimize(self, mock_context):
        step = ContourStep.create(mock_context, optimize=False)
        assert len(step.per_workpiece_transformers_dicts) == 4

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

    def test_from_dict_adds_new_transformers_from_old_project(self):
        step_registry.register(ContourStep)
        old_project_data = {
            "uid": "old-step-123",
            "type": "step",
            "step_type": "ContourStep",
            "name": "Old Contour",
            "matrix": Matrix.identity().to_list(),
            "typelabel": "Contour",
            "visible": True,
            "per_workpiece_transformers_dicts": [
                {"name": "TabOpsTransformer", "enabled": True},
            ],
            "children": [],
        }

        restored = Step.from_dict(old_project_data)

        wp_names = [
            t["name"] for t in restored.per_workpiece_transformers_dicts
        ]
        assert "TabOpsTransformer" in wp_names
        assert "Smooth" in wp_names
        assert "CropTransformer" in wp_names
        assert "Optimize" in wp_names
        assert len(restored.per_step_transformers_dicts) == 3
        step_names = [t["name"] for t in restored.per_step_transformers_dicts]
        assert "MergeLinesTransformer" in step_names
        assert "Optimize" in step_names
        assert "MultiPassTransformer" in step_names

    def test_from_dict_preserves_existing_transformer_settings(self):
        step_registry.register(ContourStep)
        old_project_data = {
            "uid": "old-step-456",
            "type": "step",
            "step_type": "ContourStep",
            "name": "Old Contour",
            "matrix": Matrix.identity().to_list(),
            "typelabel": "Contour",
            "visible": True,
            "per_workpiece_transformers_dicts": [
                {
                    "name": "TabOpsTransformer",
                    "enabled": True,
                    "custom_setting": 42,
                },
            ],
            "per_step_transformers_dicts": [],
            "children": [],
        }

        restored = Step.from_dict(old_project_data)

        tab_transformer = next(
            t
            for t in restored.per_workpiece_transformers_dicts
            if t["name"] == "TabOpsTransformer"
        )
        assert tab_transformer["custom_setting"] == 42
        assert tab_transformer["enabled"] is True

    def test_from_dict_uses_typelabel_fallback_when_no_step_type(self):
        step_registry.register(ContourStep)
        old_project_data = {
            "uid": "old-step-789",
            "type": "step",
            "name": "Old Contour",
            "matrix": Matrix.identity().to_list(),
            "typelabel": "Contour",
            "visible": True,
            "per_workpiece_transformers_dicts": [
                {"name": "TabOpsTransformer", "enabled": True},
            ],
            "children": [],
        }

        restored = Step.from_dict(old_project_data)

        assert isinstance(restored, ContourStep)
        wp_names = [
            t["name"] for t in restored.per_workpiece_transformers_dicts
        ]
        assert "CropTransformer" in wp_names

    def test_optimize_dict_is_shared_between_lists(self):
        step_registry.register(ContourStep)
        data = {
            "uid": "test-step",
            "type": "step",
            "step_type": "ContourStep",
            "name": "Test",
            "matrix": Matrix.identity().to_list(),
            "typelabel": "Contour",
            "visible": True,
            "per_workpiece_transformers_dicts": [
                {"name": "Optimize", "enabled": True},
            ],
            "per_step_transformers_dicts": [
                {"name": "Optimize", "enabled": True},
            ],
            "children": [],
        }

        restored = Step.from_dict(data)

        wp_optimize = next(
            t
            for t in restored.per_workpiece_transformers_dicts
            if t["name"] == "Optimize"
        )
        step_optimize = next(
            t
            for t in restored.per_step_transformers_dicts
            if t["name"] == "Optimize"
        )

        assert wp_optimize is step_optimize

    def test_get_assembler_kwargs(self, machine):
        step = ContourStep(name="Test")
        workpiece = MagicMock(spec=["size"])
        workpiece.size = (100, 100)
        kwargs = step.get_assembler_kwargs(machine, workpiece)
        assert isinstance(kwargs, dict)
        expected_keys = {
            "cut_side",
            "cut_order",
            "remove_inner",
            "offset_mm",
            "overcut",
            "arc_tolerance",
            "allow_arcs",
            "supports_curves",
        }
        assert set(kwargs.keys()) == expected_keys

    def test_roundtrip_serialization(self):
        step_registry.register(ContourStep)
        step = ContourStep(name="Test")
        step.cut_side = "OUTSIDE"
        step.cut_order = "OUTSIDE_INSIDE"
        step.remove_inner_paths = True
        step.offset_mm = 0.5
        step.overcut = 1.0
        data = step.to_dict()
        restored = ContourStep.from_dict(data)
        assert data == restored.to_dict()

    def test_from_dict_migrates_legacy_offset_keys(self):
        """Legacy files store path_offset_mm and kerf_mm; the combined
        displacement is offset_mm = path_offset_mm + kerf_mm / 2."""
        step_registry.register(ContourStep)
        legacy_data = {
            "uid": "legacy-step",
            "type": "step",
            "step_type": "ContourStep",
            "name": "Legacy",
            "matrix": Matrix.identity().to_list(),
            "typelabel": "Contour",
            "visible": True,
            "path_offset_mm": 0.4,
            "kerf_mm": 0.2,
            "per_workpiece_transformers_dicts": [],
            "per_step_transformers_dicts": [],
            "children": [],
        }

        restored = Step.from_dict(legacy_data)

        assert isinstance(restored, ContourStep)
        assert restored.offset_mm == pytest.approx(0.5)

    def test_from_dict_new_offset_key_wins(self):
        """A current file's offset_mm is used verbatim, ignoring any
        legacy keys that may also be present."""
        step_registry.register(ContourStep)
        data = {
            "uid": "new-step",
            "type": "step",
            "step_type": "ContourStep",
            "name": "New",
            "matrix": Matrix.identity().to_list(),
            "typelabel": "Contour",
            "visible": True,
            "path_offset_mm": 0.4,
            "kerf_mm": 0.2,
            "offset_mm": 1.5,
            "per_workpiece_transformers_dicts": [],
            "per_step_transformers_dicts": [],
            "children": [],
        }

        restored = Step.from_dict(data)

        assert isinstance(restored, ContourStep)
        assert restored.offset_mm == pytest.approx(1.5)

    def test_from_dict_migrates_legacy_opsproducer_params(self):
        """True legacy files store contour params in
        ``opsproducer_dict.params``; loading must restore them."""
        step_registry.register(ContourStep)
        data = ContourStep(name="Test").to_dict()
        for key in (
            "cut_side",
            "cut_order",
            "remove_inner_paths",
            "offset_mm",
            "overcut",
            "override_threshold",
            "threshold",
        ):
            data.pop(key, None)
        data["opsproducer_dict"] = {
            "type": "ContourProducer",
            "params": {
                "remove_inner_paths": True,
                "path_offset_mm": 0.4,
                "cut_side": "OUTSIDE",
                "cut_order": "OUTSIDE_INSIDE",
                "override_threshold": True,
                "threshold": 0.7,
                "overcut": 0.2,
            },
        }

        restored = ContourStep.from_dict(data)

        assert restored.cut_side == "OUTSIDE"
        assert restored.cut_order == "OUTSIDE_INSIDE"
        assert restored.remove_inner_paths is True
        assert restored.override_threshold is True
        assert restored.threshold == 0.7
        assert restored.overcut == 0.2
        assert restored.offset_mm == pytest.approx(0.4)

    def test_step_from_dict_preserves_subclass_attrs(self):
        """Step.from_dict (base call) must delegate to subclass from_dict."""
        step_registry.register(ContourStep)
        step = ContourStep(name="Test")
        step.cut_side = "OUTSIDE"
        step.offset_mm = 0.5
        step.cut_speed = 200
        step.power = 80
        data = step.to_dict()

        restored = Step.from_dict(data)
        assert isinstance(restored, ContourStep)
        assert restored.cut_side == "OUTSIDE"
        assert restored.offset_mm == 0.5
        assert restored.cut_speed == 200
        assert restored.power == 80


class TestContourComputePayload:
    """Verifies ContourStep's contribution to the raygeo intent pipeline
    (see target-architecture.md slice B2)."""

    def _wp(self):
        return WorkPiece(name="wp")

    def test_build_compute_payload_returns_contour_spec(self, machine):
        step = ContourStep(name="cut")
        step.cut_side = "outside"
        step.offset_mm = 0.5
        step.overcut = 0.2

        part, payload = step.build_compute_payload(machine, self._wp())
        assert isinstance(payload, ComputePayload)
        assert isinstance(payload.assembler, Assembler)
        spec = payload.assembler.spec
        assert isinstance(spec, ContourSpec)
        assert spec.cut_side == "outside"
        assert spec.offset_mm == 0.5
        assert spec.overcut == 0.2
        assert spec.arc_tolerance == machine.arc_tolerance
        assert spec.allow_arcs == machine.supports_arcs
        assert spec.supports_curves == machine.supports_curves

    def test_build_compute_payload_reflects_cut_order(self, machine):
        step = ContourStep(name="cut")
        step.cut_order = "OUTSIDE_INSIDE"

        wp = self._wp()
        _part, payload = step.build_compute_payload(machine, wp)
        spec = payload.assembler.spec
        assert spec.cut_order == "outside_inside"

    def test_assembler_token_params_mirrors_assembler_kwargs(self, machine):
        step = ContourStep(name="cut")
        step.cut_side = "inside"
        wp = self._wp()

        token_params = step.assembler_token_params(machine, wp)
        kwargs = step.get_assembler_kwargs(machine, wp)
        assert token_params == kwargs
        assert token_params is not None
        assert token_params["cut_side"] == "inside"
