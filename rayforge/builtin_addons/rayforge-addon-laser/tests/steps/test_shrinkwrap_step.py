from unittest.mock import MagicMock

import pytest
from laser_essentials.capabilities import CUT, SCORE, WITH_KERF
from laser_essentials.steps import ShrinkWrapStep

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


class TestShrinkWrapStep:
    def test_instantiation(self):
        step = ShrinkWrapStep(name="Test")
        assert step.typelabel == "Shrink Wrap"
        assert step.capabilities == (CUT, SCORE, WITH_KERF)

    def test_create(self, mock_context):
        step = ShrinkWrapStep.create(mock_context)
        assert isinstance(step, ShrinkWrapStep)

    def test_serialization_includes_step_type(self):
        step = ShrinkWrapStep(name="Test")
        data = step.to_dict()
        assert data["step_type"] == "ShrinkWrapStep"

    def test_get_assembler_kwargs(self, machine):
        step = ShrinkWrapStep(name="Test")
        workpiece = MagicMock(spec=["size"])
        workpiece.size = (100, 100)
        kwargs = step.get_assembler_kwargs(machine, workpiece)
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


class TestShrinkWrapComputePayload:
    def test_build_compute_payload_returns_shrinkwrap_spec(self, machine):
        from raygeo.cnc.execution.specs import ComputePayload
        from raygeo.ops.assembly import Assembler
        from raygeo.ops.assembly.shrinkwrap import ShrinkwrapSpec
        from raygeo.ops.part import Part

        step = ShrinkWrapStep(name="sw")
        step.cut_side = "outside"
        step.gravity = 0.3
        wp = WorkPiece(name="wp")
        wp.set_size(10.0, 10.0)

        part, payload = step.build_compute_payload(machine, wp)
        assert isinstance(part, Part)
        assert isinstance(payload, ComputePayload)
        assert isinstance(payload.assembler, Assembler)
        spec = payload.assembler.spec
        assert isinstance(spec, ShrinkwrapSpec)
        assert spec.cut_side == "outside"
        assert spec.gravity == 0.3
        assert spec.kerf_mm == step.kerf_mm

    def test_assembler_token_params_mirrors_kwargs(self, machine):
        step = ShrinkWrapStep(name="sw")
        wp = WorkPiece(name="wp")
        wp.set_size(10.0, 10.0)
        token = step.assembler_token_params(machine, wp)
        kwargs = step.get_assembler_kwargs(machine, wp)
        assert token == kwargs
