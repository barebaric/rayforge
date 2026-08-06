from unittest.mock import MagicMock

import pytest
from laser_essentials.capabilities import CUT, SCORE
from laser_essentials.steps import FrameStep

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


class TestFrameStep:
    def test_instantiation(self):
        step = FrameStep(name="Test")
        assert step.typelabel == "Frame"
        assert step.capabilities == (CUT, SCORE)

    def test_create(self, mock_context):
        step = FrameStep.create(mock_context)
        assert isinstance(step, FrameStep)

    def test_serialization_includes_step_type(self):
        step = FrameStep(name="Test")
        data = step.to_dict()
        assert data["step_type"] == "FrameStep"

    def test_get_assembler_kwargs(self, machine):
        step = FrameStep(name="Test")
        workpiece = MagicMock(spec=["size"])
        workpiece.size = (100, 100)
        kwargs = step.get_assembler_kwargs(machine, workpiece)
        assert isinstance(kwargs, dict)
        expected_keys = {"cut_side", "offset_mm"}
        assert set(kwargs.keys()) == expected_keys

    def test_roundtrip_serialization(self):
        step = FrameStep(name="Test")
        step.cut_side = "OUTSIDE"
        step.offset_mm = 0.5
        data = step.to_dict()
        restored = FrameStep.from_dict(data)
        assert data == restored.to_dict()

    def test_from_dict_migrates_legacy_opsproducer_params(self):
        """True legacy files store frame params in
        ``opsproducer_dict.params``; loading must restore them."""
        data = FrameStep(name="Test").to_dict()
        for key in ("cut_side", "offset_mm"):
            data.pop(key, None)
        data["opsproducer_dict"] = {
            "type": "FrameProducer",
            "params": {
                "path_offset_mm": 0.6,
                "cut_side": "OUTSIDE",
            },
        }

        restored = FrameStep.from_dict(data)

        assert restored.cut_side == "OUTSIDE"
        assert restored.offset_mm == pytest.approx(0.6)


class TestFrameComputePayload:
    def test_build_compute_payload_returns_frame_spec(self, machine):
        from raygeo.cnc.execution.specs import ComputePayload
        from raygeo.ops.assembly import Assembler
        from raygeo.ops.assembly.frame import FrameSpec
        from raygeo.ops.part import Part

        step = FrameStep(name="frame")
        step.cut_side = "outside"
        step.offset_mm = 0.3
        wp = WorkPiece(name="wp")
        wp.set_size(10.0, 10.0)

        part, payload = step.build_compute_payload(machine, wp)
        assert isinstance(part, Part)
        assert isinstance(payload, ComputePayload)
        assert isinstance(payload.assembler, Assembler)
        spec = payload.assembler.spec
        assert isinstance(spec, FrameSpec)
        assert spec.cut_side == "outside"
        assert spec.offset_mm == 0.3

    def test_assembler_token_params_mirrors_kwargs(self, machine):
        step = FrameStep(name="frame")
        wp = WorkPiece(name="wp")
        wp.set_size(10.0, 10.0)
        token = step.assembler_token_params(machine, wp)
        kwargs = step.get_assembler_kwargs(machine, wp)
        assert token == kwargs
