import pytest
from laser_essentials.steps import WavefrontStep

from rayforge.core.workpiece import WorkPiece
from rayforge.pipeline.stage.assembler_helpers import MachineDefaults


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


class TestWavefrontComputePayload:
    def test_build_compute_payload_returns_wavefront_spec(
        self, machine_defaults
    ):
        from raygeo.cnc.execution.specs import ComputePayload
        from raygeo.ops.assembly import Assembler
        from raygeo.ops.assembly.wavefront import AdaptiveWavefrontSpec
        from raygeo.ops.part import Part

        step = WavefrontStep(name="wf")
        step.step_over_mm = 0.5
        wp = WorkPiece(name="wp")
        wp.set_size(10.0, 10.0)

        part, payload = step.build_compute_payload(machine_defaults, wp)
        assert isinstance(part, Part)
        assert isinstance(payload, ComputePayload)
        assert isinstance(payload.assembler, Assembler)
        spec = payload.assembler.spec
        assert isinstance(spec, AdaptiveWavefrontSpec)
        assert spec.step_over == 0.5

    def test_assembler_token_params_mirrors_kwargs(self, machine_defaults):
        step = WavefrontStep(name="wf")
        wp = WorkPiece(name="wp")
        wp.set_size(10.0, 10.0)
        token = step.assembler_token_params(machine_defaults, wp)
        kwargs = step.get_assembler_kwargs(machine_defaults, wp)
        assert token == kwargs
