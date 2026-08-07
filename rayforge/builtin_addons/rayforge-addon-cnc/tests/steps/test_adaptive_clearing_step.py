"""Tests for the adaptive-clearing CNC step."""

import pytest

from rayforge.core.workpiece import WorkPiece


@pytest.fixture
def adaptive_clear_step():
    from cnc_essentials.steps import AdaptiveClearStep

    step = AdaptiveClearStep(name="adaptive_clear")
    step.tool_diameter = 6.0
    step.step_over = 2.0
    step.step_length = 0.6
    step.max_deflection_deg = 30.0
    step.wall_margin = 0.0
    step.area_tolerance = 1.0
    step.target_depth = -5.0
    step.safe_z = 2.0
    return step


class TestAdaptiveClearSpec:
    def test_build_spec_returns_adaptive_clearing_spec(
        self, adaptive_clear_step
    ):
        from raygeo.ops.assembly.adaptive import AdaptiveClearingSpec

        wp = WorkPiece(name="wp")
        wp.set_size(60.0, 60.0)

        spec = adaptive_clear_step.build_spec(wp)

        assert isinstance(spec, AdaptiveClearingSpec)
        assert spec.tool_radius == 3.0
        assert spec.step_over == 2.0
        assert spec.step_length == 0.6
        assert spec.max_deflection_deg == 30.0
        assert spec.wall_margin == 0.0
        assert spec.area_tolerance == 1.0
        assert spec.target_z == -5.0
        assert spec.safe_z == 2.0

    def test_build_compute_payload_returns_part_and_payload(
        self, adaptive_clear_step, machine
    ):
        from raygeo.cnc.execution.specs import ComputePayload
        from raygeo.ops.assembly import Assembler
        from raygeo.ops.assembly.adaptive import AdaptiveClearingSpec
        from raygeo.ops.part import Part

        wp = WorkPiece(name="wp")
        wp.set_size(60.0, 60.0)

        part, payload = adaptive_clear_step.build_compute_payload(machine, wp)

        assert isinstance(part, Part)
        assert isinstance(payload, ComputePayload)
        assert isinstance(payload.assembler, Assembler)
        spec = payload.assembler.spec
        assert isinstance(spec, AdaptiveClearingSpec)
        assert spec.tool_radius == 3.0

    def test_populate_payload_stamps_spindle_power(
        self, adaptive_clear_step, machine
    ):
        """CNC payloads express power as the spindle's RPM / max RPM
        ratio, so a running spindle renders as a cut at the right
        intensity rather than the zero-power (no-cut) colour."""
        adaptive_clear_step.spindle_rpm = 15000
        wp = WorkPiece(name="wp")
        wp.set_size(60.0, 60.0)

        _part, payload = adaptive_clear_step.build_compute_payload(machine, wp)
        adaptive_clear_step.populate_payload(payload, machine)

        assert payload.power == 0.75  # 15000 / 20000 (fixture spindle max)
        assert payload.head_uid is not None

    def test_assembler_token_params_keys_and_values(
        self, adaptive_clear_step, machine
    ):
        wp = WorkPiece(name="wp")
        wp.set_size(60.0, 60.0)

        token = adaptive_clear_step.assembler_token_params(machine, wp)

        assert token["step_over"] == 2.0
        assert token["step_length"] == 0.6
        assert token["max_deflection_deg"] == 30.0
        assert token["wall_margin"] == 0.0
        assert token["area_tolerance"] == 1.0
        # Base CNC attributes are included too.
        assert token["tool_diameter"] == 6.0
        assert token["target_depth"] == -5.0
        assert token["safe_z"] == 2.0
