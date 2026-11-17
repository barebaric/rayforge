"""Tests for the ramp-entry CNC step."""

from rayforge.core.workpiece import WorkPiece


def make_step():
    from cnc_essentials.steps import RampEntryStep

    step = RampEntryStep(name="ramp_entry")
    step.tool_diameter = 8.0
    step.target_depth = -4.0
    return step


class TestRampEntrySpec:
    def test_build_spec_returns_ramp_spec(self):
        from raygeo.ops.assembly.ramp import RampSpec

        step = make_step()
        wp = WorkPiece(name="wp")
        wp.set_size(60.0, 60.0)

        spec = step.build_spec(wp)

        assert isinstance(spec, RampSpec)
        assert spec.start == (26.0, 30.0)
        assert spec.end == (34.0, 30.0)
        assert spec.z_start == 0.0
        assert spec.z_end == -4.0

    def test_build_spec_passes_tool_radius(self):
        """The ramp seeds its cleared area with the swath it machined,
        so the spec must carry this step's real tool radius instead of
        the raygeo default."""
        from raygeo.ops.assembly.ramp import RampSpec

        step = make_step()
        wp = WorkPiece(name="wp")
        wp.set_size(60.0, 60.0)

        spec = step.build_spec(wp)

        assert isinstance(spec, RampSpec)
        assert spec.tool_radius == 4.0
