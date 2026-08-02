from laser_essentials.steps import WavefrontStep

from rayforge.core.workpiece import WorkPiece


def _make_disjoint_loops():
    """Two widely-separated square pockets as a single Geometry."""
    from raygeo.geo import Geometry

    loops = Geometry()
    left = [(-30.0, -20.0), (30.0, -20.0), (30.0, 20.0), (-30.0, 20.0)]
    right = [(70.0, -20.0), (110.0, -20.0), (110.0, 20.0), (70.0, 20.0)]
    for loop in (left, right):
        loops.move_to(*loop[0])
        for p in loop[1:]:
            loops.line_to(*p)
        loops.close_path()
    return loops


class _FakeProvider:
    """A minimal IGeometryProvider returning a fixed Geometry."""

    def __init__(self, geometry, name="fake"):
        from blinker import Signal

        self._geometry = geometry
        self.name = name
        self.updated = Signal()

    @property
    def uid(self) -> str:
        return "fake-provider-uid"

    @property
    def provider_type_name(self) -> str:
        return "fake"

    @property
    def renderer(self):
        return None

    def get_geometry(self, params=None, *, resolved_text_cache=None):
        return self._geometry.copy(), []

    def to_dict(self):
        return {}


def _make_workpiece():
    return WorkPiece.from_geometry_provider(
        _FakeProvider(_make_disjoint_loops())
    )


class TestWavefrontComputePayload:
    def test_build_compute_payload_returns_wavefront_spec(self, machine):
        from raygeo.cnc.execution.specs import ComputePayload
        from raygeo.ops.assembly import Assembler
        from raygeo.ops.assembly.wavefront import AdaptiveWavefrontSpec
        from raygeo.ops.part import Part

        step = WavefrontStep(name="wf")
        step.step_over_mm = 0.5
        wp = WorkPiece(name="wp")
        wp.set_size(10.0, 10.0)

        part, payload = step.build_compute_payload(machine, wp)
        assert isinstance(part, Part)
        assert isinstance(payload, ComputePayload)
        assert isinstance(payload.assembler, Assembler)
        spec = payload.assembler.spec
        assert isinstance(spec, AdaptiveWavefrontSpec)
        assert spec.step_over == 0.5

    def test_assembler_token_params_mirrors_kwargs(self, machine):
        step = WavefrontStep(name="wf")
        wp = WorkPiece(name="wp")
        wp.set_size(10.0, 10.0)
        token = step.assembler_token_params(machine, wp)
        kwargs = step.get_assembler_kwargs(machine, wp)
        assert token == kwargs

    def test_disjoint_pockets_become_separate_faces(self, machine):
        """A workpiece with two pockets maps to two wavefront faces."""
        step = WavefrontStep(name="wf")
        step.step_over_mm = 2.0
        wp = _make_workpiece()

        part, _ = step.build_compute_payload(machine, wp)

        assert part is not None
        assert len(part.face_ids) == 2
        assert "" in part.face_ids

    def test_single_pocket_keeps_default_face(self, machine):
        """A single-pocket workpiece keeps the default face ``""``."""
        step = WavefrontStep(name="wf")
        wp = WorkPiece(name="wp")
        wp.set_size(10.0, 10.0)

        part, _ = step.build_compute_payload(machine, wp)

        assert part is not None
        assert part.face_ids == [""]

    def test_wavefront_clears_all_faces(self, machine):
        """Running the payload through the pipeline clears every pocket,
        not just the largest one."""
        from raygeo.pipeline.execute import clear_cache, execute_stages
        from raygeo.pipeline.request import NodeRequest
        from raygeo.pipeline.stage import StageSpec

        step = WavefrontStep(name="wf")
        step.step_over_mm = 2.0
        wp = _make_workpiece()

        part, payload = step.build_compute_payload(machine, wp)
        assert len(part.face_ids) == 2

        clear_cache()
        completed = []
        node = NodeRequest(
            key="wf",
            generation_id=1,
            stage=StageSpec.Compute(part=part, params=payload),
        )
        execute_stages([node], completed.append, None)
        assert len(completed) == 1
        out = completed[0].output
        assert getattr(out, "warnings", None) == []
        assert out.ops.len() > 0
        assert any(out.ops.is_cutting(i) for i in range(out.ops.len()))
