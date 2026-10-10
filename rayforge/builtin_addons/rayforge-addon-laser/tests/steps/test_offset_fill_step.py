from unittest.mock import MagicMock

import pytest
from blinker import Signal
from laser_essentials import worker
from laser_essentials.offset_fill import offset_fill_faces, offset_fill_rings
from laser_essentials.steps import OffsetFillStep
from raygeo.cnc.execution.specs import ComputePayload
from raygeo.geo import Geometry
from raygeo.ops.assembly import Assembler
from raygeo.ops.assembly.contour import ContourSpec
from raygeo.ops.part import Part
from raygeo.pipeline.execute import clear_cache, execute_stages
from raygeo.pipeline.request import NodeRequest
from raygeo.pipeline.stage import StageSpec

from rayforge.core.step import Step
from rayforge.core.step_registry import step_registry
from rayforge.core.workpiece import WorkPiece


def _rect(geo, x0, y0, x1, y1):
    geo.move_to(x0, y0)
    geo.line_to(x1, y0)
    geo.line_to(x1, y1)
    geo.line_to(x0, y1)
    geo.close_path()
    return geo


def _square(size=10.0):
    return _rect(Geometry(), 0.0, 0.0, size, size)


class _FakeProvider:
    def __init__(self, geometry, name="fake"):
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


def _workpiece(geometry):
    return WorkPiece.from_geometry_provider(_FakeProvider(geometry))


def _run(part, payload):
    clear_cache()
    completed = []
    node = NodeRequest(
        key="offset-fill",
        generation_id=1,
        stage=StageSpec.Compute(part=part, params=payload),
    )
    execute_stages([node], completed.append, None)
    assert len(completed) == 1
    return completed[0].output.ops


class TestOffsetFillRings:
    def test_square_gets_one_ring_per_interval(self):
        rings = offset_fill_rings(_square(), step=1.0)
        lengths = [round(r.distance(), 6) for r in rings]
        assert lengths == [40.0, 32.0, 24.0, 16.0, 8.0]

    def test_first_ring_is_the_outline(self):
        rings = offset_fill_rings(_square(), step=1.0)
        assert rings[0].rect() == pytest.approx((0.0, 0.0, 10.0, 10.0))

    def test_offset_moves_the_first_ring_inward(self):
        rings = offset_fill_rings(_square(), step=1.0, offset=0.5)
        assert rings[0].distance() == pytest.approx(36.0)
        assert rings[0].rect() == pytest.approx((0.5, 0.5, 9.5, 9.5))

    def test_holes_are_kept_free(self):
        geo = _rect(Geometry(), 0.0, 0.0, 20.0, 10.0)
        _rect(geo, 5.0, 3.0, 15.0, 7.0)
        rings = offset_fill_rings(geo, step=1.0)
        assert len(rings) == 3
        for ring in rings:
            for contour in ring.split_into_contours():
                x0, y0, x1, y1 = contour.rect()
                inside_hole = x0 > 5.0 and x1 < 15.0 and y0 > 3.0
                assert not (inside_hole and y1 < 7.0)

    def test_open_paths_are_ignored(self):
        geo = _square()
        geo.move_to(30.0, 0.0)
        geo.line_to(40.0, 5.0)
        rings = offset_fill_rings(geo, step=1.0)
        for ring in rings:
            assert ring.rect()[2] <= 10.0 + 1e-9

    def test_only_open_paths_give_no_rings(self):
        geo = Geometry()
        geo.move_to(0.0, 0.0)
        geo.line_to(10.0, 0.0)
        assert offset_fill_rings(geo, step=1.0) == []

    def test_ring_count_is_capped(self):
        rings = offset_fill_rings(_square(), step=0.001, max_rings=50)
        assert len(rings) == 50

    def test_step_must_be_positive(self):
        with pytest.raises(ValueError):
            offset_fill_rings(_square(), step=0.0)


def _two_squares():
    geo = _rect(Geometry(), 0.0, 0.0, 10.0, 10.0)
    return _rect(geo, 20.0, 0.0, 26.0, 6.0)


class TestOffsetFillFaces:
    def test_every_shape_is_filled(self):
        faces = offset_fill_faces(_two_squares(), step=1.0)
        total = sum(f.distance() for f in faces)
        assert total == pytest.approx(120.0 + 24.0 + 16.0 + 8.0)

    def test_each_face_is_one_shape(self):
        faces = offset_fill_faces(_two_squares(), step=1.0)
        for face in faces:
            assert len(face.split_into_components()) == 1

    def test_shapes_are_filled_one_after_the_other(self):
        faces = offset_fill_faces(_two_squares(), step=1.0)
        sides = [face.rect()[0] >= 20.0 for face in faces]
        assert sides == sorted(sides) or sides == sorted(sides)[::-1]

    def test_outside_inside_starts_with_the_outline(self):
        faces = offset_fill_faces(_square(), step=1.0)
        assert faces[0].distance() == pytest.approx(40.0)
        assert faces[-1].distance() == pytest.approx(8.0)

    def test_inside_outside_ends_with_the_outline(self):
        faces = offset_fill_faces(_square(), step=1.0, inside_out=True)
        assert faces[0].distance() == pytest.approx(8.0)
        assert faces[-1].distance() == pytest.approx(40.0)


class TestOffsetFillStep:
    def test_typelabel(self):
        step = OffsetFillStep(name="fill")
        assert step.typelabel == "Offset Fill"
        assert step.name == "fill"

    def test_worker_registers_the_step(self):
        registry = MagicMock()
        worker.register_steps(registry)
        registered = [c.args[0] for c in registry.register.call_args_list]
        assert OffsetFillStep in registered

    def test_recipe_keys(self):
        keys = OffsetFillStep.recipe_keys()
        for key in ("line_interval_mm", "offset_mm", "fill_direction"):
            assert key in keys

    def test_defaults(self):
        step = OffsetFillStep()
        assert step.line_interval_mm is None
        assert step.offset_mm == 0.0
        assert step.fill_direction == "OUTSIDE_INSIDE"

    def test_assembler_kwargs_default_interval_is_spot_size(self, machine):
        step = OffsetFillStep()
        wp = _workpiece(_square())
        kwargs = step.get_assembler_kwargs(machine, wp)
        assert kwargs["line_interval_mm"] == pytest.approx(0.1)
        assert kwargs["offset_mm"] == 0.0
        assert kwargs["cut_order"] == "outside_inside"

    def test_assembler_token_params_mirror_kwargs(self, machine):
        step = OffsetFillStep()
        step.line_interval_mm = 0.5
        wp = _workpiece(_square())
        token = step.assembler_token_params(machine, wp)
        assert token == step.get_assembler_kwargs(machine, wp)

    def test_build_compute_payload_uses_contour_assembler(self, machine):
        step = OffsetFillStep()
        step.line_interval_mm = 1.0
        step.fill_direction = "INSIDE_OUTSIDE"
        wp = _workpiece(_square())
        part, payload = step.build_compute_payload(machine, wp)
        assert isinstance(part, Part)
        assert isinstance(payload, ComputePayload)
        assert isinstance(payload.assembler, Assembler)
        spec = payload.assembler.spec
        assert isinstance(spec, ContourSpec)
        assert spec.cut_side == "centerline"
        assert spec.cut_order == "inside_outside"
        assert spec.offset_mm == 0.0
        assert spec.overcut == 0.0

    def test_part_holds_all_rings(self, machine):
        step = OffsetFillStep()
        step.line_interval_mm = 1.0
        wp = _workpiece(_square())
        part, _payload = step.build_compute_payload(machine, wp)
        rings = [part.face(f).geometry for f in part.face_ids]
        rings = [r for r in rings if r is not None and not r.is_empty()]
        assert len(rings) == 5
        assert sum(r.distance() for r in rings) == pytest.approx(120.0)

    def test_pipeline_fills_every_shape(self, machine):
        step = OffsetFillStep()
        step.line_interval_mm = 1.0
        wp = _workpiece(_two_squares())
        part, payload = step.build_compute_payload(machine, wp)
        ops = _run(part, payload)
        ops.preload_state()
        assert ops.cut_distance() == pytest.approx(168.0, abs=1e-3)

    def test_pipeline_cuts_every_ring(self, machine):
        step = OffsetFillStep()
        step.line_interval_mm = 1.0
        wp = _workpiece(_square())
        part, payload = step.build_compute_payload(machine, wp)
        ops = _run(part, payload)
        ops.preload_state()
        assert ops.cut_distance() == pytest.approx(120.0, abs=1e-3)

    def test_outside_inside_cuts_the_outline_first(self, machine):
        step = OffsetFillStep()
        step.line_interval_mm = 1.0
        wp = _workpiece(_square())
        part, payload = step.build_compute_payload(machine, wp)
        ops = _run(part, payload)
        ops.preload_state()
        first = next(i for i in range(ops.len()) if ops.is_cutting(i))
        x, y, _z = ops.endpoint(first)
        assert x in (pytest.approx(0.0), pytest.approx(10.0))
        assert y in (pytest.approx(0.0), pytest.approx(10.0))

    def test_inside_outside_cuts_the_outline_last(self, machine):
        step = OffsetFillStep()
        step.line_interval_mm = 1.0
        step.fill_direction = "INSIDE_OUTSIDE"
        wp = _workpiece(_square())
        part, payload = step.build_compute_payload(machine, wp)
        ops = _run(part, payload)
        ops.preload_state()
        last = max(i for i in range(ops.len()) if ops.is_cutting(i))
        x, y, _z = ops.endpoint(last)
        assert x in (pytest.approx(0.0), pytest.approx(10.0))
        assert y in (pytest.approx(0.0), pytest.approx(10.0))

    def test_serialization_round_trip(self):
        step_registry.register(OffsetFillStep)
        step = OffsetFillStep(name="fill")
        step.line_interval_mm = 0.3
        step.offset_mm = 0.2
        step.fill_direction = "INSIDE_OUTSIDE"
        data = step.to_dict()
        assert data["step_type"] == "OffsetFillStep"
        restored = Step.from_dict(data)
        assert isinstance(restored, OffsetFillStep)
        assert restored.line_interval_mm == pytest.approx(0.3)
        assert restored.offset_mm == pytest.approx(0.2)
        assert restored.fill_direction == "INSIDE_OUTSIDE"

    def test_from_dict_defaults(self):
        step_registry.register(OffsetFillStep)
        data = OffsetFillStep(name="fill").to_dict()
        for key in ("line_interval_mm", "offset_mm", "fill_direction"):
            data.pop(key)
        restored = Step.from_dict(data)
        assert restored.line_interval_mm is None
        assert restored.offset_mm == 0.0
        assert restored.fill_direction == "OUTSIDE_INSIDE"

    def test_create_sets_defaults_from_machine(self):
        context = MagicMock()
        machine = MagicMock()
        machine.max_cut_speed = 5000
        machine.max_travel_speed = 10000
        machine.acceleration = 3000
        head = MagicMock()
        head.uid = "head-1"
        machine.get_default_laser_head.return_value = head
        context.machine = machine
        step = OffsetFillStep.create(context, name="created")
        assert step.selected_head_uid == "head-1"
        assert step.cut_speed == 500
        names = [t["name"] for t in step.per_workpiece_transformers_dicts]
        assert "Optimize" in names
        step_names = [t["name"] for t in step.per_step_transformers_dicts]
        assert "MultiPassTransformer" in step_names
