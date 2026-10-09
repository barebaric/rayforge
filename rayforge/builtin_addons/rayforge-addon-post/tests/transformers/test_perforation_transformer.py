from unittest.mock import MagicMock

import pytest
from post_processors.transformers import PerforationTransformer
from raygeo.geo import Geometry
from raygeo.ops import Ops
from raygeo.ops.transform.tabs import TabsSpec
from raygeo.ops.types import CommandType, SectionType

from rayforge.core.workpiece import WorkPiece


def _unit_square(x0=0.0, y0=0.0, x1=1.0, y1=1.0) -> Geometry:
    geo = Geometry()
    geo.move_to(x0, y0)
    geo.line_to(x1, y0)
    geo.line_to(x1, y1)
    geo.line_to(x0, y1)
    geo.close_path()
    return geo


def _workpiece(boundaries, size=(20.0, 20.0)):
    wp = MagicMock(spec=WorkPiece)
    wp.boundaries = boundaries
    wp.size = size
    return wp


def _square_ops(x0, y0, x1, y1) -> Ops:
    ops = Ops()
    ops.set_power(1.0)
    ops.ops_section_start(
        section_type=SectionType.VECTOR_OUTLINE, workpiece_uid="wp"
    )
    ops.move_to(x0, y0, 0.0)
    ops.line_to(x1, y0, 0.0)
    ops.line_to(x1, y1, 0.0)
    ops.line_to(x0, y1, 0.0)
    ops.line_to(x0, y0, 0.0)
    ops.ops_section_end(section_type=SectionType.VECTOR_OUTLINE)
    return ops


def _cut_runs(ops: Ops) -> int:
    return sum(
        1
        for i in range(ops.len())
        if ops.command_type(i) == CommandType.MOVE_TO
    )


class TestPerforationConfig:
    def test_defaults(self):
        t = PerforationTransformer()
        assert t.enabled is True
        assert t.cut_length == pytest.approx(2.0)
        assert t.skip_length == pytest.approx(1.0)

    def test_label_and_description(self):
        t = PerforationTransformer()
        assert t.label
        assert t.description

    def test_lengths_are_clamped_to_a_minimum(self):
        t = PerforationTransformer(cut_length=0.0, skip_length=-1.0)
        assert t.cut_length >= 0.01
        assert t.skip_length >= 0.01

    def test_setter_sends_changed(self):
        t = PerforationTransformer()
        received = []

        def on_changed(sender):
            received.append(sender)

        t.changed.connect(on_changed)
        t.cut_length = 3.0
        t.skip_length = 0.5
        assert len(received) == 2
        assert t.cut_length == pytest.approx(3.0)
        assert t.skip_length == pytest.approx(0.5)

    def test_dict_round_trip(self):
        t = PerforationTransformer(
            enabled=False, cut_length=1.5, skip_length=0.5
        )
        data = t.to_dict()
        assert data["name"] == "PerforationTransformer"
        restored = PerforationTransformer.from_dict(data)
        assert restored.enabled is False
        assert restored.cut_length == pytest.approx(1.5)
        assert restored.skip_length == pytest.approx(0.5)

    def test_from_dict_defaults(self):
        restored = PerforationTransformer.from_dict(
            {"name": "PerforationTransformer"}
        )
        assert restored.cut_length == pytest.approx(2.0)
        assert restored.skip_length == pytest.approx(1.0)

    def test_from_dict_rejects_other_name(self):
        with pytest.raises(ValueError):
            PerforationTransformer.from_dict({"name": "Optimize"})


class TestPerforationSpec:
    def test_spec_is_gap_mode_tabs(self):
        t = PerforationTransformer(cut_length=1.5, skip_length=0.5)
        spec = t.to_spec(_workpiece(_unit_square()), None, None)
        assert isinstance(spec, TabsSpec)
        assert spec.tab_power == 0.0

    def test_no_workpiece_gives_no_clips(self):
        spec = PerforationTransformer().to_spec(None, None, None)
        assert spec.clips == []

    def test_empty_boundaries_give_no_clips(self):
        wp = _workpiece(Geometry())
        spec = PerforationTransformer().to_spec(wp, None, None)
        assert spec.clips == []

    def test_square_gets_one_gap_per_period(self):
        t = PerforationTransformer(cut_length=1.5, skip_length=0.5)
        spec = t.to_spec(_workpiece(_unit_square()), None, None)
        assert len(spec.clips) == 40
        assert all(w == pytest.approx(0.5) for _x, _y, w in spec.clips)

    def test_first_gap_follows_a_full_cut(self):
        t = PerforationTransformer(cut_length=1.5, skip_length=0.5)
        spec = t.to_spec(_workpiece(_unit_square()), None, None)
        x, y, _w = spec.clips[0]
        assert x == pytest.approx(1.75)
        assert y == pytest.approx(0.0)

    def test_clips_are_in_millimetres(self):
        t = PerforationTransformer(cut_length=4.0, skip_length=1.0)
        wp = _workpiece(_unit_square(), size=(40.0, 10.0))
        spec = t.to_spec(wp, None, None)
        xs = [x for x, _y, _w in spec.clips]
        ys = [y for _x, y, _w in spec.clips]
        assert max(xs) <= 40.0 + 1e-6
        assert max(ys) <= 10.0 + 1e-6
        assert len(spec.clips) == 20

    def test_gap_never_wraps_past_the_contour_end(self):
        t = PerforationTransformer(cut_length=3.0, skip_length=1.0)
        spec = t.to_spec(_workpiece(_unit_square()), None, None)
        assert len(spec.clips) == 20
        x, y, _w = spec.clips[-1]
        assert (x, y) == (pytest.approx(0.0), pytest.approx(0.5))

    def test_pattern_restarts_for_every_contour(self):
        geo = _unit_square(0.0, 0.0, 0.25, 0.25)
        geo.extend(_unit_square(0.5, 0.5, 0.75, 0.75))
        t = PerforationTransformer(cut_length=1.5, skip_length=0.5)
        spec = t.to_spec(_workpiece(geo, size=(40.0, 40.0)), None, None)
        assert len(spec.clips) == 40
        starts = [c for c in spec.clips if c[1] == pytest.approx(20.0)]
        assert starts[0][0] == pytest.approx(21.75)

    def test_last_gap_survives_rounding_of_the_contour_length(self):
        geo = Geometry()
        geo.move_to(0.0, 0.0)
        geo.line_to(1.0 - 1e-7, 0.0)
        t = PerforationTransformer(cut_length=3.0, skip_length=1.5)
        spec = t.to_spec(_workpiece(geo, size=(90.0, 1.0)), None, None)
        assert len(spec.clips) == 20

    def test_contour_shorter_than_a_period_is_left_whole(self):
        geo = _unit_square(0.0, 0.0, 0.01, 0.01)
        t = PerforationTransformer(cut_length=1.5, skip_length=0.5)
        spec = t.to_spec(_workpiece(geo), None, None)
        assert spec.clips == []


class TestPerforationOnOps:
    def test_cut_and_skip_lengths_on_square(self):
        t = PerforationTransformer(cut_length=1.5, skip_length=0.5)
        spec = t.to_spec(_workpiece(_unit_square()), None, None)
        ops = _square_ops(0.0, 0.0, 20.0, 20.0)
        Ops.apply_transformers(ops, [spec])
        ops.preload_state()
        assert ops.cut_distance() == pytest.approx(60.0, abs=1e-6)
        assert _cut_runs(ops) == 40
