import pytest

from rayforge.core.doc import Doc
from rayforge.core.job_origin import (
    JobAnchor,
    JobOrigin,
    StartFrom,
    anchor_point,
    placement_offset,
)

RECT = (10.0, 20.0, 50.0, 40.0)


class TestJobOriginModel:
    def test_default_is_absolute_bottom_left(self):
        origin = JobOrigin()
        assert origin.start_from == StartFrom.ABSOLUTE
        assert origin.anchor == JobAnchor.BOTTOM_LEFT
        assert origin.is_absolute

    def test_relative_modes_are_not_absolute(self):
        assert not JobOrigin(StartFrom.USER_ORIGIN).is_absolute
        assert not JobOrigin(StartFrom.CURRENT_POSITION).is_absolute

    def test_round_trip(self):
        origin = JobOrigin(StartFrom.CURRENT_POSITION, JobAnchor.CENTER)
        assert JobOrigin.from_dict(origin.to_dict()) == origin

    def test_from_dict_tolerates_missing_and_unknown_values(self):
        assert JobOrigin.from_dict(None) == JobOrigin()
        assert JobOrigin.from_dict({}) == JobOrigin()
        data = {"start_from": "bogus", "anchor": "nowhere"}
        assert JobOrigin.from_dict(data) == JobOrigin()

    def test_nine_anchors(self):
        assert len(JobAnchor) == 9


class TestAnchorPoint:
    @pytest.mark.parametrize(
        "anchor,expected",
        [
            (JobAnchor.BOTTOM_LEFT, (10.0, 20.0)),
            (JobAnchor.BOTTOM, (30.0, 20.0)),
            (JobAnchor.BOTTOM_RIGHT, (50.0, 20.0)),
            (JobAnchor.LEFT, (10.0, 30.0)),
            (JobAnchor.CENTER, (30.0, 30.0)),
            (JobAnchor.RIGHT, (50.0, 30.0)),
            (JobAnchor.TOP_LEFT, (10.0, 40.0)),
            (JobAnchor.TOP, (30.0, 40.0)),
            (JobAnchor.TOP_RIGHT, (50.0, 40.0)),
        ],
    )
    def test_anchor_point_in_world_space(self, anchor, expected):
        assert anchor_point(RECT, anchor) == pytest.approx(expected)

    def test_placement_offset_moves_anchor_onto_start_point(self):
        dx, dy = placement_offset(RECT, JobAnchor.CENTER, (100.0, 200.0))
        assert (dx, dy) == pytest.approx((70.0, 170.0))

    def test_placement_offset_can_be_negative(self):
        dx, dy = placement_offset(RECT, JobAnchor.TOP_RIGHT, (0.0, 0.0))
        assert (dx, dy) == pytest.approx((-50.0, -40.0))


class TestDocJobOrigin:
    def test_new_doc_uses_absolute(self):
        assert Doc().job_origin == JobOrigin()

    def test_doc_serializes_job_origin(self):
        doc = Doc()
        origin = JobOrigin(StartFrom.USER_ORIGIN, JobAnchor.TOP_LEFT)
        doc.set_job_origin(origin)
        restored = Doc.from_dict(doc.to_dict())
        assert restored.job_origin == origin

    def test_legacy_doc_without_job_origin_is_absolute(self):
        data = Doc().to_dict()
        data.pop("job_origin", None)
        assert Doc.from_dict(data).job_origin == JobOrigin()

    def test_set_job_origin_signals_and_invalidates_job(self):
        doc = Doc()
        changed = []
        invalidated = []
        doc.job_origin_changed.connect(
            lambda sender: changed.append(sender), weak=False
        )
        doc.job_assembly_invalidated.connect(
            lambda sender: invalidated.append(sender), weak=False
        )
        doc.set_job_origin(JobOrigin(StartFrom.CURRENT_POSITION))
        assert changed == [doc]
        assert invalidated == [doc]

    def test_setting_same_origin_is_silent(self):
        doc = Doc()
        changed = []
        doc.job_origin_changed.connect(
            lambda sender: changed.append(sender), weak=False
        )
        doc.set_job_origin(JobOrigin())
        assert changed == []
