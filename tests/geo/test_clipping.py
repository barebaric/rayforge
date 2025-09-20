import pytest
from rayforge.core.geo.clipping import (
    clip_line_segment,
    subtract_regions_from_line_segment,
)


@pytest.fixture
def clip_rect():
    return (0.0, 0.0, 100.0, 100.0)


def test_clip_line_segment_fully_inside(clip_rect):
    p1 = (10.0, 10.0, -1.0)
    p2 = (90.0, 90.0, -1.0)
    result = clip_line_segment(p1, p2, clip_rect)
    assert result is not None
    res_p1, res_p2 = result
    assert res_p1 == pytest.approx(p1)
    assert res_p2 == pytest.approx(p2)


def test_clip_line_segment_fully_outside(clip_rect):
    p1 = (110.0, 110.0, 0.0)
    p2 = (120.0, 120.0, 0.0)
    result = clip_line_segment(p1, p2, clip_rect)
    assert result is None


def test_clip_line_segment_crossing_one_boundary(clip_rect):
    p1 = (50.0, 50.0, 0.0)
    p2 = (150.0, 50.0, 0.0)
    result = clip_line_segment(p1, p2, clip_rect)
    assert result is not None
    res_p1, res_p2 = result
    assert res_p1 == pytest.approx(p1)
    assert res_p2 == pytest.approx((100.0, 50.0, 0.0))


def test_clip_line_segment_crossing_two_boundaries(clip_rect):
    p1 = (-50.0, 50.0, 0.0)
    p2 = (150.0, 50.0, 0.0)
    result = clip_line_segment(p1, p2, clip_rect)
    assert result is not None
    res_p1, res_p2 = result
    assert res_p1 == pytest.approx((0.0, 50.0, 0.0))
    assert res_p2 == pytest.approx((100.0, 50.0, 0.0))


def test_clip_line_segment_interpolates_z(clip_rect):
    p1 = (50.0, -50.0, -10.0)
    p2 = (50.0, 150.0, 10.0)
    result = clip_line_segment(p1, p2, clip_rect)
    assert result is not None
    res_p1, res_p2 = result
    assert res_p1 == pytest.approx((50.0, 0.0, -5.0))  # Z is halfway
    assert res_p2 == pytest.approx((50.0, 100.0, 5.0))  # Z is 3/4 of the way


def test_subtract_regions_from_line_segment():
    # A simple gap in the middle of a line
    p1 = (0.0, 50.0, -5.0)
    p2 = (100.0, 50.0, 5.0)
    region = [(40.0, 45.0), (60.0, 45.0), (60.0, 55.0), (40.0, 55.0)]

    kept_segments = subtract_regions_from_line_segment(p1, p2, [region])

    assert len(kept_segments) == 2

    # First segment
    s1_p1, s1_p2 = kept_segments[0]
    assert s1_p1 == pytest.approx((0.0, 50.0, -5.0))
    assert s1_p2 == pytest.approx((40.0, 50.0, -1.0))  # Z is interpolated

    # Second segment
    s2_p1, s2_p2 = kept_segments[1]
    assert s2_p1 == pytest.approx((60.0, 50.0, 1.0))  # Z is interpolated
    assert s2_p2 == pytest.approx((100.0, 50.0, 5.0))


def test_subtract_regions_fully_contained():
    p1 = (45.0, 50.0, 0.0)
    p2 = (55.0, 50.0, 0.0)
    region = [(40.0, 45.0), (60.0, 45.0), (60.0, 55.0), (40.0, 55.0)]
    kept_segments = subtract_regions_from_line_segment(p1, p2, [region])
    assert len(kept_segments) == 0


def test_subtract_regions_starts_inside():
    p1 = (45.0, 50.0, 0.0)
    p2 = (70.0, 50.0, 0.0)
    region = [(40.0, 45.0), (60.0, 45.0), (60.0, 55.0), (40.0, 55.0)]
    kept_segments = subtract_regions_from_line_segment(p1, p2, [region])
    assert len(kept_segments) == 1
    s1_p1, s1_p2 = kept_segments[0]
    assert s1_p1 == pytest.approx((60.0, 50.0, 0.0))
    assert s1_p2 == pytest.approx((70.0, 50.0, 0.0))
