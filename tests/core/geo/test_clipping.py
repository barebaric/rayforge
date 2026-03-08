import math
import pytest
from rayforge.core.geo.clipping import (
    clip_line_segment,
    subtract_regions_from_line_segment,
    clip_line_segment_to_regions,
)


def create_circle_polygon(cx, cy, radius, num_segments=32):
    points = []
    for i in range(num_segments):
        angle = 2 * math.pi * i / num_segments
        x = cx + radius * math.cos(angle)
        y = cy + radius * math.sin(angle)
        points.append((x, y))
    return points


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


def test_clip_line_segment_to_regions_basic():
    p1 = (0.0, 50.0, -5.0)
    p2 = (100.0, 50.0, 5.0)
    region = [(40.0, 45.0), (60.0, 45.0), (60.0, 55.0), (40.0, 55.0)]

    kept_segments = clip_line_segment_to_regions(p1, p2, [region])

    assert len(kept_segments) == 1

    s1_p1, s1_p2 = kept_segments[0]
    assert s1_p1 == pytest.approx((40.0, 50.0, -1.0))
    assert s1_p2 == pytest.approx((60.0, 50.0, 1.0))


def test_clip_line_segment_to_regions_fully_outside():
    p1 = (0.0, 50.0, 0.0)
    p2 = (30.0, 50.0, 0.0)
    region = [(40.0, 45.0), (60.0, 45.0), (60.0, 55.0), (40.0, 55.0)]
    kept_segments = clip_line_segment_to_regions(p1, p2, [region])
    assert len(kept_segments) == 0


def test_clip_line_segment_to_regions_fully_inside():
    p1 = (45.0, 50.0, 0.0)
    p2 = (55.0, 50.0, 0.0)
    region = [(40.0, 45.0), (60.0, 45.0), (60.0, 55.0), (40.0, 55.0)]
    kept_segments = clip_line_segment_to_regions(p1, p2, [region])
    assert len(kept_segments) == 1
    s1_p1, s1_p2 = kept_segments[0]
    assert s1_p1 == pytest.approx((45.0, 50.0, 0.0))
    assert s1_p2 == pytest.approx((55.0, 50.0, 0.0))


def test_clip_line_segment_to_regions_multiple_regions():
    p1 = (0.0, 50.0, 0.0)
    p2 = (100.0, 50.0, 0.0)
    region1 = [(20.0, 45.0), (30.0, 45.0), (30.0, 55.0), (20.0, 55.0)]
    region2 = [(70.0, 45.0), (80.0, 45.0), (80.0, 55.0), (70.0, 55.0)]

    kept_segments = clip_line_segment_to_regions(p1, p2, [region1, region2])

    assert len(kept_segments) == 2

    s1_p1, s1_p2 = kept_segments[0]
    assert s1_p1 == pytest.approx((20.0, 50.0, 0.0))
    assert s1_p2 == pytest.approx((30.0, 50.0, 0.0))

    s2_p1, s2_p2 = kept_segments[1]
    assert s2_p1 == pytest.approx((70.0, 50.0, 0.0))
    assert s2_p2 == pytest.approx((80.0, 50.0, 0.0))


def test_clip_line_segment_to_regions_empty_regions():
    p1 = (0.0, 50.0, 0.0)
    p2 = (100.0, 50.0, 0.0)
    kept_segments = clip_line_segment_to_regions(p1, p2, [])
    assert len(kept_segments) == 0


class TestClipToCircleRegions:
    """Tests for clipping line segments to circular regions."""

    def test_clip_to_circle_basic(self):
        p1 = (0.0, 50.0, -5.0)
        p2 = (100.0, 50.0, 5.0)
        circle = create_circle_polygon(50, 50, 20)

        kept_segments = clip_line_segment_to_regions(p1, p2, [circle])

        assert len(kept_segments) == 1
        s1_p1, s1_p2 = kept_segments[0]
        assert s1_p1[0] == pytest.approx(30.0, abs=0.5)
        assert s1_p1[1] == pytest.approx(50.0)
        assert s1_p1[2] == pytest.approx(-2.0, abs=0.1)
        assert s1_p2[0] == pytest.approx(70.0, abs=0.5)
        assert s1_p2[1] == pytest.approx(50.0)
        assert s1_p2[2] == pytest.approx(2.0, abs=0.1)

    def test_clip_to_circle_fully_outside(self):
        p1 = (0.0, 50.0, 0.0)
        p2 = (25.0, 50.0, 0.0)
        circle = create_circle_polygon(50, 50, 20)

        kept_segments = clip_line_segment_to_regions(p1, p2, [circle])
        assert len(kept_segments) == 0

    def test_clip_to_circle_fully_inside(self):
        p1 = (45.0, 50.0, 0.0)
        p2 = (55.0, 50.0, 0.0)
        circle = create_circle_polygon(50, 50, 20)

        kept_segments = clip_line_segment_to_regions(p1, p2, [circle])

        assert len(kept_segments) == 1
        s1_p1, s1_p2 = kept_segments[0]
        assert s1_p1 == pytest.approx((45.0, 50.0, 0.0))
        assert s1_p2 == pytest.approx((55.0, 50.0, 0.0))

    def test_clip_to_circle_vertical_line(self):
        p1 = (50.0, 0.0, 0.0)
        p2 = (50.0, 100.0, 0.0)
        circle = create_circle_polygon(50, 50, 20)

        kept_segments = clip_line_segment_to_regions(p1, p2, [circle])

        assert len(kept_segments) == 1
        s1_p1, s1_p2 = kept_segments[0]
        assert s1_p1[1] == pytest.approx(30.0, abs=0.5)
        assert s1_p2[1] == pytest.approx(70.0, abs=0.5)

    def test_clip_to_circle_diagonal_line(self):
        p1 = (0.0, 0.0, 0.0)
        p2 = (100.0, 100.0, 0.0)
        circle = create_circle_polygon(50, 50, 20)

        kept_segments = clip_line_segment_to_regions(p1, p2, [circle])

        assert len(kept_segments) == 1
        s1_p1, s1_p2 = kept_segments[0]
        dist_start = math.hypot(s1_p1[0] - 50, s1_p1[1] - 50)
        dist_end = math.hypot(s1_p2[0] - 50, s1_p2[1] - 50)
        assert dist_start == pytest.approx(20.0, abs=0.5)
        assert dist_end == pytest.approx(20.0, abs=0.5)

    def test_clip_to_multiple_circles(self):
        p1 = (0.0, 50.0, 0.0)
        p2 = (100.0, 50.0, 0.0)
        circle1 = create_circle_polygon(25, 50, 10)
        circle2 = create_circle_polygon(75, 50, 10)

        kept_segments = clip_line_segment_to_regions(
            p1, p2, [circle1, circle2]
        )

        assert len(kept_segments) == 2
        s1_p1, s1_p2 = kept_segments[0]
        assert s1_p1[0] == pytest.approx(15.0, abs=0.5)
        assert s1_p2[0] == pytest.approx(35.0, abs=0.5)
        s2_p1, s2_p2 = kept_segments[1]
        assert s2_p1[0] == pytest.approx(65.0, abs=0.5)
        assert s2_p2[0] == pytest.approx(85.0, abs=0.5)

    def test_clip_to_circle_tangent_line(self):
        p1 = (0.0, 70.0, 0.0)
        p2 = (100.0, 70.0, 0.0)
        circle = create_circle_polygon(50, 50, 20)

        kept_segments = clip_line_segment_to_regions(p1, p2, [circle])
        assert len(kept_segments) == 0

    def test_clip_to_circle_with_z_interpolation(self):
        p1 = (30.0, 50.0, -10.0)
        p2 = (70.0, 50.0, 10.0)
        circle = create_circle_polygon(50, 50, 20)

        kept_segments = clip_line_segment_to_regions(p1, p2, [circle])

        assert len(kept_segments) == 1
        s1_p1, s1_p2 = kept_segments[0]
        assert s1_p1[2] == pytest.approx(-10.0, abs=0.5)
        assert s1_p2[2] == pytest.approx(10.0, abs=0.5)


class TestSubtractCircleRegions:
    """Tests for subtracting circular regions from line segments."""

    def test_subtract_circle_from_line(self):
        p1 = (0.0, 50.0, -5.0)
        p2 = (100.0, 50.0, 5.0)
        circle = create_circle_polygon(50, 50, 20)

        kept_segments = subtract_regions_from_line_segment(p1, p2, [circle])

        assert len(kept_segments) == 2
        s1_p1, s1_p2 = kept_segments[0]
        assert s1_p1[0] == pytest.approx(0.0)
        assert s1_p2[0] == pytest.approx(30.0, abs=0.5)
        s2_p1, s2_p2 = kept_segments[1]
        assert s2_p1[0] == pytest.approx(70.0, abs=0.5)
        assert s2_p2[0] == pytest.approx(100.0)

    def test_subtract_circle_fully_contained(self):
        p1 = (45.0, 50.0, 0.0)
        p2 = (55.0, 50.0, 0.0)
        circle = create_circle_polygon(50, 50, 20)

        kept_segments = subtract_regions_from_line_segment(p1, p2, [circle])
        assert len(kept_segments) == 0

    def test_subtract_circle_starts_inside(self):
        p1 = (45.0, 50.0, 0.0)
        p2 = (80.0, 50.0, 0.0)
        circle = create_circle_polygon(50, 50, 20)

        kept_segments = subtract_regions_from_line_segment(p1, p2, [circle])

        assert len(kept_segments) == 1
        s1_p1, s1_p2 = kept_segments[0]
        assert s1_p1[0] == pytest.approx(70.0, abs=0.5)
        assert s1_p2[0] == pytest.approx(80.0)
