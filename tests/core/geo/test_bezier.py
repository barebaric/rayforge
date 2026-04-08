import math
import pytest

from rayforge.core.geo.bezier import (
    evaluate_bezier,
    subdivide_bezier,
    bezier_bounds,
    intersect_bezier_rect,
    clip_bezier,
    bezier_to_quadratic,
    is_bezier_fully_inside_regions,
    linearize_bezier,
    linearize_bezier_adaptive,
    linearize_bezier_segment,
    _bezier_flatness_sq,
    flatten_bezier,
    _perp_dist_sq,
)


def approx_pt(p, tol=1e-6):
    return pytest.approx(p, abs=tol)


class TestEvaluateBezier:
    def test_endpoints(self):
        p0, c1, c2, p1 = (0, 0), (1, 2), (3, 2), (4, 0)
        assert evaluate_bezier(p0, c1, c2, p1, 0.0) == approx_pt(p0)
        assert evaluate_bezier(p0, c1, c2, p1, 1.0) == approx_pt(p1)

    def test_midpoint(self):
        p0, c1, c2, p1 = (0, 0), (0, 2), (4, 2), (4, 0)
        mid = evaluate_bezier(p0, c1, c2, p1, 0.5)
        expected_x = 0.125 * 0 + 0.375 * 0 + 0.375 * 4 + 0.125 * 4
        expected_y = 0.125 * 0 + 0.375 * 2 + 0.375 * 2 + 0.125 * 0
        assert mid == approx_pt((expected_x, expected_y))

    def test_line(self):
        p0, c1, c2, p1 = (0, 0), (1, 0), (2, 0), (3, 0)
        pt = evaluate_bezier(p0, c1, c2, p1, 0.5)
        assert pt == approx_pt((1.5, 0.0))


class TestSubdivideBezier:
    def test_midpoint_split(self):
        p0, c1, c2, p1 = (0, 0), (1, 3), (3, 3), (4, 0)
        left, right = subdivide_bezier(p0, c1, c2, p1, 0.5)
        assert left[0] == approx_pt(p0)
        assert left[3] == approx_pt(right[0])
        assert right[3] == approx_pt(p1)
        mid_from_left = left[3]
        mid_eval = evaluate_bezier(p0, c1, c2, p1, 0.5)
        assert mid_from_left == approx_pt(mid_eval)

    def test_split_at_zero(self):
        p0, c1, c2, p1 = (0, 0), (1, 2), (3, 2), (4, 0)
        left, right = subdivide_bezier(p0, c1, c2, p1, 0.0)
        assert left[0] == approx_pt(p0)
        assert left[3] == approx_pt(p0)
        assert right[3] == approx_pt(p1)

    def test_split_at_one(self):
        p0, c1, c2, p1 = (0, 0), (1, 2), (3, 2), (4, 0)
        left, right = subdivide_bezier(p0, c1, c2, p1, 1.0)
        assert left[0] == approx_pt(p0)
        assert left[3] == approx_pt(p1)
        assert right[3] == approx_pt(p1)

    def test_split_halves_reconstruct(self):
        p0, c1, c2, p1 = (0, 0), (2, 8), (6, 8), (8, 0)
        left, right = subdivide_bezier(p0, c1, c2, p1, 0.5)
        for t in [0.0, 0.1, 0.25, 0.4, 0.5]:
            orig = evaluate_bezier(p0, c1, c2, p1, t)
            from_left = evaluate_bezier(*left, t * 2)
            assert from_left == approx_pt(orig)
        for t in [0.5, 0.6, 0.75, 0.9, 1.0]:
            orig = evaluate_bezier(p0, c1, c2, p1, t)
            from_right = evaluate_bezier(*right, (t - 0.5) * 2)
            assert from_right == approx_pt(orig)


class TestBezierBounds:
    def test_straight_line(self):
        p0, c1, c2, p1 = (0, 0), (1, 0), (2, 0), (3, 0)
        bounds = bezier_bounds(p0, c1, c2, p1)
        assert bounds == approx_pt((0, 0, 3, 0))

    def test_symmetric_curve(self):
        p0, c1, c2, p1 = (0, 0), (0, 2), (4, 2), (4, 0)
        bounds = bezier_bounds(p0, c1, c2, p1)
        assert bounds[0] == pytest.approx(0.0, abs=1e-6)
        assert bounds[2] == pytest.approx(4.0, abs=1e-6)
        assert bounds[1] == pytest.approx(0.0, abs=1e-6)
        assert bounds[3] >= 1.4

    def test_bounds_contain_all_points(self):
        p0, c1, c2, p1 = (1, 1), (5, 10), (10, -3), (8, 2)
        bounds = bezier_bounds(p0, c1, c2, p1)
        for t_frac in [i / 100.0 for i in range(101)]:
            pt = evaluate_bezier(p0, c1, c2, p1, t_frac)
            assert bounds[0] - 1e-6 <= pt[0] <= bounds[2] + 1e-6
            assert bounds[1] - 1e-6 <= pt[1] <= bounds[3] + 1e-6

    def test_single_point(self):
        p = (2, 3)
        bounds = bezier_bounds(p, p, p, p)
        assert bounds == approx_pt((2, 3, 2, 3))


class TestIntersectBezierRect:
    def test_fully_inside(self):
        p0, c1, c2, p1 = (2, 2), (3, 4), (5, 4), (6, 2)
        rect = (0, 0, 10, 10)
        params = intersect_bezier_rect(p0, c1, c2, p1, rect)
        assert 0.0 in params
        assert 1.0 in params

    def test_fully_outside(self):
        p0, c1, c2, p1 = (12, 12), (13, 14), (15, 14), (16, 12)
        rect = (0, 0, 10, 10)
        params = intersect_bezier_rect(p0, c1, c2, p1, rect)
        assert params == [0.0, 1.0]

    def test_crosses_one_edge(self):
        p0, c1, c2, p1 = (-5, 5), (2, 8), (8, 8), (15, 5)
        rect = (0, 0, 10, 10)
        params = intersect_bezier_rect(p0, c1, c2, p1, rect)
        assert len(params) >= 2
        crossings = [t for t in params if 0 < t < 1]
        assert len(crossings) >= 2

    def test_crosses_two_edges(self):
        p0, c1, c2, p1 = (-2, 5), (3, 12), (7, 12), (12, 5)
        rect = (0, 0, 10, 10)
        params = intersect_bezier_rect(p0, c1, c2, p1, rect)
        crossings = [t for t in params if 0 < t < 1]
        assert len(crossings) >= 2


class TestClipBezier:
    def test_fully_inside(self):
        p0, c1, c2, p1 = (2, 2), (3, 5), (7, 5), (8, 2)
        rect = (0, 0, 10, 10)
        segments = clip_bezier(p0, c1, c2, p1, rect)
        assert len(segments) == 1
        start = evaluate_bezier(*segments[0], 0.0)
        end = evaluate_bezier(*segments[0], 1.0)
        assert start == approx_pt(p0)
        assert end == approx_pt(p1)

    def test_fully_outside(self):
        p0, c1, c2, p1 = (12, 12), (13, 14), (15, 14), (16, 12)
        rect = (0, 0, 10, 10)
        segments = clip_bezier(p0, c1, c2, p1, rect)
        assert len(segments) == 0

    def test_crosses_left_edge(self):
        p0, c1, c2, p1 = (-5, 5), (2, 8), (8, 8), (5, 5)
        rect = (0, 0, 10, 10)
        segments = clip_bezier(p0, c1, c2, p1, rect)
        assert len(segments) >= 1
        for seg in segments:
            start = seg[0]
            assert start[0] >= -1e-6

    def test_crosses_right_edge(self):
        p0, c1, c2, p1 = (5, 5), (8, 8), (12, 8), (15, 5)
        rect = (0, 0, 10, 10)
        segments = clip_bezier(p0, c1, c2, p1, rect)
        assert len(segments) >= 1
        for seg in segments:
            end = evaluate_bezier(*seg, 1.0)
            assert end[0] <= 10 + 1e-6

    def test_crosses_two_opposite_edges(self):
        p0, c1, c2, p1 = (-2, 5), (3, 8), (7, 8), (12, 5)
        rect = (0, 0, 10, 10)
        segments = clip_bezier(p0, c1, c2, p1, rect)
        assert len(segments) >= 1
        for seg in segments:
            start = evaluate_bezier(*seg, 0.0)
            end = evaluate_bezier(*seg, 1.0)
            assert start[0] >= -1e-6
            assert end[0] >= -1e-6
            assert start[0] <= 10 + 1e-6
            assert end[0] <= 10 + 1e-6

    def test_clipped_segments_stay_inside(self):
        p0, c1, c2, p1 = (-3, 5), (2, 10), (8, 10), (13, 5)
        rect = (0, 0, 10, 10)
        segments = clip_bezier(p0, c1, c2, p1, rect)
        assert len(segments) >= 1
        for seg in segments:
            for i in range(20):
                t = i / 19.0
                pt = evaluate_bezier(seg[0], seg[1], seg[2], seg[3], t)
                assert pt[0] >= -2, f"x={pt[0]} far outside rect at t={t}"
                assert pt[0] <= 12, f"x={pt[0]} far outside rect at t={t}"
                assert pt[1] >= -2, f"y={pt[1]} far outside rect at t={t}"
                assert pt[1] <= 12, f"y={pt[1]} far outside rect at t={t}"

    def test_degenerate_point(self):
        p = (5, 5)
        segments = clip_bezier(p, p, p, p, (0, 0, 10, 10))
        assert len(segments) == 1

    def test_start_inside_end_outside(self):
        p0, c1, c2, p1 = (5, 5), (8, 7), (12, 7), (15, 5)
        rect = (0, 0, 10, 10)
        segments = clip_bezier(p0, c1, c2, p1, rect)
        assert len(segments) >= 1
        start = evaluate_bezier(*segments[0], 0.0)
        assert start == approx_pt(p0)

    def test_start_outside_end_inside(self):
        p0, c1, c2, p1 = (-5, 5), (0, 7), (5, 7), (8, 5)
        rect = (0, 0, 10, 10)
        segments = clip_bezier(p0, c1, c2, p1, rect)
        assert len(segments) >= 1
        end = evaluate_bezier(*segments[-1], 1.0)
        assert end == approx_pt(p1)


class TestBezierToQuadratic:
    def test_endpoints_preserved(self):
        p0, c1, c2, p1 = (0, 0), (2, 6), (6, 6), (8, 0)
        qp0, qc, qp1 = bezier_to_quadratic(p0, c1, c2, p1)
        assert qp0 == approx_pt(p0)
        assert qp1 == approx_pt(p1)

    def test_midpoint_approximation(self):
        p0, c1, c2, p1 = (0, 0), (0, 3), (3, 3), (3, 0)
        qp0, qc, qp1 = bezier_to_quadratic(p0, c1, c2, p1)
        quad_mid = evaluate_bezier(qp0, qc, qc, qp1, 0.5)
        cubic_mid = evaluate_bezier(p0, c1, c2, p1, 0.5)
        err = math.hypot(
            quad_mid[0] - cubic_mid[0], quad_mid[1] - cubic_mid[1]
        )
        assert err < 0.5

    def test_linear_case(self):
        p0, c1, c2, p1 = (0, 0), (1, 0), (2, 0), (3, 0)
        qp0, qc, qp1 = bezier_to_quadratic(p0, c1, c2, p1)
        assert qp0 == approx_pt((0, 0))
        assert qp1 == approx_pt((3, 0))
        assert qc[1] == pytest.approx(0.0, abs=1e-6)


def test_linearize_bezier_3d():
    """Tests linearization of a 3D Bézier curve."""
    p0 = (0.0, 0.0, 0.0)
    c1 = (1.0, 1.0, 5.0)
    c2 = (2.0, 1.0, 5.0)
    p1 = (3.0, 0.0, 10.0)
    num_steps = 10

    segments = linearize_bezier(p0, c1, c2, p1, num_steps)
    assert len(segments) == num_steps

    start_of_chain = segments[0][0]
    end_of_chain = segments[-1][1]

    assert start_of_chain == pytest.approx(p0)
    assert end_of_chain == pytest.approx(p1)

    midpoint = segments[5][0]
    expected_x = 0.125 * 0 + 0.375 * 1 + 0.375 * 2 + 0.125 * 3
    expected_y = 0.125 * 0 + 0.375 * 1 + 0.375 * 1 + 0.125 * 0
    expected_z = 0.125 * 0 + 0.375 * 5 + 0.375 * 5 + 0.125 * 10
    assert midpoint == pytest.approx((expected_x, expected_y, expected_z))


def test_linearize_bezier_adaptive_flat():
    """Tests adaptive linearization on a perfectly flat line."""
    p0 = (0.0, 0.0)
    c1 = (2.5, 0.0)
    c2 = (7.5, 0.0)
    p1 = (10.0, 0.0)

    points = linearize_bezier_adaptive(p0, c1, c2, p1, tolerance_sq=0.01)

    assert len(points) == 1
    assert points[0] == p1


def test_linearize_bezier_adaptive_curved():
    """Tests adaptive linearization on a curve."""
    p0 = (0.0, 0.0)
    c1 = (0.0, 10.0)
    c2 = (10.0, 10.0)
    p1 = (10.0, 0.0)

    points_coarse = linearize_bezier_adaptive(p0, c1, c2, p1, tolerance_sq=1.0)
    points_fine = linearize_bezier_adaptive(
        p0, c1, c2, p1, tolerance_sq=0.0001
    )

    assert len(points_fine) > len(points_coarse)
    assert points_fine[-1] == p1
    assert points_coarse[-1] == p1


def test_linearize_segment_start_end():
    """Polyline starts at p0 and ends at p1."""
    p0 = (0.0, 0.0, 0.0)
    c1 = (1.0, 3.0, 2.0)
    c2 = (3.0, 3.0, 4.0)
    p1 = (4.0, 0.0, 6.0)

    pts = linearize_bezier_segment(p0, c1, c2, p1)

    assert len(pts) >= 2
    assert pts[0] == pytest.approx(p0)
    assert pts[-1] == pytest.approx(p1)


def test_linearize_segment_flat_curve():
    """Collinear control points produce exactly two points (start, end)."""
    p0 = (0.0, 0.0, 0.0)
    c1 = (3.0, 0.0, 3.0)
    c2 = (7.0, 0.0, 7.0)
    p1 = (10.0, 0.0, 10.0)

    pts = linearize_bezier_segment(p0, c1, c2, p1, tolerance=0.1)

    assert len(pts) == 2
    assert pts[0] == pytest.approx(p0)
    assert pts[1] == pytest.approx(p1)


def test_linearize_segment_tolerance_controls_detail():
    """Tighter tolerance produces more points."""
    p0 = (0.0, 0.0, 0.0)
    c1 = (0.0, 10.0, 5.0)
    c2 = (10.0, 10.0, 5.0)
    p1 = (10.0, 0.0, 10.0)

    coarse = linearize_bezier_segment(p0, c1, c2, p1, tolerance=1.0)
    fine = linearize_bezier_segment(p0, c1, c2, p1, tolerance=0.001)

    assert len(fine) > len(coarse)


def test_linearize_segment_midpoint_accuracy():
    """Midpoint of the polyline matches the analytic B(0.5)."""
    p0 = (0.0, 0.0, 0.0)
    c1 = (1.0, 1.0, 5.0)
    c2 = (2.0, 1.0, 5.0)
    p1 = (3.0, 0.0, 10.0)

    pts = linearize_bezier_segment(p0, c1, c2, p1, tolerance=0.001)

    mid_idx = len(pts) // 2
    actual_mid = pts[mid_idx]

    expected_x = 0.125 * 0 + 0.375 * 1 + 0.375 * 2 + 0.125 * 3
    expected_y = 0.125 * 0 + 0.375 * 1 + 0.375 * 1 + 0.125 * 0
    expected_z = 0.125 * 0 + 0.375 * 5 + 0.375 * 5 + 0.125 * 10
    assert actual_mid == pytest.approx(
        (expected_x, expected_y, expected_z), abs=0.01
    )


def test_linearize_segment_default_tolerance():
    """Calling without tolerance uses the default (no crash)."""
    p0 = (0.0, 0.0, 0.0)
    c1 = (5.0, 10.0, 3.0)
    c2 = (10.0, 10.0, 6.0)
    p1 = (15.0, 0.0, 9.0)

    pts = linearize_bezier_segment(p0, c1, c2, p1)

    assert len(pts) >= 2
    assert pts[0] == pytest.approx(p0)
    assert pts[-1] == pytest.approx(p1)


def test_linearize_segment_zero_length():
    """Degenerate bezier where all points coincide returns two points."""
    p = (1.0, 2.0, 3.0)
    pts = linearize_bezier_segment(p, p, p, p)

    assert len(pts) == 2
    assert pts[0] == pytest.approx(p)
    assert pts[1] == pytest.approx(p)


def test_perp_dist_sq_on_line():
    origin = (0.0, 0.0, 0.0)
    pt = (5.0, 0.0, 0.0)
    assert _perp_dist_sq(pt, origin, 1.0, 0.0, 0.0, 1.0) == pytest.approx(0.0)


def test_perp_dist_sq_off_line():
    origin = (0.0, 0.0, 0.0)
    pt = (0.0, 3.0, 0.0)
    assert _perp_dist_sq(pt, origin, 1.0, 0.0, 0.0, 1.0) == pytest.approx(9.0)


def test_perp_dist_sq_3d():
    origin = (0.0, 0.0, 0.0)
    pt = (0.0, 0.0, 4.0)
    assert _perp_dist_sq(pt, origin, 1.0, 0.0, 0.0, 1.0) == pytest.approx(16.0)


def test_bezier_flatness_sq_collinear():
    a = (0.0, 0.0, 0.0)
    b = (3.0, 0.0, 0.0)
    c = (7.0, 0.0, 0.0)
    d = (10.0, 0.0, 0.0)
    assert _bezier_flatness_sq(a, b, c, d) == pytest.approx(0.0, abs=1e-9)


def test_bezier_flatness_sq_nonzero():
    a = (0.0, 0.0, 0.0)
    b = (0.0, 5.0, 0.0)
    c = (10.0, 5.0, 0.0)
    d = (10.0, 0.0, 0.0)
    assert _bezier_flatness_sq(a, b, c, d) > 0.0


def test_bezier_flatness_sq_coincident_endpoints():
    a = (1.0, 1.0, 1.0)
    d = (1.0, 1.0, 1.0)
    b = (1.0, 1.0, 1.0)
    c = (1.0, 1.0, 1.0)
    assert _bezier_flatness_sq(a, b, c, d) == pytest.approx(0.0)


def test_bezier_flatness_sq_coincident_endpoints_offset_ctrl():
    a = (0.0, 0.0, 0.0)
    d = (0.0, 0.0, 0.0)
    b = (1.0, 0.0, 0.0)
    c = (0.0, 2.0, 0.0)
    result = _bezier_flatness_sq(a, b, c, d)
    assert result == pytest.approx(4.0)


def test_flatten_bezier_produces_points():
    a = (0.0, 0.0, 0.0)
    b = (0.0, 10.0, 0.0)
    c = (10.0, 10.0, 0.0)
    d = (10.0, 0.0, 0.0)
    pts = [a]
    flatten_bezier(a, b, c, d, 0.01, 0, pts)
    assert len(pts) >= 3
    assert pts[0] == a
    assert pts[-1] == d


def test_flatten_bezier_flat_emits_endpoint():
    a = (0.0, 0.0, 0.0)
    b = (3.0, 0.0, 0.0)
    c = (7.0, 0.0, 0.0)
    d = (10.0, 0.0, 0.0)
    pts = [a]
    flatten_bezier(a, b, c, d, 0.1, 0, pts)
    assert len(pts) == 2
    assert pts[0] == a
    assert pts[1] == d


class TestIsBezierFullyInsideRegions:
    def _square(self, x0, y0, size):
        return [
            (x0, y0),
            (x0 + size, y0),
            (x0 + size, y0 + size),
            (x0, y0 + size),
        ]

    def test_fully_inside_single_region(self):
        region = self._square(0, 0, 10)
        p0, c1, c2, p1 = (2, 2), (3, 4), (5, 4), (6, 2)
        assert is_bezier_fully_inside_regions(p0, c1, c2, p1, [region])

    def test_fully_outside(self):
        region = self._square(0, 0, 10)
        p0, c1, c2, p1 = (12, 12), (13, 14), (15, 14), (16, 12)
        assert not is_bezier_fully_inside_regions(p0, c1, c2, p1, [region])

    def test_bbox_corner_outside(self):
        region = self._square(0, 0, 5)
        p0, c1, c2, p1 = (2, 2), (3, 4), (5, 4), (6, 2)
        assert not is_bezier_fully_inside_regions(p0, c1, c2, p1, [region])

    def test_inside_one_of_multiple_regions(self):
        r1 = self._square(0, 0, 5)
        r2 = self._square(10, 0, 5)
        p0, c1, c2, p1 = (11, 1), (12, 3), (13, 3), (14, 1)
        assert is_bezier_fully_inside_regions(p0, c1, c2, p1, [r1, r2])

    def test_spread_across_regions(self):
        r1 = self._square(0, 0, 5)
        r2 = self._square(10, 0, 5)
        p0, c1, c2, p1 = (2, 2), (5, 4), (8, 4), (12, 2)
        assert not is_bezier_fully_inside_regions(p0, c1, c2, p1, [r1, r2])

    def test_degenerate_point_inside(self):
        region = self._square(0, 0, 10)
        p = (5, 5)
        assert is_bezier_fully_inside_regions(p, p, p, p, [region])

    def test_degenerate_point_outside(self):
        region = self._square(0, 0, 10)
        p = (15, 15)
        assert not is_bezier_fully_inside_regions(p, p, p, p, [region])

    def test_empty_regions(self):
        p0, c1, c2, p1 = (2, 2), (3, 4), (5, 4), (6, 2)
        assert not is_bezier_fully_inside_regions(p0, c1, c2, p1, [])

    def test_midpoint_outside(self):
        region = self._square(0, 0, 5)
        p0, c1, c2, p1 = (2, 1), (3, 8), (4, 8), (3, 1)
        assert not is_bezier_fully_inside_regions(p0, c1, c2, p1, [region])

    def test_linear_segment_inside(self):
        region = self._square(0, 0, 10)
        p0, c1, c2, p1 = (1, 5), (3, 5), (5, 5), (7, 5)
        assert is_bezier_fully_inside_regions(p0, c1, c2, p1, [region])
