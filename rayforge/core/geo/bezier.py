"""Pure geometry utilities for cubic Bezier curves.

All public functions operate on cubic Bezier curves defined by four
2D control points: *p0* (start), *c1* (first control), *c2* (second
control), and *p1* (end).  The curve is the parametric polynomial
B(t) for t in [0, 1].
"""

import math
from typing import List, Optional, Set, Tuple, cast

import numpy as np

from .constants import COL_C1X, COL_C1Y, COL_C2X, COL_C2Y, COL_X, COL_Y, COL_Z
from .primitives import midpoint, find_closest_point_on_line_segment
from .types import CubicBezier, Point, Point3D, Polygon, Polygon3D, Rect


def evaluate_bezier(
    p0: Point, c1: Point, c2: Point, p1: Point, t: float
) -> Point:
    """Evaluate a cubic Bezier curve at parameter *t* in [0, 1].

    Args:
        p0: Start point of the curve.
        c1: First control point (affects the tangent leaving p0).
        c2: Second control point (affects the tangent arriving at p1).
        p1: End point of the curve.
        t: Parameter value; 0 returns p0, 1 returns p1.

    Returns:
        The (x, y) point on the curve at *t*.
    """
    complement = 1.0 - t
    x = (
        complement**3 * p0[0]
        + 3 * complement**2 * t * c1[0]
        + 3 * complement * t**2 * c2[0]
        + t**3 * p1[0]
    )
    y = (
        complement**3 * p0[1]
        + 3 * complement**2 * t * c1[1]
        + 3 * complement * t**2 * c2[1]
        + t**3 * p1[1]
    )
    return (x, y)


def subdivide_bezier(
    p0: Point, c1: Point, c2: Point, p1: Point, t: float
) -> Tuple[CubicBezier, CubicBezier]:
    """Split a cubic Bezier at parameter *t* using De Casteljau's algorithm.

    Args:
        p0: Start point of the original curve.
        c1: First control point of the original curve.
        c2: Second control point of the original curve.
        p1: End point of the original curve.
        t: Split position in [0, 1].

    Returns:
        A pair (left, right) of CubicBezier tuples. ``left`` traces the
        original curve from t=0 to *t*, ``right`` from *t* to t=1.
        The two segments share an exact endpoint at the split point.
    """
    mid_p0_c1 = _lerp2(p0, c1, t)
    mid_c1_c2 = _lerp2(c1, c2, t)
    mid_c2_p1 = _lerp2(c2, p1, t)
    mid_p0c1_c1c2 = _lerp2(mid_p0_c1, mid_c1_c2, t)
    mid_c1c2_c2p1 = _lerp2(mid_c1_c2, mid_c2_p1, t)
    split_point = _lerp2(mid_p0c1_c1c2, mid_c1c2_c2p1, t)
    left = (p0, mid_p0_c1, mid_p0c1_c1c2, split_point)
    right = (split_point, mid_c1c2_c2p1, mid_c2_p1, p1)
    return left, right


def bezier_bounds(p0: Point, c1: Point, c2: Point, p1: Point) -> Rect:
    """Compute the tight axis-aligned bounding box of a cubic Bezier.

    Finds where the curve's derivative is zero along each axis and
    evaluates the curve at those extrema, together with the endpoints.

    Args:
        p0: Start point of the curve.
        c1: First control point.
        c2: Second control point.
        p1: End point of the curve.

    Returns:
        A Rect tuple (x_min, y_min, x_max, y_max).
    """
    candidates_x: List[float] = [p0[0], p1[0]]
    candidates_y: List[float] = [p0[1], p1[1]]
    _add_axis_extrema(candidates_x, p0[0], c1[0], c2[0], p1[0])
    _add_axis_extrema(candidates_y, p0[1], c1[1], c2[1], p1[1])
    return (
        min(candidates_x),
        min(candidates_y),
        max(candidates_x),
        max(candidates_y),
    )


def intersect_bezier_rect(
    p0: Point, c1: Point, c2: Point, p1: Point, rect: Rect
) -> List[float]:
    """Find parameter values where the curve crosses or touches a rect.

    For each of the four rect edges the curve's axis coordinate is
    solved for that edge value.  Roots whose other-axis coordinate
    falls within the rect's span are kept.

    Args:
        p0: Start point of the curve.
        c1: First control point.
        c2: Second control point.
        p1: End point of the curve.
        rect: An axis-aligned rectangle (x_min, y_min, x_max, y_max).

    Returns:
        A sorted list of *t* values in [0, 1], always including 0.0
        and 1.0.  Adjacent values bracket regions that are either
        entirely inside or entirely outside the rectangle.
    """
    x_min, y_min, x_max, y_max = rect
    t_crossings: Set[float] = set()
    rect_edges: List[Tuple[int, float]] = [
        (0, x_min),
        (0, x_max),
        (1, y_min),
        (1, y_max),
    ]
    for axis_idx, edge_val in rect_edges:
        poly_a = p0[axis_idx]
        poly_b = 3.0 * (c1[axis_idx] - p0[axis_idx])
        poly_c = 3.0 * (c2[axis_idx] - c1[axis_idx]) - poly_b
        poly_d = p1[axis_idx] - poly_a - poly_b - poly_c
        roots = _solve_cubic(poly_d, poly_c, poly_b, poly_a - edge_val)
        for root in roots:
            if -1e-9 <= root <= 1 + 1e-9:
                clamped = max(0.0, min(1.0, root))
                point_on_curve = evaluate_bezier(p0, c1, c2, p1, clamped)
                other_axis = 1 - axis_idx
                other_coord = point_on_curve[other_axis]
                axis_lo = y_min if other_axis == 1 else x_min
                axis_hi = y_max if other_axis == 1 else x_max
                if axis_lo - 1e-9 <= other_coord <= axis_hi + 1e-9:
                    t_crossings.add(round(clamped, 12))
    t_crossings.update([0.0, 1.0])
    return sorted(t_crossings)


def clip_bezier(
    p0: Point, c1: Point, c2: Point, p1: Point, rect: Rect
) -> List[CubicBezier]:
    """Clip a cubic Bezier to a rectangle.

    Computes the crossing parameters with the rect edges, then extracts
    the sub-segments whose midpoints fall inside the rect.

    Args:
        p0: Start point of the curve.
        c1: First control point.
        c2: Second control point.
        p1: End point of the curve.
        rect: Clipping rectangle (x_min, y_min, x_max, y_max).

    Returns:
        A list of CubicBezier tuples representing the portions of the
        original curve that lie inside the rect.  Empty if the curve is
        entirely outside.
    """
    x_min, y_min, x_max, y_max = rect
    crossing_params = intersect_bezier_rect(p0, c1, c2, p1, rect)
    if len(crossing_params) < 2:
        return []
    inside_segments: List[CubicBezier] = []
    for i in range(len(crossing_params) - 1):
        t_start = crossing_params[i]
        t_end = crossing_params[i + 1]
        if abs(t_end - t_start) < 1e-12:
            continue
        t_mid = (t_start + t_end) / 2.0
        midpoint = evaluate_bezier(p0, c1, c2, p1, t_mid)
        if (
            x_min - 1e-9 <= midpoint[0] <= x_max + 1e-9
            and y_min - 1e-9 <= midpoint[1] <= y_max + 1e-9
        ):
            segment = _extract_subsegment(p0, c1, c2, p1, t_start, t_end)
            inside_segments.append(segment)
    return inside_segments


def bezier_to_quadratic(
    p0: Point, c1: Point, c2: Point, p1: Point
) -> Tuple[Point, Point, Point]:
    """Degree-reduce a cubic Bezier to the best-fit quadratic.

    Uses the least-squares approximation that minimises the L2 error.
    The endpoints are preserved exactly; only the control point changes.

    Args:
        p0: Start point of the cubic (kept as the quadratic start).
        c1: First control point of the cubic.
        c2: Second control point of the cubic.
        p1: End point of the cubic (kept as the quadratic end).

    Returns:
        A tuple (start, control, end) representing the best-fit
        quadratic Bezier.
    """
    quadratic_control = (
        3.0 / 7.0 * c1[0] + 3.0 / 7.0 * c2[0] + 1.0 / 7.0 * p0[0],
        3.0 / 7.0 * c1[1] + 3.0 / 7.0 * c2[1] + 1.0 / 7.0 * p0[1],
    )
    return (p0, quadratic_control, p1)


def find_closest_point_on_bezier(
    bezier_row: np.ndarray,
    start_pos: Point3D,
    x: float,
    y: float,
) -> Optional[Tuple[float, Point, float]]:
    """Find the closest point on a Bezier curve to (x, y).

    Linearizes the curve at fine resolution and checks each segment.

    Args:
        bezier_row: NumPy array row representing the Bezier command.
        start_pos: The (x, y, z) position at the start of this curve.
        x: X coordinate of the query point.
        y: Y coordinate of the query point.

    Returns:
        A tuple (t, point, dist_sq) where *t* is the approximate
        parameter value, *point* is the closest (x, y) on the curve,
        and *dist_sq* is the squared distance to the query point.
        Returns None if the curve has no segments.
    """
    bezier_segments = linearize_bezier_from_array(bezier_row, start_pos, 0.005)
    if not bezier_segments:
        return None

    min_dist_sq = float("inf")
    best_result = None

    for seg_idx, (seg_start, seg_end) in enumerate(bezier_segments):
        t_sub, pt_sub, dist_sq_sub = find_closest_point_on_line_segment(
            seg_start[:2], seg_end[:2], x, y
        )
        if dist_sq_sub < min_dist_sq:
            min_dist_sq = dist_sq_sub
            best_result = (seg_idx, t_sub, pt_sub, dist_sq_sub)

    if not best_result:
        return None

    best_seg_idx, best_t_sub, best_pt, best_dist_sq = best_result
    t_bezier = (best_seg_idx + best_t_sub) / len(bezier_segments)
    return t_bezier, best_pt, best_dist_sq


def linearize_bezier_from_array(
    bezier_row: np.ndarray,
    start_point: Point3D,
    resolution: float = 0.1,
) -> List[Tuple[Point3D, Point3D]]:
    """Linearize a cubic Bezier stored in a geometry array row.

    Uses the NumPy-vectorized fixed-step subdivision for performance
    on large datasets.  For adaptive (tolerance-based) subdivision,
    see :func:`linearize_bezier_segment`.
    """
    p0 = start_point
    p1 = (bezier_row[COL_X], bezier_row[COL_Y], bezier_row[COL_Z])
    c1_2d = (bezier_row[COL_C1X], bezier_row[COL_C1Y])
    c2_2d = (bezier_row[COL_C2X], bezier_row[COL_C2Y])

    z0, z1 = p0[2], p1[2]
    c1 = (c1_2d[0], c1_2d[1], z0 * (2 / 3) + z1 * (1 / 3))
    c2 = (c2_2d[0], c2_2d[1], z0 * (1 / 3) + z1 * (2 / 3))

    l01 = math.dist(p0, c1)
    l12 = math.dist(c1, c2)
    l23 = math.dist(c2, p1)
    estimated_len = l01 + l12 + l23
    num_steps = max(2, int(estimated_len / resolution))

    return cast(
        List[Tuple[Point3D, Point3D]],
        linearize_bezier(p0, c1, c2, p1, num_steps),
    )


def linearize_bezier(
    p0: Tuple[float, ...],
    c1: Tuple[float, ...],
    c2: Tuple[float, ...],
    p1: Tuple[float, ...],
    num_steps: int,
) -> List[Tuple[Tuple[float, ...], Tuple[float, ...]]]:
    """
    Converts a cubic Bézier curve into a list of line segments.
    This function is generic and supports points of any dimension (e.g., 2D
    or 3D).

    Args:
        p0: The starting point of the curve.
        c1: The first control point.
        c2: The second control point.
        p1: The ending point of the curve.
        num_steps: The number of line segments to approximate the curve with.

    Returns:
        A list of tuples, where each tuple is a line segment represented by
        (start_point, end_point).
    """
    if num_steps < 1:
        return []

    points_np = np.array([p0, c1, c2, p1])
    t_values = np.linspace(0, 1, num_steps + 1)

    interpolated_points_np = np.array(
        [
            (1 - t) ** 3 * points_np[0]
            + 3 * (1 - t) ** 2 * t * points_np[1]
            + 3 * (1 - t) * t**2 * points_np[2]
            + t**3 * points_np[3]
            for t in t_values
        ]
    )
    interpolated_points = [tuple(p) for p in interpolated_points_np]

    return [
        (interpolated_points[i], interpolated_points[i + 1])
        for i in range(num_steps)
    ]


def linearize_bezier_adaptive(
    p0: Point,
    c1: Point,
    c2: Point,
    p1: Point,
    tolerance_sq: float,
    max_depth: int = 10,
) -> Polygon:
    """
    Recursively flattens a cubic Bezier curve based on geometric error.

    Args:
        p0, c1, c2, p1: 2D control points (x, y).
        tolerance_sq: The squared maximum allowable distance error.
        max_depth: Maximum recursion depth to prevent infinite loops on
                   singularities/cusps. 10 = max 1024 segments.

    Returns:
        A list of points (excluding p0, including p1) that approximate the
        curve.
    """
    points: Polygon = []

    def recursive_step(
        p0: Point,
        c1: Point,
        c2: Point,
        p1: Point,
        depth: int,
    ):
        vx, vy = p1[0] - p0[0], p1[1] - p0[1]
        norm_sq = vx * vx + vy * vy

        is_flat = False

        if depth >= max_depth:
            is_flat = True
        elif norm_sq < 1e-9:
            d1_sq = (c1[0] - p0[0]) ** 2 + (c1[1] - p0[1]) ** 2
            d2_sq = (c2[0] - p0[0]) ** 2 + (c2[1] - p0[1]) ** 2
            if d1_sq < tolerance_sq and d2_sq < tolerance_sq:
                is_flat = True
        else:
            term1 = -vy
            term2 = vx
            term3 = p0[0] * p1[1] - p0[1] * p1[0]

            cross1 = abs(term1 * c1[0] + term2 * c1[1] - term3)
            cross2 = abs(term1 * c2[0] + term2 * c2[1] - term3)

            limit = tolerance_sq * norm_sq
            if (cross1 * cross1) < limit and (cross2 * cross2) < limit:
                is_flat = True

        if is_flat:
            return

        m01 = ((p0[0] + c1[0]) / 2, (p0[1] + c1[1]) / 2)
        m12 = ((c1[0] + c2[0]) / 2, (c1[1] + c2[1]) / 2)
        m23 = ((c2[0] + p1[0]) / 2, (c2[1] + p1[1]) / 2)

        q01 = ((m01[0] + m12[0]) / 2, (m01[1] + m12[1]) / 2)
        q12 = ((m12[0] + m23[0]) / 2, (m12[1] + m12[1]) / 2)

        r = ((q01[0] + q12[0]) / 2, (q01[1] + q12[1]) / 2)

        recursive_step(p0, m01, q01, r, depth + 1)
        points.append(r)
        recursive_step(r, q12, m23, p1, depth + 1)

    recursive_step(p0, c1, c2, p1, 0)
    points.append(p1)
    return points


def _perp_dist_sq(
    pt: Point3D,
    origin: Point3D,
    vx: float,
    vy: float,
    vz: float,
    norm_sq: float,
) -> float:
    """Squared perpendicular distance from *pt* to the line through
    *origin* with direction (vx, vy, vz)."""
    px = pt[0] - origin[0]
    py = pt[1] - origin[1]
    pz = pt[2] - origin[2]
    cx = py * vz - pz * vy
    cy = pz * vx - px * vz
    cz = px * vy - py * vx
    return (cx * cx + cy * cy + cz * cz) / norm_sq


def _bezier_flatness_sq(
    a: Point3D,
    b: Point3D,
    c: Point3D,
    d: Point3D,
) -> float:
    """Max squared distance of control points *b*, *c* from chord *a*→*d*."""
    vx = d[0] - a[0]
    vy = d[1] - a[1]
    vz = d[2] - a[2]
    norm_sq = vx * vx + vy * vy + vz * vz

    if norm_sq < 1e-9:
        d1 = (b[0] - a[0]) ** 2 + (b[1] - a[1]) ** 2 + (b[2] - a[2]) ** 2
        d2 = (c[0] - a[0]) ** 2 + (c[1] - a[1]) ** 2 + (c[2] - a[2]) ** 2
        return max(d1, d2)

    return max(
        _perp_dist_sq(b, a, vx, vy, vz, norm_sq),
        _perp_dist_sq(c, a, vx, vy, vz, norm_sq),
    )


_BEZIER_SEG_MAX_DEPTH = 10


def flatten_bezier(
    a: Point3D,
    b: Point3D,
    c: Point3D,
    d: Point3D,
    tolerance_sq: float,
    depth: int,
    points: Polygon3D,
) -> None:
    """Recursively flatten a cubic Bezier into polyline points in-place.

    Uses De Casteljau subdivision at the midpoint.  Recurses until the
    curve is flat enough (control-point deviation from the chord is
    within *tolerance_sq*) or *depth* reaches
    ``_BEZIER_SEG_MAX_DEPTH``.

    The caller must seed *points* with the curve's start point.
    This function appends only the endpoint of each accepted flat
    subsegment, so the resulting polyline is contiguous.

    Args:
        a: Start point (x, y, z).
        b: First control point (x, y, z).
        c: Second control point (x, y, z).
        d: End point (x, y, z).
        tolerance_sq: Maximum allowed squared deviation of control
            points from the chord before the segment is accepted as
            flat.
        depth: Current recursion depth (pass 0 at the top level).
        points: Accumulator list that receives Point3D values.
            Modified in-place.
    """
    if (
        depth >= _BEZIER_SEG_MAX_DEPTH
        or _bezier_flatness_sq(a, b, c, d) <= tolerance_sq
    ):
        points.append(d)
        return

    m01 = midpoint(a, b)
    m12 = midpoint(b, c)
    m23 = midpoint(c, d)
    q01 = midpoint(m01, m12)
    q12 = midpoint(m12, m23)
    r = midpoint(q01, q12)

    flatten_bezier(a, m01, q01, r, tolerance_sq, depth + 1, points)
    flatten_bezier(r, q12, m23, d, tolerance_sq, depth + 1, points)


_BEZIER_SEG_DEFAULT_TOLERANCE = 0.1


def linearize_bezier_segment(
    p0: Point3D,
    c1: Point3D,
    c2: Point3D,
    p1: Point3D,
    tolerance: Optional[float] = None,
) -> Polygon3D:
    """
    Linearizes a single cubic Bezier curve into a polyline using
    adaptive subdivision.

    Recursively splits the curve until the maximum deviation of control
    points from the chord is within *tolerance*, then emits polyline
    vertices.  Works with full 3D control points.

    Args:
        p0: Start point (x, y, z).
        c1: First control point (x, y, z).
        c2: Second control point (x, y, z).
        p1: End point (x, y, z).
        tolerance: Maximum allowed deviation from the true curve.
            Defaults to 0.1.

    Returns:
        A list of Point3D vertices forming the polyline.  The first
        element is *p0* and the last is *p1*.
    """
    if tolerance is None:
        tolerance = _BEZIER_SEG_DEFAULT_TOLERANCE
    tolerance_sq = tolerance * tolerance

    points: Polygon3D = [p0]
    flatten_bezier(p0, c1, c2, p1, tolerance_sq, 0, points)
    return points


def _lerp2(a: Point, b: Point, t: float) -> Point:
    """Linearly interpolate between two 2D points.

    Args:
        a: First point.
        b: Second point.
        t: Interpolation factor; 0 returns *a*, 1 returns *b*.
    """
    return (a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t)


def _add_axis_extrema(
    candidates: List[float],
    p0: float,
    c1: float,
    c2: float,
    p1: float,
) -> None:
    """Append axis extrema of a cubic Bezier to *candidates*.

    The extrema occur where the derivative B'(t) is zero.  The
    derivative is a quadratic with coefficients derived from the
    control-point coordinates for a single axis.

    Args:
        candidates: List to append extrema coordinate values to.
        p0: Start coordinate on this axis.
        c1: First control-point coordinate on this axis.
        c2: Second control-point coordinate on this axis.
        p1: End coordinate on this axis.
    """
    coeff_a = -p0 + 3 * c1 - 3 * c2 + p1
    coeff_b = 2 * (p0 - 2 * c1 + c2)
    coeff_c = -p0 + c1
    if abs(coeff_a) < 1e-12:
        if abs(coeff_b) < 1e-12:
            return
        t = -coeff_c / coeff_b
        if 0 < t < 1:
            candidates.append(_eval_axis(p0, c1, c2, p1, t))
        return
    discriminant = coeff_b * coeff_b - 4 * coeff_a * coeff_c
    if discriminant < 0:
        return
    sqrt_disc = math.sqrt(discriminant)
    for sign in (-1, 1):
        t = (-coeff_b + sign * sqrt_disc) / (2 * coeff_a)
        if 0 < t < 1:
            candidates.append(_eval_axis(p0, c1, c2, p1, t))


def _eval_axis(p0: float, c1: float, c2: float, p1: float, t: float) -> float:
    """Evaluate one axis of a cubic Bezier at parameter *t*.

    Args:
        p0: Start coordinate on this axis.
        c1: First control-point coordinate on this axis.
        c2: Second control-point coordinate on this axis.
        p1: End coordinate on this axis.
        t: Parameter value in [0, 1].
    """
    complement = 1.0 - t
    return (
        complement**3 * p0
        + 3 * complement**2 * t * c1
        + 3 * complement * t**2 * c2
        + t**3 * p1
    )


def _extract_subsegment(
    p0: Point,
    c1: Point,
    c2: Point,
    p1: Point,
    t_start: float,
    t_end: float,
) -> CubicBezier:
    """Extract the sub-curve between two parameter values.

    Uses two successive De Casteljau splits: first at *t_start*, then
    at the re-parameterised *t_end* on the resulting right half.

    Args:
        p0: Start point of the original curve.
        c1: First control point of the original curve.
        c2: Second control point of the original curve.
        p1: End point of the original curve.
        t_start: Start parameter of the sub-segment in [0, 1].
        t_end: End parameter of the sub-segment in [0, 1].

    Returns:
        A CubicBezier tuple tracing B(t) for t in [*t_start*, *t_end*].
    """
    starts_at_zero = t_start < 1e-12
    ends_at_one = abs(t_end - 1.0) < 1e-12
    if starts_at_zero and ends_at_one:
        return (p0, c1, c2, p1)
    if starts_at_zero:
        left, _ = subdivide_bezier(p0, c1, c2, p1, t_end)
        return left
    if ends_at_one:
        _, right = subdivide_bezier(p0, c1, c2, p1, t_start)
        return right
    _, right_after_start = subdivide_bezier(p0, c1, c2, p1, t_start)
    reparam_end = (t_end - t_start) / (1.0 - t_start)
    left_of_end, _ = subdivide_bezier(
        right_after_start[0],
        right_after_start[1],
        right_after_start[2],
        right_after_start[3],
        reparam_end,
    )
    return left_of_end


def _solve_cubic(a: float, b: float, c: float, d: float) -> List[float]:
    """Find the real roots of the cubic polynomial *a*t^3 + b*t^2 + c*t + d*.

    Uses the trigonometric method for three real roots and Cardano's
    formula for one real root.  Falls back to quadratic / linear
    solutions when the leading coefficient is near zero.

    Args:
        a: Coefficient of t^3.
        b: Coefficient of t^2.
        c: Coefficient of t^1.
        d: Constant term.

    Returns:
        A list of real roots (1 to 3 elements).
    """
    if abs(a) < 1e-12:
        if abs(b) < 1e-12:
            if abs(c) < 1e-12:
                return []
            return [-d / c]
        discriminant = c * c - 4 * b * d
        if discriminant < 0:
            return []
        sqrt_disc = math.sqrt(discriminant)
        return [(-c + sqrt_disc) / (2 * b), (-c - sqrt_disc) / (2 * b)]
    b /= a
    c /= a
    d /= a
    a = 1.0
    depressed_q = (3.0 * c - b * b) / 9.0
    depressed_r = (9.0 * b * c - 27.0 * d - 2.0 * b * b * b) / 54.0
    discriminant = depressed_q**3 + depressed_r**2
    if discriminant >= 0:
        sqrt_disc = math.sqrt(discriminant)
        cube_root_sum = _cbrt(depressed_r + sqrt_disc)
        cube_root_diff = _cbrt(depressed_r - sqrt_disc)
        real_root = cube_root_sum + cube_root_diff - b / 3.0
        return [real_root]
    neg_q_cubed = -(depressed_q**3) if depressed_q < 0 else 1e-30
    cos_arg = max(-1.0, min(1.0, depressed_r / math.sqrt(neg_q_cubed)))
    theta = math.acos(cos_arg)
    amplitude = 2.0 * math.sqrt(-depressed_q) if depressed_q < 0 else 0.0
    offset = b / 3.0
    roots = [
        amplitude * math.cos(theta / 3.0) - offset,
        amplitude * math.cos((theta + 2.0 * math.pi) / 3.0) - offset,
        amplitude * math.cos((theta + 4.0 * math.pi) / 3.0) - offset,
    ]
    return roots


def _cbrt(x: float) -> float:
    """Real cube root that handles negative arguments.

    Args:
        x: The value whose cube root to compute.

    Returns:
        The real cube root of *x*.
    """
    if x >= 0:
        return x ** (1.0 / 3.0)
    return -((-x) ** (1.0 / 3.0))
