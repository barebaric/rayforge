//! Bezier: Cubic Bezier curve operations.
//!
//! This module provides functions for working with cubic Bezier curves including:
//! - Point evaluation at parameter t
//! - Curve subdivision
//! - Bounding box computation
//! - Intersection with rectangles
//! - Clipping to rectangular regions
//! - Linearization into line segments
//! - Conversion to quadratic approximation

use std::f64::consts::PI;

use crate::constants::*;
use crate::shape::line::get_line_segment_closest_point;
use crate::shape::point::midpoint;
use crate::shape::polygon::is_point_inside_polygon;
use crate::types::{CubicBezier, Point, Point3D, Polygon, Polygon3D, Rect};

/// Evaluates a cubic Bezier curve at parameter t [0, 1] using the Bernstein polynomial.
pub fn get_bezier_point_at(
    p0: Point,
    c1: Point,
    c2: Point,
    p1: Point,
    t: f64,
) -> Point {
    let complement = 1.0 - t;
    let x = complement.powi(3) * p0.0
        + 3.0 * complement.powi(2) * t * c1.0
        + 3.0 * complement * t.powi(2) * c2.0
        + t.powi(3) * p1.0;
    let y = complement.powi(3) * p0.1
        + 3.0 * complement.powi(2) * t * c1.1
        + 3.0 * complement * t.powi(2) * c2.1
        + t.powi(3) * p1.1;
    (x, y)
}

/// Splits a Bezier curve into two sub-curves at parameter t.
/// Uses de Casteljau's algorithm for subdivision.
pub fn split_bezier(
    p0: Point,
    c1: Point,
    c2: Point,
    p1: Point,
    t: f64,
) -> (CubicBezier, CubicBezier) {
    let mid_p0_c1 = _lerp2(p0, c1, t);
    let mid_c1_c2 = _lerp2(c1, c2, t);
    let mid_c2_p1 = _lerp2(c2, p1, t);
    let mid_p0c1_c1c2 = _lerp2(mid_p0_c1, mid_c1_c2, t);
    let mid_c1c2_c2p1 = _lerp2(mid_c1_c2, mid_c2_p1, t);
    let split_point = _lerp2(mid_p0c1_c1c2, mid_c1c2_c2p1, t);

    let left: CubicBezier = (p0, mid_p0_c1, mid_p0c1_c1c2, split_point);
    let right: CubicBezier = (split_point, mid_c1c2_c2p1, mid_c2_p1, p1);
    (left, right)
}

/// Computes the axis-aligned bounding box of a Bezier curve.
/// Finds extrema analytically by solving for points where the derivative is zero.
pub fn get_bezier_bounds(p0: Point, c1: Point, c2: Point, p1: Point) -> Rect {
    let mut candidates_x = vec![p0.0, p1.0];
    let mut candidates_y = vec![p0.1, p1.1];
    _add_axis_extrema(&mut candidates_x, p0.0, c1.0, c2.0, p1.0);
    _add_axis_extrema(&mut candidates_y, p0.1, c1.1, c2.1, p1.1);

    (
        candidates_x.iter().cloned().fold(f64::INFINITY, f64::min),
        candidates_y.iter().cloned().fold(f64::INFINITY, f64::min),
        candidates_x
            .iter()
            .cloned()
            .fold(f64::NEG_INFINITY, f64::max),
        candidates_y
            .iter()
            .cloned()
            .fold(f64::NEG_INFINITY, f64::max),
    )
}

/// Tests if a Bezier curve is fully contained within all specified regions.
/// Samples key points (corners of bbox, endpoints, midpoint) for containment check.
pub fn is_bezier_inside_polygons(
    start_pos: Point,
    c1: Point,
    c2: Point,
    end_pos: Point,
    regions: &[Polygon],
) -> bool {
    let bbox = get_bezier_bounds(start_pos, c1, c2, end_pos);
    let mid = get_bezier_point_at(start_pos, c1, c2, end_pos, 0.5);

    let sample_points: Vec<Point> = vec![
        (bbox.0, bbox.1),
        (bbox.2, bbox.1),
        (bbox.2, bbox.3),
        (bbox.0, bbox.3),
        start_pos,
        end_pos,
        mid,
    ];

    for p in sample_points {
        if !regions
            .iter()
            .any(|region| is_point_inside_polygon(p, region))
        {
            return false;
        }
    }
    true
}

/// Finds all t parameters where a Bezier curve intersects a rectangle.
/// Solves the cubic equation for each edge of the rectangle.
pub fn get_bezier_rect_intersections(
    p0: Point,
    c1: Point,
    c2: Point,
    p1: Point,
    rect: Rect,
) -> Vec<f64> {
    let (x_min, y_min, x_max, y_max) = rect;
    let mut t_crossings: Vec<f64> = Vec::new();

    let rect_edges: [(usize, f64); 4] =
        [(0, x_min), (0, x_max), (1, y_min), (1, y_max)];

    for (axis_idx, edge_val) in rect_edges {
        let p0_coord = if axis_idx == 0 { p0.0 } else { p0.1 };
        let c1_coord = if axis_idx == 0 { c1.0 } else { c1.1 };
        let c2_coord = if axis_idx == 0 { c2.0 } else { c2.1 };
        let p1_coord = if axis_idx == 0 { p1.0 } else { p1.1 };

        let poly_a = p0_coord;
        let poly_b = 3.0 * (c1_coord - p0_coord);
        let poly_c = 3.0 * (c2_coord - c1_coord) - poly_b;
        let poly_d = p1_coord - poly_a - poly_b - poly_c;

        let roots = _solve_cubic(poly_d, poly_c, poly_b, poly_a - edge_val);
        for root in roots {
            if (-1e-9..=1.0 + 1e-9).contains(&root) {
                let clamped = root.clamp(0.0, 1.0);
                let point_on_curve =
                    get_bezier_point_at(p0, c1, c2, p1, clamped);
                let other_axis = 1 - axis_idx;
                let other_coord = if other_axis == 1 {
                    point_on_curve.1
                } else {
                    point_on_curve.0
                };
                let axis_lo = if other_axis == 1 { y_min } else { x_min };
                let axis_hi = if other_axis == 1 { y_max } else { x_max };
                if axis_lo - 1e-9 <= other_coord
                    && other_coord <= axis_hi + 1e-9
                {
                    let rounded = (clamped * 1e12).round() / 1e12;
                    if !t_crossings.contains(&rounded) {
                        t_crossings.push(rounded);
                    }
                }
            }
        }
    }

    if !t_crossings.contains(&0.0) {
        t_crossings.push(0.0);
    }
    if !t_crossings.contains(&1.0) {
        t_crossings.push(1.0);
    }

    t_crossings.sort_by(|a, b| a.partial_cmp(b).unwrap());
    t_crossings
}

pub fn clip_bezier_with_rect(
    p0: Point,
    c1: Point,
    c2: Point,
    p1: Point,
    rect: Rect,
) -> Vec<CubicBezier> {
    let (x_min, y_min, x_max, y_max) = rect;
    let crossing_params = get_bezier_rect_intersections(p0, c1, c2, p1, rect);

    if crossing_params.len() < 2 {
        return vec![];
    }

    // Extract segments that fall inside the rectangle
    let mut inside_segments: Vec<CubicBezier> = vec![];

    for i in 0..crossing_params.len() - 1 {
        let t_start = crossing_params[i];
        let t_end = crossing_params[i + 1];
        if (t_end - t_start).abs() < 1e-12 {
            continue;
        }
        let t_mid = (t_start + t_end) / 2.0;
        let midpoint_pt = get_bezier_point_at(p0, c1, c2, p1, t_mid);

        // Check if midpoint is inside rect
        if x_min - 1e-9 <= midpoint_pt.0
            && midpoint_pt.0 <= x_max + 1e-9
            && y_min - 1e-9 <= midpoint_pt.1
            && midpoint_pt.1 <= y_max + 1e-9
        {
            let segment = _extract_subsegment(p0, c1, c2, p1, t_start, t_end);
            inside_segments.push(segment);
        }
    }

    inside_segments
}

/// Approximates a cubic Bezier curve with a quadratic (single control point).
/// Uses a 3/7 weighted average of the control points for approximation.
pub fn convert_cubic_bezier_to_quadratic(
    p0: Point,
    c1: Point,
    c2: Point,
    p1: Point,
) -> (Point, Point, Point) {
    let quadratic_control = (
        3.0 / 7.0 * c1.0 + 3.0 / 7.0 * c2.0 + 1.0 / 7.0 * p0.0,
        3.0 / 7.0 * c1.1 + 3.0 / 7.0 * c2.1 + 1.0 / 7.0 * p0.1,
    );
    (p0, quadratic_control, p1)
}

pub fn get_bezier_closest_point(
    bezier_row: &[f64; 8],
    start_pos: Point3D,
    x: f64,
    y: f64,
) -> Option<(f64, Point, f64)> {
    // Linearize and search for closest point on segments
    let bezier_segments =
        linearize_bezier_from_array(bezier_row, start_pos, 0.005);
    if bezier_segments.is_empty() {
        return None;
    }

    let mut min_dist_sq = f64::INFINITY;
    let mut best_result: Option<(usize, f64, Point, f64)> = None;

    for (seg_idx, (seg_start, seg_end)) in bezier_segments.iter().enumerate() {
        let t_sub = get_line_segment_closest_point(
            (seg_start.0, seg_start.1),
            (seg_end.0, seg_end.1),
            x,
            y,
        );
        if t_sub.2 < min_dist_sq {
            min_dist_sq = t_sub.2;
            best_result = Some((seg_idx, t_sub.0, t_sub.1, t_sub.2));
        }
    }

    best_result.map(|(best_seg_idx, best_t_sub, best_pt, best_dist_sq)| {
        let t_bezier =
            (best_seg_idx as f64 + best_t_sub) / bezier_segments.len() as f64;
        (t_bezier, best_pt, best_dist_sq)
    })
}

/// Converts a Bezier curve from array format into line segments.
/// Estimates number of steps based on control point distances and resolution.
pub fn linearize_bezier_from_array(
    bezier_row: &[f64; 8],
    start_point: Point3D,
    resolution: f64,
) -> Vec<(Point3D, Point3D)> {
    let p0 = start_point;
    let p1 = (bezier_row[COL_X], bezier_row[COL_Y], bezier_row[COL_Z]);
    let c1_2d = (bezier_row[COL_C1X], bezier_row[COL_C1Y]);
    let c2_2d = (bezier_row[COL_C2X], bezier_row[COL_C2Y]);

    let z0 = p0.2;
    let z1 = p1.2;
    // Linear interpolation of Z coordinate for control points
    let c1: Point3D = (c1_2d.0, c1_2d.1, z0 * (2.0 / 3.0) + z1 * (1.0 / 3.0));
    let c2: Point3D = (c2_2d.0, c2_2d.1, z0 * (1.0 / 3.0) + z1 * (2.0 / 3.0));

    // Estimate curve length using polygon approximation
    let l01 = (p0.0 - c1.0).hypot(p0.1 - c1.1);
    let l12 = (c1.0 - c2.0).hypot(c1.1 - c2.1);
    let l23 = (c2.0 - p1.0).hypot(c2.1 - p1.1);
    let estimated_len = l01 + l12 + l23;
    let num_steps = (estimated_len / resolution).ceil().max(2.0) as usize;

    linearize_bezier(p0, c1, c2, p1, num_steps)
}

/// Converts a Bezier curve into line segments using uniform parameter steps.
pub fn linearize_bezier(
    p0: Point3D,
    c1: Point3D,
    c2: Point3D,
    p1: Point3D,
    num_steps: usize,
) -> Vec<(Point3D, Point3D)> {
    if num_steps < 1 {
        return vec![];
    }

    let mut result = Vec::with_capacity(num_steps);
    let step_f = num_steps as f64;

    for i in 0..num_steps {
        let t = i as f64 / step_f;
        let t_next = (i as f64 + 1.0) / step_f;

        let p_start = (
            (1.0 - t).powi(3) * p0.0
                + 3.0 * (1.0 - t).powi(2) * t * c1.0
                + 3.0 * (1.0 - t) * t.powi(2) * c2.0
                + t.powi(3) * p1.0,
            (1.0 - t).powi(3) * p0.1
                + 3.0 * (1.0 - t).powi(2) * t * c1.1
                + 3.0 * (1.0 - t) * t.powi(2) * c2.1
                + t.powi(3) * p1.1,
            (1.0 - t).powi(3) * p0.2
                + 3.0 * (1.0 - t).powi(2) * t * c1.2
                + 3.0 * (1.0 - t) * t.powi(2) * c2.2
                + t.powi(3) * p1.2,
        );

        let p_end = (
            (1.0 - t_next).powi(3) * p0.0
                + 3.0 * (1.0 - t_next).powi(2) * t_next * c1.0
                + 3.0 * (1.0 - t_next) * t_next.powi(2) * c2.0
                + t_next.powi(3) * p1.0,
            (1.0 - t_next).powi(3) * p0.1
                + 3.0 * (1.0 - t_next).powi(2) * t_next * c1.1
                + 3.0 * (1.0 - t_next) * t_next.powi(2) * c2.1
                + t_next.powi(3) * p1.1,
            (1.0 - t_next).powi(3) * p0.2
                + 3.0 * (1.0 - t_next).powi(2) * t_next * c1.2
                + 3.0 * (1.0 - t_next) * t_next.powi(2) * c2.2
                + t_next.powi(3) * p1.2,
        );

        result.push((p_start, p_end));
    }

    result
}

pub fn linearize_bezier_adaptive(
    p0: Point,
    c1: Point,
    c2: Point,
    p1: Point,
    tolerance_sq: f64,
    max_depth: usize,
) -> Polygon {
    let mut points: Polygon = vec![];

    /// Recursive subdivision with flatness test based on control point distances
    fn recursive_step(
        curve: CubicBezier,
        depth: usize,
        max_depth: usize,
        tolerance_sq: f64,
        points: &mut Polygon,
    ) {
        let (p0, c1, c2, p1) = curve;
        let vx = p1.0 - p0.0;
        let vy = p1.1 - p0.1;
        let norm_sq = vx * vx + vy * vy;

        // Test if curve is flat enough to approximate with a line
        let is_flat = if depth >= max_depth {
            true
        } else if norm_sq < 1e-9 {
            // Degenerate case: zero-length chord
            let d1_sq = (c1.0 - p0.0).powi(2) + (c1.1 - p0.1).powi(2);
            let d2_sq = (c2.0 - p0.0).powi(2) + (c2.1 - p0.1).powi(2);
            d1_sq < tolerance_sq && d2_sq < tolerance_sq
        } else {
            // Test cross product distance from chord line
            let term1 = -vy;
            let term2 = vx;
            let term3 = p0.0 * p1.1 - p0.1 * p1.0;

            let cross1 = (term1 * c1.0 + term2 * c1.1 - term3).abs();
            let cross2 = (term1 * c2.0 + term2 * c2.1 - term3).abs();

            let limit = tolerance_sq * norm_sq;
            cross1 * cross1 < limit && cross2 * cross2 < limit
        };

        if is_flat {
            return;
        }

        // Subdivide using de Casteljau's algorithm
        let m01 = ((p0.0 + c1.0) / 2.0, (p0.1 + c1.1) / 2.0);
        let m12 = ((c1.0 + c2.0) / 2.0, (c1.1 + c2.1) / 2.0);
        let m23 = ((c2.0 + p1.0) / 2.0, (c2.1 + p1.1) / 2.0);

        let q01 = ((m01.0 + m12.0) / 2.0, (m01.1 + m12.1) / 2.0);
        let q12 = ((m12.0 + m23.0) / 2.0, (m12.1 + m23.1) / 2.0);

        let r = ((q01.0 + q12.0) / 2.0, (q01.1 + q12.1) / 2.0);

        // Recurse on both halves
        recursive_step(
            (p0, m01, q01, r),
            depth + 1,
            max_depth,
            tolerance_sq,
            points,
        );
        points.push(r);
        recursive_step(
            (r, q12, m23, p1),
            depth + 1,
            max_depth,
            tolerance_sq,
            points,
        );
    }

    recursive_step((p0, c1, c2, p1), 0, max_depth, tolerance_sq, &mut points);
    points.push(p1);
    points
}

/// Internal: Computes perpendicular distance squared from point to line.
fn _perp_dist_sq(
    pt: Point3D,
    origin: Point3D,
    vx: f64,
    vy: f64,
    vz: f64,
    norm_sq: f64,
) -> f64 {
    let px = pt.0 - origin.0;
    let py = pt.1 - origin.1;
    let pz = pt.2 - origin.2;
    let cx = py * vz - pz * vy;
    let cy = pz * vx - px * vz;
    let cz = px * vy - py * vx;
    (cx * cx + cy * cy + cz * cz) / norm_sq
}

fn _bezier_flatness_sq(a: Point3D, b: Point3D, c: Point3D, d: Point3D) -> f64 {
    let vx = d.0 - a.0;
    let vy = d.1 - a.1;
    let vz = d.2 - a.2;
    let norm_sq = vx * vx + vy * vy + vz * vz;

    if norm_sq < 1e-9 {
        let d1 =
            (b.0 - a.0).powi(2) + (b.1 - a.1).powi(2) + (b.2 - a.2).powi(2);
        let d2 =
            (c.0 - a.0).powi(2) + (c.1 - a.1).powi(2) + (c.2 - a.2).powi(2);
        return d1.max(d2);
    }

    _perp_dist_sq(b, a, vx, vy, vz, norm_sq)
        .max(_perp_dist_sq(c, a, vx, vy, vz, norm_sq))
}

const BEZIER_SEG_MAX_DEPTH: usize = 10;

pub fn flatten_bezier(
    a: Point3D,
    b: Point3D,
    c: Point3D,
    d: Point3D,
    tolerance_sq: f64,
    depth: usize,
    points: &mut Polygon3D,
) {
    // Recursive subdivision with flatness test based on perpendicular distance
    if depth >= BEZIER_SEG_MAX_DEPTH
        || _bezier_flatness_sq(a, b, c, d) <= tolerance_sq
    {
        points.push(d);
        return;
    }

    let m01 = midpoint(a, b);
    let m12 = midpoint(b, c);
    let m23 = midpoint(c, d);
    let q01 = midpoint(m01, m12);
    let q12 = midpoint(m12, m23);
    let r = midpoint(q01, q12);

    flatten_bezier(a, m01, q01, r, tolerance_sq, depth + 1, points);
    flatten_bezier(r, q12, m23, d, tolerance_sq, depth + 1, points);
}

/// Default tolerance for Bezier linearization (0.01 units).
const BEZIER_SEG_DEFAULT_TOLERANCE: f64 = 0.01;

/// Converts a 3D Bezier curve to a polygon using adaptive subdivision.
/// Uses perpendicular distance for flatness testing.
pub fn linearize_bezier_segment(
    p0: Point3D,
    c1: Point3D,
    c2: Point3D,
    p1: Point3D,
    tolerance: Option<f64>,
) -> Polygon3D {
    let tolerance = tolerance.unwrap_or(BEZIER_SEG_DEFAULT_TOLERANCE);
    let tolerance_sq = tolerance * tolerance;

    let mut points: Polygon3D = vec![p0];
    flatten_bezier(p0, c1, c2, p1, tolerance_sq, 0, &mut points);
    points
}

fn _lerp2(a: Point, b: Point, t: f64) -> Point {
    (a.0 + (b.0 - a.0) * t, a.1 + (b.1 - a.1) * t)
}

fn _add_axis_extrema(
    candidates: &mut Vec<f64>,
    p0: f64,
    c1: f64,
    c2: f64,
    p1: f64,
) {
    let coeff_a = -p0 + 3.0 * c1 - 3.0 * c2 + p1;
    let coeff_b = 2.0 * (p0 - 2.0 * c1 + c2);
    let coeff_c = -p0 + c1;

    if coeff_a.abs() < 1e-12 {
        if coeff_b.abs() < 1e-12 {
            return;
        }
        let t = -coeff_c / coeff_b;
        if 0.0 < t && t < 1.0 {
            candidates.push(_eval_axis(p0, c1, c2, p1, t));
        }
        return;
    }

    let discriminant = coeff_b * coeff_b - 4.0 * coeff_a * coeff_c;
    if discriminant < 0.0 {
        return;
    }

    let sqrt_disc = discriminant.sqrt();
    for sign in [-1.0, 1.0] {
        let t = (-coeff_b + sign * sqrt_disc) / (2.0 * coeff_a);
        if 0.0 < t && t < 1.0 {
            candidates.push(_eval_axis(p0, c1, c2, p1, t));
        }
    }
}

fn _eval_axis(p0: f64, c1: f64, c2: f64, p1: f64, t: f64) -> f64 {
    let complement = 1.0 - t;
    complement.powi(3) * p0
        + 3.0 * complement.powi(2) * t * c1
        + 3.0 * complement * t.powi(2) * c2
        + t.powi(3) * p1
}

fn _extract_subsegment(
    p0: Point,
    c1: Point,
    c2: Point,
    p1: Point,
    t_start: f64,
    t_end: f64,
) -> CubicBezier {
    let starts_at_zero = t_start < 1e-12;
    let ends_at_one = (t_end - 1.0).abs() < 1e-12;

    if starts_at_zero && ends_at_one {
        return (p0, c1, c2, p1);
    }
    if starts_at_zero {
        let (left, _) = split_bezier(p0, c1, c2, p1, t_end);
        return left;
    }
    if ends_at_one {
        let (_, right) = split_bezier(p0, c1, c2, p1, t_start);
        return right;
    }

    let (_, right_after_start) = split_bezier(p0, c1, c2, p1, t_start);
    let reparam_end = (t_end - t_start) / (1.0 - t_start);
    let (left_of_end, _) = split_bezier(
        right_after_start.0,
        right_after_start.1,
        right_after_start.2,
        right_after_start.3,
        reparam_end,
    );
    left_of_end
}

fn _solve_cubic(a: f64, b: f64, c: f64, d: f64) -> Vec<f64> {
    if a.abs() < 1e-12 {
        if b.abs() < 1e-12 {
            if c.abs() < 1e-12 {
                return vec![];
            }
            return vec![-d / c];
        }
        let discriminant = c * c - 4.0 * b * d;
        if discriminant < 0.0 {
            return vec![];
        }
        let sqrt_disc = discriminant.sqrt();
        return vec![
            (-c + sqrt_disc) / (2.0 * b),
            (-c - sqrt_disc) / (2.0 * b),
        ];
    }

    let b = b / a;
    let c = c / a;
    let d = d / a;
    let _a = 1.0;

    let depressed_q = (3.0 * c - b * b) / 9.0;
    let depressed_r = (9.0 * b * c - 27.0 * d - 2.0 * b * b * b) / 54.0;
    let discriminant = depressed_q.powi(3) + depressed_r.powi(2);

    if discriminant >= 0.0 {
        let sqrt_disc = discriminant.sqrt();
        let cube_root_sum = _cbrt(depressed_r + sqrt_disc);
        let cube_root_diff = _cbrt(depressed_r - sqrt_disc);
        let real_root = cube_root_sum + cube_root_diff - b / 3.0;
        return vec![real_root];
    }

    let neg_q_cubed = if depressed_q < 0.0 {
        -depressed_q.powi(3)
    } else {
        1e-30
    };
    let cos_arg = (-1.0_f64)
        .max(1.0_f64)
        .min(depressed_r / neg_q_cubed.sqrt());
    let theta = cos_arg.acos();
    let amplitude = if depressed_q < 0.0 {
        2.0 * (-depressed_q).sqrt()
    } else {
        0.0
    };
    let offset = b / 3.0;

    vec![
        amplitude * (theta / 3.0).cos() - offset,
        amplitude * ((theta + 2.0 * PI) / 3.0).cos() - offset,
        amplitude * ((theta + 4.0 * PI) / 3.0).cos() - offset,
    ]
}

fn _cbrt(x: f64) -> f64 {
    if x >= 0.0 {
        x.powf(1.0 / 3.0)
    } else {
        -((-x).powf(1.0 / 3.0))
    }
}

pub fn perp_dist_sq(
    pt: Point3D,
    origin: Point3D,
    vx: f64,
    vy: f64,
    vz: f64,
    norm_sq: f64,
) -> f64 {
    let px = pt.0 - origin.0;
    let py = pt.1 - origin.1;
    let pz = pt.2 - origin.2;
    let cx = py * vz - pz * vy;
    let cy = pz * vx - px * vz;
    let cz = px * vy - py * vx;
    (cx * cx + cy * cy + cz * cz) / norm_sq
}

pub fn bezier_flatness_sq(
    a: Point3D,
    b: Point3D,
    c: Point3D,
    d: Point3D,
) -> f64 {
    let vx = d.0 - a.0;
    let vy = d.1 - a.1;
    let vz = d.2 - a.2;
    let norm_sq = vx * vx + vy * vy + vz * vz;

    if norm_sq < 1e-9 {
        let d1 =
            (b.0 - a.0).powi(2) + (b.1 - a.1).powi(2) + (b.2 - a.2).powi(2);
        let d2 =
            (c.0 - a.0).powi(2) + (c.1 - a.1).powi(2) + (c.2 - a.2).powi(2);
        return d1.max(d2);
    }

    let dist_b = perp_dist_sq(b, a, vx, vy, vz, norm_sq);
    let dist_c = perp_dist_sq(c, a, vx, vy, vz, norm_sq);
    dist_b.max(dist_c)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_get_bezier_point_at() {
        let p0: Point = (0.0, 0.0);
        let c1: Point = (0.0, 1.0);
        let c2: Point = (1.0, 1.0);
        let p1: Point = (1.0, 0.0);

        let result = get_bezier_point_at(p0, c1, c2, p1, 0.5);
        assert!((result.0 - 0.5).abs() < 1e-9);
        assert!((result.1 - 0.75).abs() < 1e-9);
    }

    #[test]
    fn test_split_bezier() {
        let (left, right) =
            split_bezier((0.0, 0.0), (0.0, 1.0), (1.0, 1.0), (1.0, 0.0), 0.5);
        assert!((left.3 .0 - right.0 .0).abs() < 1e-9);
        assert!((left.3 .1 - right.0 .1).abs() < 1e-9);
    }

    #[test]
    fn test_get_bezier_bounds() {
        let bounds =
            get_bezier_bounds((0.0, 0.0), (0.5, 1.0), (0.5, 1.0), (1.0, 0.0));
        assert!(bounds.0 >= 0.0 && bounds.2 <= 1.0);
        assert!(bounds.1 >= 0.0 && bounds.3 <= 1.0);
    }

    #[test]
    fn test_is_bezier_inside_polygons() {
        let region: Polygon =
            vec![(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)];
        assert!(is_bezier_inside_polygons(
            (0.25, 0.25),
            (0.3, 0.75),
            (0.7, 0.75),
            (0.75, 0.25),
            &[region]
        ));
    }

    #[test]
    fn test_get_bezier_rect_intersections() {
        let crossings = get_bezier_rect_intersections(
            (0.0, 0.0),
            (0.5, 1.0),
            (0.5, 1.0),
            (1.0, 0.0),
            (0.0, 0.0, 1.0, 1.0),
        );
        assert!(!crossings.is_empty());
        assert!(crossings.contains(&0.0));
        assert!(crossings.contains(&1.0));
    }

    #[test]
    fn test_clip_bezier_with_rect() {
        let segments = clip_bezier_with_rect(
            (0.0, 0.0),
            (0.5, 1.0),
            (0.5, 1.0),
            (1.0, 0.0),
            (0.0, 0.0, 1.0, 1.0),
        );
        assert!(!segments.is_empty());
    }

    #[test]
    fn test_convert_cubic_bezier_to_quadratic() {
        let result = convert_cubic_bezier_to_quadratic(
            (0.0, 0.0),
            (0.0, 1.0),
            (1.0, 1.0),
            (1.0, 0.0),
        );
        assert_eq!(result.0, (0.0, 0.0));
        assert_eq!(result.2, (1.0, 0.0));
    }

    #[test]
    fn test_linearize_bezier() {
        let segments = linearize_bezier(
            (0.0, 0.0, 0.0),
            (0.0, 1.0, 0.0),
            (1.0, 1.0, 0.0),
            (1.0, 0.0, 0.0),
            10,
        );
        assert_eq!(segments.len(), 10);
    }

    #[test]
    fn test_linearize_bezier_from_array() {
        let bezier_row: [f64; 8] =
            [CMD_TYPE_BEZIER, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0];
        let start: Point3D = (0.0, 0.0, 0.0);
        let segments = linearize_bezier_from_array(&bezier_row, start, 0.1);
        assert!(!segments.is_empty());
        assert_eq!(segments.last().unwrap().1, (1.0, 0.0, 0.0));
    }

    #[test]
    fn test_linearize_bezier_adaptive() {
        let result = linearize_bezier_adaptive(
            (0.0, 0.0),
            (0.0, 1.0),
            (1.0, 1.0),
            (1.0, 0.0),
            0.01,
            10,
        );
        assert!(!result.is_empty());
        assert_eq!(result.last(), Some(&(1.0, 0.0)));
    }

    #[test]
    fn test_flatten_bezier() {
        let mut points: Polygon3D = vec![(0.0, 0.0, 0.0)];
        flatten_bezier(
            (0.0, 0.0, 0.0),
            (0.0, 1.0, 0.0),
            (1.0, 1.0, 0.0),
            (1.0, 0.0, 0.0),
            0.01,
            0,
            &mut points,
        );
        assert!(!points.is_empty());
        assert_eq!(points.last(), Some(&(1.0, 0.0, 0.0)));
    }

    #[test]
    fn test_linearize_bezier_segment() {
        let result = linearize_bezier_segment(
            (0.0, 0.0, 0.0),
            (0.0, 1.0, 0.0),
            (1.0, 1.0, 0.0),
            (1.0, 0.0, 0.0),
            None,
        );
        assert!(!result.is_empty());
        assert_eq!(result.first(), Some(&(0.0, 0.0, 0.0)));
        assert_eq!(result.last(), Some(&(1.0, 0.0, 0.0)));
    }

    #[test]
    fn test_solve_cubic() {
        let roots = _solve_cubic(1.0, -6.0, 11.0, -6.0);
        assert_eq!(roots.len(), 3);
    }
}
