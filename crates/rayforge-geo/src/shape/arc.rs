//! Arc: Circular arc geometry operations.
//!
//! This module provides functions for working with circular arcs including:
//! - Angle normalization and computation
//! - Bounding box calculation
//! - Direction determination
//! - Linearization into line segments
//! - Intersection tests with rectangles, circles, and regions

use std::f64::consts::PI;

use crate::constants::*;
use crate::shape::line::{
    does_line_segment_intersect_rect, get_line_segment_closest_point,
};
use crate::shape::polygon::is_point_inside_polygon;
use crate::types::{Point, Point3D, Polygon, Rect};

/// Normalizes an angle to the range [0, 2*PI).
pub fn normalize_angle(angle: f64) -> f64 {
    ((angle % (2.0 * PI)) + 2.0 * PI) % (2.0 * PI)
}

/// Computes the start angle, end angle, and sweep angle for an arc.
/// Handles the direction (CW/CCW) to compute the correct sweep.
pub fn get_arc_angles(
    start_pos: Point,
    end_pos: Point,
    center: Point,
    clockwise: bool,
) -> (f64, f64, f64) {
    let start_angle = (start_pos.1 - center.1).atan2(start_pos.0 - center.0);
    let end_angle = (end_pos.1 - center.1).atan2(end_pos.0 - center.0);

    // Compute sweep angle, adjusting for direction
    let mut sweep = end_angle - start_angle;
    if clockwise {
        if sweep > 0.0 {
            sweep -= 2.0 * PI;
        }
    } else {
        if sweep < 0.0 {
            sweep += 2.0 * PI;
        }
    }

    (start_angle, end_angle, sweep)
}

/// Computes the midpoint of an arc (at t=0.5 along the arc).
pub fn get_arc_midpoint(
    start_pos: Point,
    end_pos: Point,
    center: Point,
    clockwise: bool,
) -> Point {
    let (start_a, _, sweep) =
        get_arc_angles(start_pos, end_pos, center, clockwise);
    let mid_angle = start_a + sweep / 2.0;
    let radius = (start_pos.0 - center.0).hypot(start_pos.1 - center.1);
    (
        center.0 + radius * mid_angle.cos(),
        center.1 + radius * mid_angle.sin(),
    )
}

/// Tests if an angle falls between start and end angles following the arc direction.
pub fn is_angle_between(
    target: f64,
    start: f64,
    end: f64,
    clockwise: bool,
) -> bool {
    let target = normalize_angle(target);
    let start = normalize_angle(start);
    let end = normalize_angle(end);

    if clockwise {
        // For clockwise: sweep goes from higher to lower (wrapping around 0)
        if start < end {
            target <= start || target >= end
        } else {
            end <= target && target <= start
        }
    } else {
        // For counter-clockwise: sweep goes from lower to higher
        if start > end {
            target >= start || target <= end
        } else {
            start <= target && target <= end
        }
    }
}

/// Computes the axis-aligned bounding box that contains the entire arc.
/// Checks cardinal directions (0, 90, 180, 270 degrees) to find extrema.
pub fn get_arc_bounds(
    start_pos: Point,
    end_pos: Point,
    center_offset: Point,
    clockwise: bool,
) -> Rect {
    let center_x = start_pos.0 + center_offset.0;
    let center_y = start_pos.1 + center_offset.1;
    let radius = center_offset.0.hypot(center_offset.1);

    let mut min_x = start_pos.0.min(end_pos.0);
    let mut min_y = start_pos.1.min(end_pos.1);
    let mut max_x = start_pos.0.max(end_pos.0);
    let mut max_y = start_pos.1.max(end_pos.1);

    let start_angle = (start_pos.1 - center_y).atan2(start_pos.0 - center_x);
    let end_angle = (end_pos.1 - center_y).atan2(end_pos.0 - center_x);

    if is_angle_between(0.0, start_angle, end_angle, clockwise) {
        max_x = max_x.max(center_x + radius);
    }
    if is_angle_between(PI / 2.0, start_angle, end_angle, clockwise) {
        max_y = max_y.max(center_y + radius);
    }
    if is_angle_between(PI, start_angle, end_angle, clockwise) {
        min_x = min_x.min(center_x - radius);
    }
    if is_angle_between(3.0 * PI / 2.0, start_angle, end_angle, clockwise) {
        min_y = min_y.min(center_y - radius);
    }

    (min_x, min_y, max_x, max_y)
}

/// Determines arc direction based on center, start point, and a third reference point.
/// Uses cross product of vectors from center to determine if the reference is on
/// the clockwise or counter-clockwise side. Used for interactive arc drawing.
pub fn get_arc_direction(center: Point, start: Point, mouse: Point) -> bool {
    let vec_s_x = start.0 - center.0;
    let vec_s_y = start.1 - center.1;
    let vec_m_x = mouse.0 - center.0;
    let vec_m_y = mouse.1 - center.1;

    // Negative cross product indicates clockwise
    let det = vec_s_x * vec_m_y - vec_s_y * vec_m_x;
    det < 0.0
}

/// Internal: Linearizes an arc into line segments for approximation.
pub fn _linearize_arc_from_array(
    arc_row: &[f64; 8],
    start_point: Point3D,
    resolution: f64,
) -> Vec<(Point3D, Point3D)> {
    let mut segments: Vec<(Point3D, Point3D)> = Vec::new();
    let p0 = start_point;
    let p1 = (arc_row[COL_X], arc_row[COL_Y], arc_row[COL_Z]);
    let center_offset = (arc_row[COL_I], arc_row[COL_J]);
    let clockwise = arc_row[COL_CW] != 0.0;
    let z0 = p0.2;
    let z1 = p1.2;

    let center = (p0.0 + center_offset.0, p0.1 + center_offset.1);

    let radius_start = (p0.0 - center.0).hypot(p0.1 - center.1);
    let radius_end = (p1.0 - center.0).hypot(p1.1 - center.1);

    if radius_start < 1e-9 {
        return vec![(p0, p1)];
    }

    let start_angle = (p0.1 - center.1).atan2(p0.0 - center.0);
    let end_angle = (p1.1 - center.1).atan2(p1.0 - center.0);

    let is_coincident =
        (p0.0 - p1.0).abs() < 1e-9 && (p0.1 - p1.1).abs() < 1e-9;

    let mut angle_range = if is_coincident && radius_start > 1e-9 {
        if clockwise {
            -2.0 * PI
        } else {
            2.0 * PI
        }
    } else {
        end_angle - start_angle
    };

    if clockwise {
        if angle_range > 1e-9 {
            angle_range -= 2.0 * PI;
        }
    } else {
        if angle_range < -1e-9 {
            angle_range += 2.0 * PI;
        }
    }

    let avg_radius = (radius_start + radius_end) / 2.0;
    let arc_len = angle_range.abs() * avg_radius;
    let mut num_segments = (arc_len / resolution).ceil().max(2.0) as usize;
    if is_coincident {
        num_segments = num_segments.div_ceil(2) * 2;
    }

    let mut prev_pt = p0;
    for i in 1..=num_segments {
        let t = i as f64 / num_segments as f64;
        let radius = radius_start + (radius_end - radius_start) * t;
        let angle = start_angle + angle_range * t;
        let z = z0 + (z1 - z0) * t;
        let next_pt = (
            center.0 + radius * angle.cos(),
            center.1 + radius * angle.sin(),
            z,
        );
        segments.push((prev_pt, next_pt));
        prev_pt = next_pt;
    }
    segments
}

/// Converts an arc into a series of line segments for approximation.
/// The resolution parameter controls the maximum length of each segment.
pub fn linearize_arc(
    arc_row: &[f64; 8],
    start_point: Point3D,
    resolution: f64,
) -> Vec<(Point3D, Point3D)> {
    _linearize_arc_from_array(arc_row, start_point, resolution)
}

/// Internal: Finds closest point on arc using linearized approximation.
fn _find_closest_on_linearized_arc(
    arc_row: &[f64; 8],
    start_pos: Point3D,
    x: f64,
    y: f64,
) -> Option<(f64, Point, f64)> {
    let arc_segments = linearize_arc(arc_row, start_pos, 0.1);
    if arc_segments.is_empty() {
        return None;
    }

    let mut min_dist_sq = f64::INFINITY;
    let mut best_result: Option<(usize, f64, Point, f64)> = None;

    for (j, (p1_3d, p2_3d)) in arc_segments.iter().enumerate() {
        let t_sub = get_line_segment_closest_point(
            (p1_3d.0, p1_3d.1),
            (p2_3d.0, p2_3d.1),
            x,
            y,
        );
        if t_sub.2 < min_dist_sq {
            min_dist_sq = t_sub.2;
            best_result = Some((j, t_sub.0, t_sub.1, t_sub.2));
        }
    }

    best_result.map(|(j_best, t_sub_best, pt_best, dist_sq_best)| {
        let t_arc = (j_best as f64 + t_sub_best) / arc_segments.len() as f64;
        (t_arc, pt_best, dist_sq_best)
    })
}

fn _find_closest_point_on_arc_from_array(
    arc_row: &[f64; 8],
    start_pos: Point3D,
    x: f64,
    y: f64,
) -> Option<(f64, Point, f64)> {
    let p0 = (start_pos.0, start_pos.1);
    let p1 = (arc_row[COL_X], arc_row[COL_Y]);
    let center_offset = (arc_row[COL_I], arc_row[COL_J]);
    let clockwise = arc_row[COL_CW] != 0.0;
    let center = (p0.0 + center_offset.0, p0.1 + center_offset.1);

    let radius_start = (p0.0 - center.0).hypot(p0.1 - center.1);
    let radius_end = (p1.0 - center.0).hypot(p1.1 - center.1);

    if (radius_start - radius_end).abs() > 1e-9 {
        return _find_closest_on_linearized_arc(arc_row, start_pos, x, y);
    }

    let radius = radius_start;
    if radius < 1e-9 {
        let dist_sq = (x - p0.0).powi(2) + (y - p0.1).powi(2);
        return Some((0.0, p0, dist_sq));
    }

    let vec_to_point = (x - center.0, y - center.1);
    let dist_to_center = vec_to_point.0.hypot(vec_to_point.1);
    let closest_on_circle = if dist_to_center < 1e-9 {
        p0
    } else {
        (
            center.0 + vec_to_point.0 / dist_to_center * radius,
            center.1 + vec_to_point.1 / dist_to_center * radius,
        )
    };

    let start_angle = (p0.1 - center.1).atan2(p0.0 - center.0);
    let end_angle = (p1.1 - center.1).atan2(p1.0 - center.0);
    let point_angle =
        (closest_on_circle.1 - center.1).atan2(closest_on_circle.0 - center.0);

    let mut angle_range = end_angle - start_angle;
    let mut angle_to_check = point_angle - start_angle;

    if clockwise {
        if angle_range > 1e-9 {
            angle_range -= 2.0 * PI;
        }
        if angle_to_check > 1e-9 {
            angle_to_check -= 2.0 * PI;
        }
    } else {
        if angle_range < -1e-9 {
            angle_range += 2.0 * PI;
        }
        if angle_to_check < -1e-9 {
            angle_to_check += 2.0 * PI;
        }
    }

    let is_on_arc = if clockwise {
        angle_to_check >= angle_range - 1e-9 && angle_to_check <= 1e-9
    } else {
        angle_to_check <= angle_range + 1e-9 && angle_to_check >= -1e-9
    };

    let (closest_point, t) = if is_on_arc {
        (
            closest_on_circle,
            if angle_range.abs() > 1e-9 {
                angle_to_check / angle_range
            } else {
                0.0
            },
        )
    } else {
        let dist_sq_p0 = (x - p0.0).powi(2) + (y - p0.1).powi(2);
        let dist_sq_p1 = (x - p1.0).powi(2) + (y - p1.1).powi(2);
        if dist_sq_p0 <= dist_sq_p1 {
            (p0, 0.0)
        } else {
            (p1, 1.0)
        }
    };

    let dist_sq = (x - closest_point.0).powi(2) + (y - closest_point.1).powi(2);
    let t = t.clamp(0.0, 1.0);
    Some((t, closest_point, dist_sq))
}

/// Finds the closest point on an arc to a given (x, y) coordinate.
/// Returns (t_parameter, closest_point, distance_squared).
pub fn get_arc_closest_point(
    arc_row: &[f64; 8],
    start_pos: Point3D,
    x: f64,
    y: f64,
) -> Option<(f64, Point, f64)> {
    _find_closest_point_on_arc_from_array(arc_row, start_pos, x, y)
}

/// Tests if an arc intersects an axis-aligned rectangle.
/// Uses bounding box check followed by linearized segment testing.
pub fn does_arc_intersect_rect(
    start_pos: Point,
    end_pos: Point,
    center: Point,
    clockwise: bool,
    rect: Rect,
) -> bool {
    // Quick bounding box rejection test
    let arc_box = get_arc_bounds(
        start_pos,
        end_pos,
        (center.0 - start_pos.0, center.1 - start_pos.1),
        clockwise,
    );
    if arc_box.2 < rect.0
        || arc_box.0 > rect.2
        || arc_box.3 < rect.1
        || arc_box.1 > rect.3
    {
        return false;
    }

    // Linearize and test each segment
    let center_offset = (center.0 - start_pos.0, center.1 - start_pos.1);
    let radius = (start_pos.0 - center.0).hypot(start_pos.1 - center.1);
    let start_3d: Point3D = (start_pos.0, start_pos.1, 0.0);

    let arc_data: [f64; 8] = [
        CMD_TYPE_ARC,
        end_pos.0,
        end_pos.1,
        0.0,
        center_offset.0,
        center_offset.1,
        if clockwise { 1.0 } else { 0.0 },
        0.0,
    ];

    let segments = linearize_arc(&arc_data, start_3d, radius * 0.1);
    for (p1_3d, p2_3d) in segments {
        if does_line_segment_intersect_rect(
            (p1_3d.0, p1_3d.1),
            (p2_3d.0, p2_3d.1),
            rect,
        ) {
            return true;
        }
    }

    false
}

/// Tests if an arc intersects a circle. Checks:
/// 1. If either endpoint is inside the circle
/// 2. Circle-arc intersection points fall on the arc
/// 3. Midpoint of arc is inside the circle
pub fn does_arc_intersect_circle(
    start_pos: Point,
    end_pos: Point,
    center: Point,
    clockwise: bool,
    circle_center: Point,
    circle_radius: f64,
) -> bool {
    use crate::shape::circle::get_circle_circle_intersections;

    let radius = (start_pos.0 - center.0).hypot(start_pos.1 - center.1);
    if radius < 1e-9 {
        return (start_pos.0 - circle_center.0)
            .hypot(start_pos.1 - circle_center.1)
            <= circle_radius;
    }

    // Check if either endpoint is inside the circle
    if (start_pos.0 - circle_center.0).hypot(start_pos.1 - circle_center.1)
        <= circle_radius
    {
        return true;
    }
    if (end_pos.0 - circle_center.0).hypot(end_pos.1 - circle_center.1)
        <= circle_radius
    {
        return true;
    }

    // Check if circle-arc intersection points fall on the arc
    let intersections = get_circle_circle_intersections(
        center,
        radius,
        circle_center,
        circle_radius,
    );
    if !intersections.is_empty() {
        let start_angle =
            (start_pos.1 - center.1).atan2(start_pos.0 - center.0);
        let end_angle = (end_pos.1 - center.1).atan2(end_pos.0 - center.0);
        for pt in intersections {
            let angle = (pt.1 - center.1).atan2(pt.0 - center.0);
            if is_angle_between(angle, start_angle, end_angle, clockwise) {
                return true;
            }
        }
    }

    // Check midpoint of arc as fallback
    let mid = get_arc_midpoint(start_pos, end_pos, center, clockwise);
    if (mid.0 - circle_center.0).hypot(mid.1 - circle_center.1) <= circle_radius
    {
        return true;
    }

    false
}

/// Tests if an arc is fully contained within all specified regions.
/// Samples key points (corners of bbox, endpoints, midpoint) for containment check.
pub fn is_arc_inside_polygons(
    start_pos: Point,
    end_pos: Point,
    center_offset: Point,
    clockwise: bool,
    regions: &[Polygon],
) -> bool {
    let center = (start_pos.0 + center_offset.0, start_pos.1 + center_offset.1);
    let bbox = get_arc_bounds(start_pos, end_pos, center_offset, clockwise);
    let mid = get_arc_midpoint(start_pos, end_pos, center, clockwise);

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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_normalize_angle() {
        assert!((normalize_angle(0.0) - 0.0).abs() < 1e-9);
        assert!((normalize_angle(2.0 * PI) - 0.0).abs() < 1e-9);
        assert!((normalize_angle(-PI) - PI).abs() < 1e-9);
    }

    #[test]
    fn test_get_arc_midpoint() {
        let start = (1.0, 0.0);
        let end = (-1.0, 0.0);
        let center = (0.0, 0.0);
        let mid = get_arc_midpoint(start, end, center, true);
        assert!((mid.0).abs() < 1e-9);
        assert!((mid.1 - (-1.0)).abs() < 1e-9);
    }

    #[test]
    fn test_is_angle_between() {
        assert!(is_angle_between(PI, 0.0, PI, false));
        assert!(is_angle_between(PI, PI, 0.0, true));
        assert!(!is_angle_between(PI / 2.0, 0.0, PI, true));
    }

    #[test]
    fn test_arc_bounding_box() {
        let start = (1.0, 0.0);
        let end = (-1.0, 0.0);
        let center_offset = (-1.0, 0.0);
        let bbox = get_arc_bounds(start, end, center_offset, true);
        assert!((bbox.0 - (-1.0)).abs() < 1e-9);
        assert!((bbox.2 - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_get_arc_direction() {
        let center = (0.0, 0.0);
        let start = (1.0, 0.0);
        let mouse = (0.0, 1.0);
        assert!(!get_arc_direction(center, start, mouse));

        let mouse2 = (0.0, -1.0);
        assert!(get_arc_direction(center, start, mouse2));
    }

    #[test]
    fn test_linearize_arc() {
        let arc_row: [f64; 8] =
            [CMD_TYPE_ARC, 1.0, 0.0, 0.0, -1.0, 0.0, 1.0, 0.0];
        let start: Point3D = (0.0, 0.0, 0.0);
        let segments = linearize_arc(&arc_row, start, 0.1);
        assert!(!segments.is_empty());
        assert_eq!(segments.last().unwrap().1, (1.0, 0.0, 0.0));
    }

    #[test]
    fn test_get_arc_closest_point() {
        let arc_row: [f64; 8] =
            [CMD_TYPE_ARC, 1.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0];
        let start: Point3D = (0.0, 0.0, 0.0);
        let result = get_arc_closest_point(&arc_row, start, 0.5, 0.5);
        assert!(result.is_some());
        let (t, _pt, _) = result.unwrap();
        assert!((0.0..=1.0).contains(&t));
    }

    #[test]
    fn test_does_arc_intersect_rect() {
        let rect: Rect = (-2.0, -2.0, 2.0, 2.0);
        assert!(does_arc_intersect_rect(
            (0.0, 1.0),
            (0.0, -1.0),
            (0.0, 0.0),
            true,
            rect
        ));
        assert!(!does_arc_intersect_rect(
            (5.0, 0.0),
            (6.0, 0.0),
            (5.5, 0.0),
            true,
            rect
        ));
    }

    #[test]
    fn test_does_arc_intersect_circle() {
        assert!(does_arc_intersect_circle(
            (1.0, 0.0),
            (-1.0, 0.0),
            (0.0, 0.0),
            true,
            (0.0, 0.0),
            1.5
        ));
        assert!(!does_arc_intersect_circle(
            (1.0, 0.0),
            (-1.0, 0.0),
            (0.0, 0.0),
            true,
            (10.0, 10.0),
            0.5
        ));
    }

    #[test]
    fn test_is_arc_inside_polygons() {
        let center_offset = (-1.0, 0.0);
        let region: Polygon =
            vec![(-2.0, -2.0), (2.0, -2.0), (2.0, 2.0), (-2.0, 2.0)];
        assert!(is_arc_inside_polygons(
            (0.0, 1.0),
            (0.0, -1.0),
            center_offset,
            true,
            &[region]
        ));

        let small_region: Polygon =
            vec![(0.0, -0.5), (0.5, -0.5), (0.5, 0.5), (0.0, 0.5)];
        assert!(!is_arc_inside_polygons(
            (0.0, 1.0),
            (0.0, -1.0),
            center_offset,
            true,
            &[small_region]
        ));
    }
}
