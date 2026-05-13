//! Line: Line segment operations.
//!
//! This module provides functions for working with lines and line segments.

use crate::types::{Point, Polygon, Rect};

/// Checks if a point lies on a line segment using dot product projection.
pub fn is_point_on_segment(pt: Point, p1: Point, p2: Point) -> bool {
    let dot1 = (pt.0 - p1.0) * (p2.0 - p1.0) + (pt.1 - p1.1) * (p2.1 - p1.1);
    if dot1 < 0.0 {
        return false;
    }
    let dot2 = (pt.0 - p2.0) * (p1.0 - p2.0) + (pt.1 - p2.1) * (p1.1 - p2.1);
    if dot2 < 0.0 {
        return false;
    }
    true
}

/// Computes the intersection of two infinite lines.
/// Returns None if lines are parallel (denominator = 0).
pub fn get_line_line_intersection(
    p1: Point,
    p2: Point,
    p3: Point,
    p4: Point,
) -> Option<Point> {
    let (x1, y1) = p1;
    let (x2, y2) = p2;
    let (x3, y3) = p3;
    let (x4, y4) = p4;

    let denom = (y4 - y3) * (x2 - x1) - (x4 - x3) * (y2 - y1);
    if denom == 0.0 {
        return None;
    }

    let ua = ((x4 - x3) * (y1 - y3) - (y4 - y3) * (x1 - x3)) / denom;
    Some((x1 + ua * (x2 - x1), y1 + ua * (y2 - y1)))
}

/// Computes the intersection of two line segments.
/// Returns None if segments don't intersect (even if their infinite lines do).
/// Uses parameter t for first segment [0,1] and u for second segment [0,1].
pub fn get_line_segment_intersection(
    p1: Point,
    p2: Point,
    p3: Point,
    p4: Point,
) -> Option<Point> {
    let (x1, y1) = p1;
    let (x2, y2) = p2;
    let (x3, y3) = p3;
    let (x4, y4) = p4;

    let den = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4);
    if den.abs() < 1e-9 {
        return None;
    }

    let t_num = (x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4);
    let u_num = -((x1 - x2) * (y1 - y3) - (y1 - y2) * (x1 - x3));

    let t = t_num / den;
    let u = u_num / den;

    if (0.0..=1.0).contains(&t) && (0.0..=1.0).contains(&u) {
        Some((x1 + t * (x2 - x1), y1 + t * (y2 - y1)))
    } else {
        None
    }
}

/// Finds the closest point on an infinite line to a given point.
/// Uses projection to find the point, returns first endpoint if line is degenerate.
pub fn get_line_closest_point(p1: Point, p2: Point, x: f64, y: f64) -> Point {
    let dx = p2.0 - p1.0;
    let dy = p2.1 - p1.1;
    let px = x - p1.0;
    let py = y - p1.1;

    let len_sq = dx * dx + dy * dy;
    if len_sq < 1e-12 {
        return p1;
    }

    let t = (px * dx + py * dy) / len_sq;
    (p1.0 + t * dx, p1.1 + t * dy)
}

/// Finds the closest point on a line segment to a given point.
/// Returns (t_parameter, closest_point, distance_squared).
/// t is in range [0, 1] representing position along the segment.
pub fn get_line_segment_closest_point(
    p1: Point,
    p2: Point,
    x: f64,
    y: f64,
) -> (f64, Point, f64) {
    let dx = p2.0 - p1.0;
    let dy = p2.1 - p1.1;
    let len_sq = dx * dx + dy * dy;

    let t = if len_sq < 1e-12 {
        0.0
    } else {
        ((x - p1.0) * dx + (y - p1.1) * dy) / len_sq
    };

    let t = t.clamp(0.0, 1.0);

    let closest_x = p1.0 + t * dx;
    let closest_y = p1.1 + t * dy;
    let dist_sq = (x - closest_x).powi(2) + (y - closest_y).powi(2);

    (t, (closest_x, closest_y), dist_sq)
}

/// Perpendicular distance from a point to a line segment.
pub fn get_point_line_distance(
    point: Point,
    line_start: Point,
    line_end: Point,
) -> f64 {
    let line_vec = (line_end.0 - line_start.0, line_end.1 - line_start.1);
    let line_len = (line_vec.0 * line_vec.0 + line_vec.1 * line_vec.1).sqrt();
    if line_len < 1e-6 {
        let dx = point.0 - line_start.0;
        let dy = point.1 - line_start.1;
        return (dx * dx + dy * dy).sqrt();
    }
    let line_unit = (line_vec.0 / line_len, line_vec.1 / line_len);
    let point_vec = (point.0 - line_start.0, point.1 - line_start.1);
    let mut proj_len = point_vec.0 * line_unit.0 + point_vec.1 * line_unit.1;
    proj_len = proj_len.max(0.0).min(line_len);
    let closest_x = line_start.0 + proj_len * line_unit.0;
    let closest_y = line_start.1 + proj_len * line_unit.1;
    let dx = point.0 - closest_x;
    let dy = point.1 - closest_y;
    (dx * dx + dy * dy).sqrt()
}

/// Finds all intersection parameters along a line segment with multiple polygons.
/// Returns sorted list of t values [0, 1] where segment intersects any region.
pub fn get_line_segment_polygon_intersections(
    p1_2d: Point,
    p2_2d: Point,
    regions: &[Polygon],
) -> Vec<f64> {
    let mut cut_points_t: Vec<f64> = vec![0.0, 1.0];

    for region in regions {
        for i in 0..region.len() {
            let p3 = region[i];
            let p4 = region[(i + 1) % region.len()];
            if let Some(intersection) =
                get_line_segment_intersection(p1_2d, p2_2d, p3, p4)
            {
                let (ix, iy) = intersection;
                let seg_dx = p2_2d.0 - p1_2d.0;
                let seg_dy = p2_2d.1 - p1_2d.1;

                let t = if seg_dx.abs() > seg_dy.abs() {
                    if seg_dx != 0.0 {
                        (ix - p1_2d.0) / seg_dx
                    } else {
                        0.0
                    }
                } else if seg_dy != 0.0 {
                    (iy - p1_2d.1) / seg_dy
                } else {
                    0.0
                };
                let t_clamped = t.clamp(0.0, 1.0);
                if !cut_points_t.contains(&t_clamped) {
                    cut_points_t.push(t_clamped);
                }
            }
        }
    }

    cut_points_t.sort_by(|a, b| a.partial_cmp(b).unwrap());
    cut_points_t
}

/// Tests if a 2D point is inside an axis-aligned rectangle.
pub fn is_point_inside_rect(point: Point, rect: Rect) -> bool {
    let (x, y) = point;
    let (rx1, ry1, rx2, ry2) = rect;
    x >= rx1 && x <= rx2 && y >= ry1 && y <= ry2
}

/// Tests if rect_b is completely contained within rect_a.
pub fn does_rect_contain_rect(rect_a: Rect, rect_b: Rect) -> bool {
    let (ax1, ay1, ax2, ay2) = rect_a;
    let (bx1, by1, bx2, by2) = rect_b;
    bx1 >= ax1 && by1 >= ay1 && bx2 <= ax2 && by2 <= ay2
}

/// Tests if two rectangles intersect.
pub fn does_rect_intersect_rect(rect_a: Rect, rect_b: Rect) -> bool {
    let (ax1, ay1, ax2, ay2) = rect_a;
    let (bx1, by1, bx2, by2) = rect_b;
    !(ax2 < bx1 || ax1 > bx2 || ay2 < by1 || ay1 > by2)
}

/// Tests if a line segment intersects an axis-aligned rectangle.
/// Checks if either endpoint is inside the rect or if segment crosses any edge.
pub fn does_line_segment_intersect_rect(
    p1: Point,
    p2: Point,
    rect: Rect,
) -> bool {
    let (xmin, ymin, xmax, ymax) = rect;

    if p1.0 >= xmin && p1.0 <= xmax && p1.1 >= ymin && p1.1 <= ymax {
        return true;
    }
    if p2.0 >= xmin && p2.0 <= xmax && p2.1 >= ymin && p2.1 <= ymax {
        return true;
    }

    let intersections = [
        get_line_segment_intersection(p1, p2, (xmin, ymin), (xmax, ymin)),
        get_line_segment_intersection(p1, p2, (xmax, ymin), (xmax, ymax)),
        get_line_segment_intersection(p1, p2, (xmax, ymax), (xmin, ymax)),
        get_line_segment_intersection(p1, p2, (xmin, ymax), (xmin, ymin)),
    ];

    intersections.iter().filter(|x| x.is_some()).count() > 0
}

/// Tests if a line segment intersects a circle by checking closest point distance.
pub fn does_line_segment_intersect_circle(
    p1: Point,
    p2: Point,
    center: Point,
    radius: f64,
) -> bool {
    let (_, _, dist_sq) =
        get_line_segment_closest_point(p1, p2, center.0, center.1);
    dist_sq <= radius * radius
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_is_point_on_segment() {
        let pt: Point = (0.5, 0.0);
        let p1: Point = (0.0, 0.0);
        let p2: Point = (1.0, 0.0);
        assert!(is_point_on_segment(pt, p1, p2));

        let pt2: Point = (1.5, 0.0);
        assert!(!is_point_on_segment(pt2, p1, p2));
    }

    #[test]
    fn test_line_intersection() {
        let p1: Point = (0.0, 0.0);
        let p2: Point = (4.0, 4.0);
        let p3: Point = (0.0, 4.0);
        let p4: Point = (4.0, 0.0);

        let result = get_line_line_intersection(p1, p2, p3, p4);
        assert!(result.is_some());
        let intersection = result.unwrap();
        assert!((intersection.0 - 2.0).abs() < 1e-9);
        assert!((intersection.1 - 2.0).abs() < 1e-9);
    }

    #[test]
    fn test_line_segment_intersection() {
        let p1: Point = (0.0, 0.0);
        let p2: Point = (2.0, 2.0);
        let p3: Point = (0.0, 2.0);
        let p4: Point = (2.0, 0.0);

        let result = get_line_segment_intersection(p1, p2, p3, p4);
        assert!(result.is_some());

        let no_intersect = get_line_segment_intersection(
            (0.0, 0.0),
            (1.0, 1.0),
            (2.0, 0.0),
            (3.0, 1.0),
        );
        assert!(no_intersect.is_none());
    }

    #[test]
    fn test_closest_point_on_line_segment() {
        let p1: Point = (0.0, 0.0);
        let p2: Point = (4.0, 0.0);

        let (t, pt, dist_sq) = get_line_segment_closest_point(p1, p2, 2.0, 1.0);
        assert!((t - 0.5).abs() < 1e-9);
        assert!((pt.0 - 2.0).abs() < 1e-9);
        assert!((pt.1 - 0.0).abs() < 1e-9);
        assert!((dist_sq - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_is_point_in_rect() {
        assert!(is_point_inside_rect((2.0, 2.0), (0.0, 0.0, 4.0, 4.0)));
        assert!(!is_point_inside_rect((5.0, 5.0), (0.0, 0.0, 4.0, 4.0)));
    }

    #[test]
    fn test_rect_a_contains_rect_b() {
        let rect_a: Rect = (0.0, 0.0, 10.0, 10.0);
        let rect_b: Rect = (2.0, 2.0, 8.0, 8.0);
        assert!(does_rect_contain_rect(rect_a, rect_b));
        assert!(!does_rect_contain_rect(rect_b, rect_a));
    }

    #[test]
    fn test_does_rect_intersect_rect() {
        let rect_a: Rect = (0.0, 0.0, 4.0, 4.0);
        let rect_b: Rect = (2.0, 2.0, 6.0, 6.0);
        assert!(does_rect_intersect_rect(rect_a, rect_b));

        let rect_c: Rect = (10.0, 10.0, 12.0, 12.0);
        assert!(!does_rect_intersect_rect(rect_a, rect_c));
    }

    #[test]
    fn test_line_segment_intersects_rect() {
        assert!(does_line_segment_intersect_rect(
            (2.0, 2.0),
            (6.0, 6.0),
            (0.0, 0.0, 4.0, 4.0)
        ));
        assert!(does_line_segment_intersect_rect(
            (2.0, 2.0),
            (2.0, 6.0),
            (0.0, 0.0, 4.0, 4.0)
        ));
        assert!(!does_line_segment_intersect_rect(
            (5.0, 5.0),
            (6.0, 6.0),
            (0.0, 0.0, 4.0, 4.0)
        ));
    }

    #[test]
    fn test_line_segment_intersects_circle() {
        assert!(does_line_segment_intersect_circle(
            (0.0, 0.0),
            (2.0, 0.0),
            (1.0, 0.0),
            0.5
        ));
        assert!(does_line_segment_intersect_circle(
            (0.0, 0.0),
            (1.0, 1.0),
            (0.5, 0.5),
            0.3
        ));
        assert!(!does_line_segment_intersect_circle(
            (0.0, 0.0),
            (1.0, 0.0),
            (5.0, 5.0),
            0.5
        ));
    }

    #[test]
    fn test_point_line_distance() {
        let d = get_point_line_distance((0.0, 1.0), (0.0, 0.0), (1.0, 0.0));
        assert!((d - 1.0).abs() < 1e-9);

        let d = get_point_line_distance((0.5, 0.0), (0.0, 0.0), (1.0, 0.0));
        assert!((d - 0.0).abs() < 1e-9);

        let d = get_point_line_distance((1.0, 1.0), (0.0, 0.0), (0.0, 0.0));
        assert!((d - 2.0f64.sqrt()).abs() < 1e-9);
    }
}
