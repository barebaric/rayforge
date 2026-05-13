//! Circle: Circle geometry operations.
//!
//! This module provides functions for working with circles including:
//! - Circle-circle intersection
//! - Point projection onto circle
//! - Circle-rectangle containment and intersection
//! - Line segment intersection with circles

use crate::shape::line::get_line_segment_closest_point;
use crate::types::{Point, Rect};

/// Computes the intersection points between two circles.
/// Uses the geometry of intersecting circles to find 0, 1, or 2 points.
pub fn get_circle_circle_intersections(
    c1: Point,
    r1: f64,
    c2: Point,
    r2: f64,
) -> Vec<Point> {
    let dx = c2.0 - c1.0;
    let dy = c2.1 - c1.1;
    let d_sq = dx * dx + dy * dy;
    let d = d_sq.sqrt();

    if d < 1e-9 || d > r1 + r2 || d < (r1 - r2).abs() {
        return vec![];
    }

    let a = (r1 * r1 - r2 * r2 + d_sq) / (2.0 * d);
    let h_sq = (r1 * r1 - a * a).max(0.0);
    let h = h_sq.sqrt();

    let x2 = c1.0 + a * dx / d;
    let y2 = c1.1 + a * dy / d;

    let x3_1 = x2 + h * dy / d;
    let y3_1 = y2 - h * dx / d;
    let x3_2 = x2 - h * dy / d;
    let y3_2 = y2 + h * dx / d;

    vec![(x3_1, y3_1), (x3_2, y3_2)]
}

/// Projects a point onto a circle's circumference.
/// Returns None if the point is at the circle's center (direction undefined).
pub fn project_point_onto_circle(
    point: Point,
    center: Point,
    radius: f64,
) -> Option<Point> {
    let dx = point.0 - center.0;
    let dy = point.1 - center.1;
    let dist = (dx * dx + dy * dy).sqrt();

    if dist < 1e-9 {
        return None;
    }

    let scale = radius / dist;
    Some((center.0 + dx * scale, center.1 + dy * scale))
}

/// Tests if a circle is completely contained within an axis-aligned rectangle.
pub fn is_circle_inside_rect(center: Point, radius: f64, rect: Rect) -> bool {
    let (cx, cy) = center;
    let (rx1, ry1, rx2, ry2) = rect;
    (cx - radius) >= rx1
        && (cy - radius) >= ry1
        && (cx + radius) <= rx2
        && (cy + radius) <= ry2
}

/// Tests if a circle intersects an axis-aligned rectangle.
/// Uses multiple tests: containment check, closest point check, farthest point check.
pub fn does_circle_intersect_rect(
    center: Point,
    radius: f64,
    rect: Rect,
) -> bool {
    let (cx, cy) = center;
    let (rx1, ry1, rx2, ry2) = rect;

    // Fully contained circles don't intersect
    if is_circle_inside_rect(center, radius, rect) {
        return false;
    }

    // Check if circle's closest point to rect is within radius
    let closest_x = rx1.max(cx.min(rx2));
    let closest_y = ry1.max(cy.min(ry2));
    let dist_sq_closest = (closest_x - cx).powi(2) + (closest_y - cy).powi(2);
    if dist_sq_closest > radius * radius {
        return false;
    }

    // Check if circle's farthest point from rect center is outside radius
    let dx_far = (rx1 - cx).abs().max((rx2 - cx).abs());
    let dy_far = (ry1 - cy).abs().max((ry2 - cy).abs());
    let dist_sq_farthest = dx_far * dx_far + dy_far * dy_far;
    if dist_sq_farthest < radius * radius {
        return false;
    }

    true
}

/// Tests if a line segment intersects a circle by checking closest point distance.
pub fn line_segment_intersects_circle(
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
    fn test_get_circle_circle_intersections_no_intersection() {
        let result =
            get_circle_circle_intersections((0.0, 0.0), 1.0, (10.0, 10.0), 1.0);
        assert!(result.is_empty());
    }

    #[test]
    fn test_get_circle_circle_intersections_one_point() {
        let result =
            get_circle_circle_intersections((0.0, 0.0), 1.0, (2.0, 0.0), 1.0);
        assert_eq!(result.len(), 1);
    }

    #[test]
    fn test_get_circle_circle_intersections_two_points() {
        let result =
            get_circle_circle_intersections((0.0, 0.0), 2.0, (3.0, 0.0), 2.0);
        assert_eq!(result.len(), 2);
    }

    #[test]
    fn test_project_point_onto_circle() {
        let result = project_point_onto_circle((2.0, 0.0), (0.0, 0.0), 1.0);
        assert!(result.is_some());
        let pt = result.unwrap();
        assert!((pt.0 - 1.0).abs() < 1e-9);
        assert!(pt.1.abs() < 1e-9);
    }

    #[test]
    fn test_project_point_onto_circle_at_center() {
        let result = project_point_onto_circle((0.0, 0.0), (0.0, 0.0), 1.0);
        assert!(result.is_none());
    }

    #[test]
    fn test_is_circle_inside_rect() {
        let rect: Rect = (0.0, 0.0, 10.0, 10.0);
        assert!(is_circle_inside_rect((5.0, 5.0), 2.0, rect));
        assert!(!is_circle_inside_rect((5.0, 5.0), 6.0, rect));
    }

    #[test]
    fn test_does_circle_intersect_rect() {
        let rect: Rect = (0.0, 0.0, 10.0, 10.0);
        assert!(does_circle_intersect_rect((8.0, 5.0), 3.0, rect));
        assert!(!does_circle_intersect_rect((15.0, 15.0), 2.0, rect));
        assert!(does_circle_intersect_rect((10.0, 5.0), 3.0, rect));
    }

    #[test]
    fn test_line_segment_intersects_circle() {
        assert!(line_segment_intersects_circle(
            (0.0, 0.0),
            (2.0, 0.0),
            (1.0, 0.0),
            0.5
        ));
        assert!(line_segment_intersects_circle(
            (0.0, 0.0),
            (1.0, 1.0),
            (0.5, 0.5),
            0.3
        ));
        assert!(!line_segment_intersects_circle(
            (0.0, 0.0),
            (1.0, 0.0),
            (5.0, 5.0),
            0.5
        ));
    }
}
