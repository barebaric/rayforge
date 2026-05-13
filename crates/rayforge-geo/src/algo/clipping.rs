//! Clipping: Line segment clipping and region operations.

use crate::shape::line::get_line_segment_polygon_intersections;
use crate::shape::polygon::is_point_inside_polygon;
use crate::types::{Point, Point3D, Polygon, Rect};

const INSIDE: i32 = 0;
const LEFT: i32 = 1;
const RIGHT: i32 = 2;
const BOTTOM: i32 = 4;
const TOP: i32 = 8;

fn compute_outcode(x: f64, y: f64, rect: Rect) -> i32 {
    let (x_min, y_min, x_max, y_max) = rect;
    let mut code = INSIDE;
    if x < x_min {
        code |= LEFT;
    } else if x > x_max {
        code |= RIGHT;
    }
    if y < y_min {
        code |= BOTTOM;
    } else if y > y_max {
        code |= TOP;
    }
    code
}

/// Clips a 3D line segment to an axis-aligned 2D rectangle using
/// the Cohen-Sutherland algorithm. Z-coordinates are interpolated.
pub fn clip_line_segment_with_rect(
    p1: Point3D,
    p2: Point3D,
    rect: Rect,
) -> Option<(Point3D, Point3D)> {
    let (x_min, y_min, x_max, y_max) = rect;
    let (mut x1, mut y1, mut z1) = p1;
    let (mut x2, mut y2, mut z2) = p2;
    let (dx, dy, dz) = (x2 - x1, y2 - y1, z2 - z1);

    let mut outcode1 = compute_outcode(x1, y1, rect);
    let mut outcode2 = compute_outcode(x2, y2, rect);

    loop {
        if (outcode1 | outcode2) == 0 {
            return Some(((x1, y1, z1), (x2, y2, z2)));
        }
        if (outcode1 & outcode2) != 0 {
            return None;
        }

        let outcode_out = if outcode1 != 0 { outcode1 } else { outcode2 };
        let (mut x, mut y, mut z) = (0.0, 0.0, 0.0);

        if (outcode_out & TOP) != 0 {
            y = y_max;
            x = if dy != 0.0 {
                x1 + dx * (y_max - y1) / dy
            } else {
                x1
            };
            z = if dy != 0.0 {
                z1 + dz * (y_max - y1) / dy
            } else {
                z1
            };
        } else if (outcode_out & BOTTOM) != 0 {
            y = y_min;
            x = if dy != 0.0 {
                x1 + dx * (y_min - y1) / dy
            } else {
                x1
            };
            z = if dy != 0.0 {
                z1 + dz * (y_min - y1) / dy
            } else {
                z1
            };
        } else if (outcode_out & RIGHT) != 0 {
            x = x_max;
            y = if dx != 0.0 {
                y1 + dy * (x_max - x1) / dx
            } else {
                y1
            };
            z = if dx != 0.0 {
                z1 + dz * (x_max - x1) / dx
            } else {
                z1
            };
        } else if (outcode_out & LEFT) != 0 {
            x = x_min;
            y = if dx != 0.0 {
                y1 + dy * (x_min - x1) / dx
            } else {
                y1
            };
            z = if dx != 0.0 {
                z1 + dz * (x_min - x1) / dx
            } else {
                z1
            };
        }

        if outcode_out == outcode1 {
            x1 = x;
            y1 = y;
            z1 = z;
            outcode1 = compute_outcode(x1, y1, rect);
        } else {
            x2 = x;
            y2 = y;
            z2 = z;
            outcode2 = compute_outcode(x2, y2, rect);
        }
    }
}

/// Calculates the sub-segments of a line that lie outside a list of polygons.
pub fn subtract_polygons_from_line_segment(
    p1: Point3D,
    p2: Point3D,
    regions: &[Polygon],
) -> Vec<(Point3D, Point3D)> {
    let p1_2d: Point = (p1.0, p1.1);
    let p2_2d: Point = (p2.0, p2.1);
    let sorted_cuts =
        get_line_segment_polygon_intersections(p1_2d, p2_2d, regions);

    let mut kept_segments: Vec<(Point3D, Point3D)> = Vec::new();

    for i in 0..sorted_cuts.len().saturating_sub(1) {
        let t1 = sorted_cuts[i];
        let t2 = sorted_cuts[i + 1];
        if (t1 - t2).abs() < 1e-9 {
            continue;
        }

        let mid_t = (t1 + t2) / 2.0;
        let mid_p: Point =
            (p1.0 + (p2.0 - p1.0) * mid_t, p1.1 + (p2.1 - p1.1) * mid_t);

        let is_inside_any =
            regions.iter().any(|r| is_point_inside_polygon(mid_p, r));

        if !is_inside_any {
            let sub_p1: Point3D = (
                p1.0 + (p2.0 - p1.0) * t1,
                p1.1 + (p2.1 - p1.1) * t1,
                p1.2 + (p2.2 - p1.2) * t1,
            );
            let sub_p2: Point3D = (
                p1.0 + (p2.0 - p1.0) * t2,
                p1.1 + (p2.1 - p1.1) * t2,
                p1.2 + (p2.2 - p1.2) * t2,
            );
            kept_segments.push((sub_p1, sub_p2));
        }
    }

    kept_segments
}

/// Returns the sub-segments of a line segment that lie inside a list of polygons.
/// This is the inverse of subtract_polygons_from_line_segment.
pub fn clip_line_segment_with_polygons(
    p1: Point3D,
    p2: Point3D,
    regions: &[Polygon],
) -> Vec<(Point3D, Point3D)> {
    if regions.is_empty() {
        return Vec::new();
    }

    let p1_2d: Point = (p1.0, p1.1);
    let p2_2d: Point = (p2.0, p2.1);
    let sorted_cuts =
        get_line_segment_polygon_intersections(p1_2d, p2_2d, regions);

    let mut kept_segments: Vec<(Point3D, Point3D)> = Vec::new();

    for i in 0..sorted_cuts.len().saturating_sub(1) {
        let t1 = sorted_cuts[i];
        let t2 = sorted_cuts[i + 1];
        if (t1 - t2).abs() < 1e-9 {
            continue;
        }

        let mid_t = (t1 + t2) / 2.0;
        let mid_p: Point =
            (p1.0 + (p2.0 - p1.0) * mid_t, p1.1 + (p2.1 - p1.1) * mid_t);

        let is_inside_any =
            regions.iter().any(|r| is_point_inside_polygon(mid_p, r));

        if is_inside_any {
            let sub_p1: Point3D = (
                p1.0 + (p2.0 - p1.0) * t1,
                p1.1 + (p2.1 - p1.1) * t1,
                p1.2 + (p2.2 - p1.2) * t1,
            );
            let sub_p2: Point3D = (
                p1.0 + (p2.0 - p1.0) * t2,
                p1.1 + (p2.1 - p1.1) * t2,
                p1.2 + (p2.2 - p1.2) * t2,
            );
            kept_segments.push((sub_p1, sub_p2));
        }
    }

    kept_segments
}
