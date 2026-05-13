//! Query: Path queries and geometric analysis.
//!
//! This module provides functions for querying geometric paths including:
//! - Bounding box computation
//! - Total path distance calculation
//! - Closest point on path
//! - Bounding box intersection tests
//! - Path segment extraction

use std::f64::consts::PI;

use crate::shape::arc::{get_arc_bounds, get_arc_closest_point};
use crate::shape::bezier::{
    get_bezier_closest_point, linearize_bezier_from_array,
};
use crate::shape::line::get_line_segment_closest_point;
use crate::types::{Command, Point, Point3D, Rect};

/// Internal helper: computes min/max bounds for a 1D cubic Bezier curve.
/// Uses analytical solution to find extrema by solving the derivative polynomial.
fn _compute_cubic_bezier_bounds_1d(
    p0: &[f64],
    p1: &[f64],
    p2: &[f64],
    p3: &[f64],
) -> (Vec<f64>, Vec<f64>) {
    let n = p0.len();
    let mut local_min: Vec<f64> =
        p0.iter().zip(p3.iter()).map(|(a, b)| a.min(*b)).collect();
    let mut local_max: Vec<f64> =
        p0.iter().zip(p3.iter()).map(|(a, b)| a.max(*b)).collect();

    for i in 0..n {
        let p0i = p0[i];
        let p1i = p1[i];
        let p2i = p2[i];
        let p3i = p3[i];

        let a_coeff = 3.0 * (-p0i + 3.0 * p1i - 3.0 * p2i + p3i);
        let b_coeff = 6.0 * (p0i - 2.0 * p1i + p2i);
        let c_coeff = 3.0 * (p1i - p0i);

        let discriminant = b_coeff * b_coeff - 4.0 * a_coeff * c_coeff;

        if discriminant >= 0.0 && a_coeff.abs() > 1e-9 {
            let sqrt_disc = discriminant.sqrt();
            let t1 = (-b_coeff - sqrt_disc) / (2.0 * a_coeff);
            let t2 = (-b_coeff + sqrt_disc) / (2.0 * a_coeff);

            if t1 > 0.0 && t1 < 1.0 {
                let mt = 1.0 - t1;
                let val = mt.powi(3) * p0i
                    + 3.0 * mt.powi(2) * t1 * p1i
                    + 3.0 * mt * t1.powi(2) * p2i
                    + t1.powi(3) * p3i;
                local_min[i] = local_min[i].min(val);
                local_max[i] = local_max[i].max(val);
            }

            if t2 > 0.0 && t2 < 1.0 {
                let mt = 1.0 - t2;
                let val = mt.powi(3) * p0i
                    + 3.0 * mt.powi(2) * t2 * p1i
                    + 3.0 * mt * t2.powi(2) * p2i
                    + t2.powi(3) * p3i;
                local_min[i] = local_min[i].min(val);
                local_max[i] = local_max[i].max(val);
            }
        }

        if a_coeff.abs() <= 1e-9 && b_coeff.abs() > 1e-9 {
            let t = -c_coeff / b_coeff;
            if t > 0.0 && t < 1.0 {
                let mt = 1.0 - t;
                let val = mt.powi(3) * p0i
                    + 3.0 * mt.powi(2) * t * p1i
                    + 3.0 * mt * t.powi(2) * p2i
                    + t.powi(3) * p3i;
                local_min[i] = local_min[i].min(val);
                local_max[i] = local_max[i].max(val);
            }
        }
    }

    (local_min, local_max)
}

/// Computes the axis-aligned bounding rectangle for a geometry array.
/// Handles line, arc, and Bezier segments by computing their exact bounds.
pub fn get_bounding_rect_from_array(data: &[[f64; 8]]) -> Rect {
    if data.is_empty() {
        return (0.0, 0.0, 0.0, 0.0);
    }

    // First pass: compute bounds from all line endpoints
    let mut min_x = f64::INFINITY;
    let mut max_x = f64::NEG_INFINITY;
    let mut min_y = f64::INFINITY;
    let mut max_y = f64::NEG_INFINITY;

    for row in data {
        let cmd = Command::from_row(row).expect("invalid command");
        let end = cmd.end_point();
        if end.0 < min_x {
            min_x = end.0;
        }
        if end.0 > max_x {
            max_x = end.0;
        }
        if end.1 < min_y {
            min_y = end.1;
        }
        if end.1 > max_y {
            max_y = end.1;
        }
    }

    // Second pass: check arcs for potentially larger bounds
    let mut last_point_2d: Point = (0.0, 0.0);
    for row in data {
        let cmd = Command::from_row(row).expect("invalid command");
        let end = cmd.end_point();
        if let Command::Arc {
            center_offset,
            clockwise,
            ..
        } = &cmd
        {
            let (ax1, ay1, ax2, ay2) = get_arc_bounds(
                last_point_2d,
                (end.0, end.1),
                *center_offset,
                *clockwise,
            );

            if ax1 < min_x {
                min_x = ax1;
            }
            if ay1 < min_y {
                min_y = ay1;
            }
            if ax2 > max_x {
                max_x = ax2;
            }
            if ay2 > max_y {
                max_y = ay2;
            }
        }
        last_point_2d = (end.0, end.1);
    }

    // Third pass: compute Bezier curve extrema analytically
    let mut last_point_3d: Point3D = (0.0, 0.0, 0.0);
    for row in data {
        let cmd = Command::from_row(row).expect("invalid command");
        let end = cmd.end_point();
        if let Command::Bezier {
            control1, control2, ..
        } = &cmd
        {
            let p0_x = vec![last_point_3d.0];
            let p0_y = vec![last_point_3d.1];
            let p1_x = vec![control1.0];
            let p1_y = vec![control1.1];
            let p2_x = vec![control2.0];
            let p2_y = vec![control2.1];
            let p3_x = vec![end.0];
            let p3_y = vec![end.1];

            let (bx_min, bx_max) =
                _compute_cubic_bezier_bounds_1d(&p0_x, &p1_x, &p2_x, &p3_x);
            let (by_min, by_max) =
                _compute_cubic_bezier_bounds_1d(&p0_y, &p1_y, &p2_y, &p3_y);

            min_x = min_x.min(bx_min[0]);
            max_x = max_x.max(bx_max[0]);
            min_y = min_y.min(by_min[0]);
            max_y = max_y.max(by_max[0]);
        }
        last_point_3d = end;
    }

    (min_x, min_y, max_x, max_y)
}

/// Computes the total path length by summing segment lengths.
/// Handles lines (Euclidean distance), arcs (arc length), and Beziers (linearized).
pub fn get_total_distance_from_array(data: &[[f64; 8]]) -> f64 {
    let mut total_dist = 0.0;
    let mut last_point: Point3D = (0.0, 0.0, 0.0);

    for row in data {
        let cmd = Command::from_row(row).expect("invalid command");
        let end_point = cmd.end_point();

        match &cmd {
            Command::Move { .. } | Command::Line { .. } => {
                // Line segment: Euclidean distance
                total_dist += (end_point.0 - last_point.0)
                    .hypot(end_point.1 - last_point.1);
            }
            Command::Arc {
                center_offset,
                clockwise,
                ..
            } => {
                // Arc segment: arc length = radius * angle
                let center_x = last_point.0 + center_offset.0;
                let center_y = last_point.1 + center_offset.1;
                let radius = center_offset.0.hypot(center_offset.1);

                if radius > 1e-9 {
                    let start_angle = (last_point.1 - center_y)
                        .atan2(last_point.0 - center_x);
                    let end_angle =
                        (end_point.1 - center_y).atan2(end_point.0 - center_x);
                    let mut angle_span = end_angle - start_angle;

                    // Normalize to handle full circles
                    if *clockwise {
                        if angle_span > 1e-9 {
                            angle_span -= 2.0 * PI;
                        }
                    } else {
                        if angle_span < -1e-9 {
                            angle_span += 2.0 * PI;
                        }
                    }

                    total_dist += (angle_span * radius).abs();
                }
            }
            Command::Bezier { .. } => {
                // Bezier: linearize and sum segment lengths
                let segments =
                    linearize_bezier_from_array(row, last_point, 0.1);
                for (p1, p2) in segments {
                    total_dist += (p2.0 - p1.0).hypot(p2.1 - p1.1);
                }
            }
        }

        last_point = end_point;
    }

    total_dist
}

/// Finds the closest point on the path to a given (x, y) coordinate.
/// Returns (command_index, t_parameter, closest_point).
/// - command_index: Index of the command segment containing the closest point
/// - t_parameter: Position along the segment [0, 1]
/// - closest_point: The closest point on the path
pub fn find_closest_point_on_path_from_array(
    data: &[[f64; 8]],
    x: f64,
    y: f64,
) -> Option<(usize, f64, Point)> {
    let mut min_dist_sq = f64::INFINITY;
    let mut closest_info: Option<(usize, f64, Point)> = None;

    let mut last_pos_3d: Point3D = (0.0, 0.0, 0.0);

    for (i, row) in data.iter().enumerate() {
        let cmd = Command::from_row(row).expect("invalid command");
        let end_point_3d = cmd.end_point();

        if matches!(cmd, Command::Move { .. }) {
            last_pos_3d = end_point_3d;
            continue;
        }

        let start_pos_3d = last_pos_3d;

        match &cmd {
            Command::Line { .. } => {
                let t = get_line_segment_closest_point(
                    (start_pos_3d.0, start_pos_3d.1),
                    (end_point_3d.0, end_point_3d.1),
                    x,
                    y,
                );
                if t.2 < min_dist_sq {
                    min_dist_sq = t.2;
                    closest_info = Some((i, t.0, t.1));
                }
            }
            Command::Arc { .. } => {
                if let Some((t_arc, pt_arc, dist_sq_arc)) =
                    get_arc_closest_point(row, start_pos_3d, x, y)
                {
                    if dist_sq_arc < min_dist_sq {
                        min_dist_sq = dist_sq_arc;
                        closest_info = Some((i, t_arc, pt_arc));
                    }
                }
            }
            Command::Bezier { .. } => {
                if let Some((t_bezier, pt_bezier, dist_sq_bezier)) =
                    get_bezier_closest_point(row, start_pos_3d, x, y)
                {
                    if dist_sq_bezier < min_dist_sq {
                        min_dist_sq = dist_sq_bezier;
                        closest_info = Some((i, t_bezier, pt_bezier));
                    }
                }
            }
            Command::Move { .. } => {}
        }

        last_pos_3d = end_point_3d;
    }

    closest_info
}

fn _segment_length_from_row(row: &[f64; 8], start_point: Point3D) -> f64 {
    let cmd = Command::from_row(row).expect("invalid command");
    let sx = start_point.0;
    let sy = start_point.1;
    let end = cmd.end_point();
    let ex = end.0;
    let ey = end.1;

    match &cmd {
        Command::Move { .. } | Command::Line { .. } => (ex - sx).hypot(ey - sy),
        Command::Arc {
            center_offset,
            clockwise,
            ..
        } => {
            let i_off = center_offset.0;
            let j_off = center_offset.1;
            let cx = sx + i_off;
            let cy = sy + j_off;
            let radius = i_off.hypot(j_off);

            if radius < 1e-9 {
                return 0.0;
            }

            let start_angle = (sy - cy).atan2(sx - cx);
            let end_angle = (ey - cy).atan2(ex - cx);
            let mut angle_span = end_angle - start_angle;

            if angle_span.abs() < 1e-9 {
                angle_span = if *clockwise { -2.0 * PI } else { 2.0 * PI };
            } else if *clockwise {
                if angle_span > 1e-9 {
                    angle_span -= 2.0 * PI;
                }
            } else {
                if angle_span < -1e-9 {
                    angle_span += 2.0 * PI;
                }
            }

            (angle_span * radius).abs()
        }
        Command::Bezier { .. } => {
            let segments = linearize_bezier_from_array(row, start_point, 0.1);
            segments
                .iter()
                .map(|(p1, p2)| (p2.0 - p1.0).hypot(p2.1 - p1.1))
                .sum()
        }
    }
}

fn _partial_segment_from_row(
    row: &[f64; 8],
    start_point: Point3D,
    t: f64,
) -> Option<[f64; 8]> {
    let cmd = Command::from_row(row).expect("invalid command");
    let sx = start_point.0;
    let sy = start_point.1;
    let sz = start_point.2;
    let end = cmd.end_point();
    let ex = end.0;
    let ey = end.1;
    let ez = end.2;

    match &cmd {
        Command::Line { .. } => {
            let nx = sx + t * (ex - sx);
            let ny = sy + t * (ey - sy);
            let nz = sz + t * (ez - sz);
            Some(Command::Line { end: (nx, ny, nz) }.to_row())
        }
        Command::Arc {
            center_offset,
            clockwise,
            ..
        } => {
            let i_off = center_offset.0;
            let j_off = center_offset.1;
            let cx = sx + i_off;
            let cy = sy + j_off;
            let radius_start = i_off.hypot(j_off);
            let radius_end = (ex - cx).hypot(ey - cy);

            let start_angle = (sy - cy).atan2(sx - cx);
            let end_angle = (ey - cy).atan2(ex - cx);
            let mut angle_span = end_angle - start_angle;

            if angle_span.abs() < 1e-9 {
                angle_span = if *clockwise { -2.0 * PI } else { 2.0 * PI };
            } else if *clockwise {
                if angle_span > 1e-9 {
                    angle_span -= 2.0 * PI;
                }
            } else {
                if angle_span < -1e-9 {
                    angle_span += 2.0 * PI;
                }
            }

            let mid_angle = start_angle + t * angle_span;
            let radius = radius_start + t * (radius_end - radius_start);
            let nx = cx + radius * mid_angle.cos();
            let ny = cy + radius * mid_angle.sin();
            let nz = sz + t * (ez - sz);

            Some(
                Command::Arc {
                    end: (nx, ny, nz),
                    center_offset: *center_offset,
                    clockwise: *clockwise,
                }
                .to_row(),
            )
        }
        Command::Bezier {
            control1, control2, ..
        } => {
            let c1x = control1.0;
            let c1y = control1.1;
            let c2x = control2.0;
            let c2y = control2.1;

            let p01x = sx + t * (c1x - sx);
            let p01y = sy + t * (c1y - sy);
            let p12x = c1x + t * (c2x - c1x);
            let p12y = c1y + t * (c2y - c1y);
            let p23x = c2x + t * (ex - c2x);
            let p23y = c2y + t * (ey - c2y);
            let p012x = p01x + t * (p12x - p01x);
            let p012y = p01y + t * (p12y - p01y);
            let p123x = p12x + t * (p23x - p12x);
            let p123y = p12y + t * (p23y - p12y);
            let p0123x = p012x + t * (p123x - p012x);
            let p0123y = p012y + t * (p123y - p012y);
            let nz = sz + t * (ez - sz);

            Some(
                Command::Bezier {
                    end: (p0123x, p0123y, nz),
                    control1: (p01x, p01y),
                    control2: (p012x, p012y),
                }
                .to_row(),
            )
        }
        Command::Move { .. } => None,
    }
}

/// Extracts path segments up to a maximum length for overcut operations.
/// Returns the commands that fall within max_length, including a partial command
/// at the boundary if needed. Used for tool path overcut calculations.
pub fn extract_overcut_rows(
    data: &[[f64; 8]],
    max_length: f64,
) -> Option<Vec<[f64; 8]>> {
    if data.len() < 2 || max_length <= 0.0 {
        return None;
    }

    let first_cmd = Command::from_row(&data[0]).expect("invalid command");
    let mut last_point: Point3D = first_cmd.end_point();
    let mut accumulated = 0.0;
    let mut collected: Vec<[f64; 8]> = Vec::new();

    for row in data.iter().skip(1) {
        let cmd = Command::from_row(row).expect("invalid command");
        let seg_length = _segment_length_from_row(row, last_point);
        if seg_length < 1e-9 {
            last_point = cmd.end_point();
            continue;
        }

        if accumulated + seg_length <= max_length + 1e-9 {
            collected.push(*row);
            accumulated += seg_length;
        } else {
            let remaining = max_length - accumulated;
            if remaining > 1e-9 {
                let t = remaining / seg_length;
                if let Some(partial) =
                    _partial_segment_from_row(row, last_point, t)
                {
                    collected.push(partial);
                }
            }
            break;
        }
        last_point = cmd.end_point();
    }

    if collected.is_empty() {
        None
    } else {
        Some(collected)
    }
}

/// Tests if two axis-aligned bounding boxes intersect using the separating axis theorem.
pub fn bboxes_intersect(bbox1: Rect, bbox2: Rect) -> bool {
    let (ax1, ay1, ax2, ay2) = bbox1;
    let (bx1, by1, bx2, by2) = bbox2;

    !(ax2 < bx1 || ax1 > bx2 || ay2 < by1 || ay1 > by2)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::constants::*;

    #[test]
    fn test_get_bounding_rect() {
        let data: [[f64; 8]; 3] = [
            [CMD_TYPE_MOVE, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [CMD_TYPE_LINE, 10.0, 5.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [CMD_TYPE_LINE, 10.0, 10.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        ];
        let rect = get_bounding_rect_from_array(&data[..]);
        assert_eq!(rect, (0.0, 0.0, 10.0, 10.0));
    }

    #[test]
    fn test_get_total_distance() {
        let data: [[f64; 8]; 3] = [
            [CMD_TYPE_MOVE, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [CMD_TYPE_LINE, 3.0, 4.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [CMD_TYPE_LINE, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        ];
        let dist = get_total_distance_from_array(&data[..]);
        assert!((dist - 10.0).abs() < 1e-6);
    }

    #[test]
    fn test_find_closest_point_on_path() {
        let data: [[f64; 8]; 2] = [
            [CMD_TYPE_MOVE, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [CMD_TYPE_LINE, 10.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        ];
        let result = find_closest_point_on_path_from_array(&data[..], 5.0, 1.0);
        assert!(result.is_some());
        let (idx, _t, pt) = result.unwrap();
        assert_eq!(idx, 1);
        assert!((pt.0 - 5.0).abs() < 1e-9);
        assert!((pt.1 - 0.0).abs() < 1e-9);
    }

    #[test]
    fn test_extract_overcut_rows() {
        let data: [[f64; 8]; 4] = [
            [CMD_TYPE_MOVE, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [CMD_TYPE_LINE, 5.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [CMD_TYPE_LINE, 10.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [CMD_TYPE_LINE, 15.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        ];
        let result = extract_overcut_rows(&data[..], 7.5);
        assert!(result.is_some());
        let rows = result.unwrap();
        assert_eq!(rows.len(), 2);
    }

    #[test]
    fn test_bboxes_intersect() {
        let bbox1: Rect = (0.0, 0.0, 5.0, 5.0);
        let bbox2: Rect = (3.0, 3.0, 8.0, 8.0);
        assert!(bboxes_intersect(bbox1, bbox2));

        let bbox3: Rect = (10.0, 10.0, 15.0, 15.0);
        assert!(!bboxes_intersect(bbox1, bbox3));
    }
}
