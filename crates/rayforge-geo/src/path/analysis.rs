//! Analysis: Path analysis and geometry metrics.
//!
//! This module provides functions for analyzing geometric paths including:
//! - Closed path detection
//! - Area calculation using the shoelace formula
//! - Winding order determination
//! - Point and tangent computation along paths
//! - Outward normal computation

use std::f64::consts::PI;

use crate::path::geometry::Geometry;
use crate::path::split::{get_valid_contours_data, split_into_contours};
use crate::shape::arc::linearize_arc;
use crate::shape::bezier::linearize_bezier_from_array;
use crate::shape::polygon::is_point_inside_polygon;
use crate::types::{Command, Point, Point3D, Polygon};

/// Checks if a path forms a closed loop within the given tolerance.
/// A closed path starts and ends at the same point.
pub fn is_closed(commands: &[[f64; 8]], tolerance: f64) -> bool {
    if commands.len() < 2 {
        return false;
    }

    // Must start with a MOVE command
    let first = Command::from_row(&commands[0]).expect("invalid command");
    if !matches!(first, Command::Move { .. }) {
        return false;
    }

    let start_point = first.end_point();
    let last = Command::from_row(&commands[commands.len() - 1])
        .expect("invalid command");
    let end_point = last.end_point();

    // Check if start and end are within tolerance distance
    let dist_sq = (start_point.0 - end_point.0).powi(2)
        + (start_point.1 - end_point.1).powi(2)
        + (start_point.2 - end_point.2).powi(2);

    dist_sq < tolerance * tolerance
}

/// Extracts all vertices from a subpath starting at the given command index.
/// Linearizes arcs and Beziers into vertex sequences.
pub fn get_subpath_vertices_from_array(
    data: &[[f64; 8]],
    start_cmd_index: usize,
) -> Polygon {
    let mut vertices: Polygon = Vec::new();
    if start_cmd_index >= data.len() {
        return vertices;
    }

    let start_cmd =
        Command::from_row(&data[start_cmd_index]).expect("invalid command");
    let last_pos_3d = start_cmd.end_point();
    vertices.push((last_pos_3d.0, last_pos_3d.1));

    for row in data.iter().skip(start_cmd_index + 1) {
        let cmd = Command::from_row(row).expect("invalid command");

        if matches!(cmd, Command::Move { .. }) {
            break;
        }

        let end_point_3d = cmd.end_point();

        match &cmd {
            Command::Line { .. } => {
                vertices.push((end_point_3d.0, end_point_3d.1));
            }
            Command::Arc { .. } => {
                let start_3d: Point3D = if vertices.len() >= 2 {
                    (
                        vertices[vertices.len() - 1].0,
                        vertices[vertices.len() - 1].1,
                        last_pos_3d.2,
                    )
                } else {
                    last_pos_3d
                };
                let segments = linearize_arc(row, start_3d, 0.1);
                for (_, p2) in segments {
                    vertices.push((p2.0, p2.1));
                }
            }
            Command::Bezier { .. } => {
                let start_3d: Point3D = if vertices.len() >= 2 {
                    (
                        vertices[vertices.len() - 1].0,
                        vertices[vertices.len() - 1].1,
                        last_pos_3d.2,
                    )
                } else {
                    last_pos_3d
                };
                let segments = linearize_bezier_from_array(row, start_3d, 0.1);
                for (_, p2) in segments {
                    vertices.push((p2.0, p2.1));
                }
            }
            _ => {}
        }
    }

    vertices
}

/// Computes the signed area of a subpath using the shoelace formula.
/// Positive area indicates counter-clockwise (CCW), negative indicates clockwise (CW).
pub fn get_subpath_area_from_array(
    data: &[[f64; 8]],
    start_cmd_index: usize,
) -> f64 {
    let vertices = get_subpath_vertices_from_array(data, start_cmd_index);
    if vertices.len() < 3 {
        return 0.0;
    }

    // Must be a closed polygon (start = end)
    let p_start = vertices[0];
    let p_end = vertices[vertices.len() - 1];

    if (p_start.0 - p_end.0).abs() >= 1e-9
        || (p_start.1 - p_end.1).abs() >= 1e-9
    {
        return 0.0;
    }

    // Shoelace formula: sum(x_i * y_{i+1} - x_{i+1} * y_i) / 2
    let mut area = 0.0;
    for i in 0..vertices.len() - 1 {
        let x = vertices[i].0;
        let y_shifted = vertices[i + 1].1;
        let y = vertices[i].1;
        let x_shifted = vertices[i + 1].0;
        area += x * y_shifted - x_shifted * y;
    }

    area / 2.0
}

/// Computes the total area enclosed by the geometry, summing all subpaths.
/// Returns the absolute value (unsigned area).
pub fn get_area_from_array(data: &[[f64; 8]]) -> f64 {
    if data.is_empty() {
        return 0.0;
    }

    // Validate that path starts with MOVE
    let first = Command::from_row(&data[0]).expect("invalid command");
    if !matches!(first, Command::Move { .. }) {
        return 0.0;
    }

    let mut total_signed_area = 0.0;

    // Find all MOVE command indices (start of each subpath)
    let mut move_indices: Vec<usize> = Vec::new();
    for (i, row) in data.iter().enumerate() {
        let cmd = Command::from_row(row).expect("invalid command");
        if matches!(cmd, Command::Move { .. }) {
            move_indices.push(i);
        }
    }

    // Sum area from each subpath
    for &i in &move_indices {
        total_signed_area += get_subpath_area_from_array(data, i);
    }

    total_signed_area.abs()
}

/// Determines the winding order of a subpath based on signed area.
/// Returns "ccw" for counter-clockwise, "cw" for clockwise, or "unknown" if degenerate.
pub fn get_path_winding_order_from_array(
    data: &[[f64; 8]],
    start_cmd_index: usize,
) -> &'static str {
    let area = get_subpath_area_from_array(data, start_cmd_index);

    if area.abs() < 1e-9 {
        "unknown"
    } else if area > 0.0 {
        "ccw"
    } else {
        "cw"
    }
}

/// Evaluates a point and its tangent vector at a given t parameter along a path segment.
/// Returns None for MOVE commands or invalid indices.
pub fn get_point_and_tangent_at_from_array(
    data: &[[f64; 8]],
    row_index: usize,
    t: f64,
) -> Option<(Point, Point)> {
    if row_index >= data.len() {
        return None;
    }

    let row = data[row_index];
    let cmd = Command::from_row(&row).expect("invalid command");

    let start_pos_3d: Point3D = if row_index > 0 {
        Command::from_row(&data[row_index - 1])
            .expect("invalid command")
            .end_point()
    } else {
        (0.0, 0.0, 0.0)
    };

    let p0 = (start_pos_3d.0, start_pos_3d.1);
    let end_3d = cmd.end_point();
    let p1 = (end_3d.0, end_3d.1);

    let (point, tangent_vec) = match &cmd {
        Command::Line { .. } => {
            let point = (p0.0 + t * (p1.0 - p0.0), p0.1 + t * (p1.1 - p0.1));
            let tangent_vec = (p1.0 - p0.0, p1.1 - p0.1);
            (point, tangent_vec)
        }
        Command::Arc {
            center_offset,
            clockwise,
            ..
        } => {
            let center = (p0.0 + center_offset.0, p0.1 + center_offset.1);

            let start_angle = (p0.1 - center.1).atan2(p0.0 - center.0);
            let end_angle = (p1.1 - center.1).atan2(p1.0 - center.0);
            let mut angle_range = end_angle - start_angle;

            if *clockwise {
                if angle_range > 0.0 {
                    angle_range -= 2.0 * PI;
                }
            } else {
                if angle_range < 0.0 {
                    angle_range += 2.0 * PI;
                }
            }

            let current_angle = start_angle + t * angle_range;
            let radius_start = (p0.0 - center.0).hypot(p0.1 - center.1);
            let radius_end = (p1.0 - center.0).hypot(p1.1 - center.1);
            let radius = radius_start + t * (radius_end - radius_start);

            let point = (
                center.0 + radius * current_angle.cos(),
                center.1 + radius * current_angle.sin(),
            );

            let radius_vec = (point.0 - center.0, point.1 - center.1);
            let tangent_vec = if *clockwise {
                (radius_vec.1, -radius_vec.0)
            } else {
                (-radius_vec.1, radius_vec.0)
            };

            (point, tangent_vec)
        }
        Command::Bezier {
            control1, control2, ..
        } => {
            let c1 = *control1;
            let c2 = *control2;

            let omt = 1.0 - t;
            let p_x = omt.powi(3) * p0.0
                + 3.0 * omt.powi(2) * t * c1.0
                + 3.0 * omt * t.powi(2) * c2.0
                + t.powi(3) * p1.0;
            let p_y = omt.powi(3) * p0.1
                + 3.0 * omt.powi(2) * t * c1.1
                + 3.0 * omt * t.powi(2) * c2.1
                + t.powi(3) * p1.1;

            let point = (p_x, p_y);

            let tx = 3.0 * omt.powi(2) * (c1.0 - p0.0)
                + 6.0 * omt * t * (c2.0 - c1.0)
                + 3.0 * t.powi(2) * (p1.0 - c2.0);
            let ty = 3.0 * omt.powi(2) * (c1.1 - p0.1)
                + 6.0 * omt * t * (c2.1 - c1.1)
                + 3.0 * t.powi(2) * (p1.1 - c2.1);

            (point, (tx, ty))
        }
        _ => return None,
    };

    let norm = (tangent_vec.0.powi(2) + tangent_vec.1.powi(2)).sqrt();
    if norm < 1e-9 {
        return Some((point, (1.0, 0.0)));
    }

    let normalized_tangent = (tangent_vec.0 / norm, tangent_vec.1 / norm);
    Some((point, normalized_tangent))
}

/// Computes the outward-facing normal vector at a point on the path.
/// The normal is perpendicular to the tangent, pointing outward from the path
/// based on the winding order (CCW = left of tangent, CW = right of tangent).
pub fn get_outward_normal_at_from_array(
    data: &[[f64; 8]],
    row_index: usize,
    t: f64,
) -> Option<Point> {
    // Find the start of the subpath containing this segment
    let mut subpath_start_index: isize = -1;
    for i in (0..=row_index).rev() {
        let cmd = Command::from_row(&data[i]).expect("invalid command");
        if matches!(cmd, Command::Move { .. }) {
            subpath_start_index = i as isize;
            break;
        }
    }
    if subpath_start_index == -1 {
        subpath_start_index = 0;
    }

    let winding =
        get_path_winding_order_from_array(data, subpath_start_index as usize);
    if winding == "unknown" {
        return None;
    }

    let result = get_point_and_tangent_at_from_array(data, row_index, t)?;
    let (_, tangent) = result;
    let (tx, ty) = tangent;

    // Rotate tangent 90 degrees to get normal (direction depends on winding)
    if winding == "ccw" {
        Some((ty, -tx))
    } else {
        Some((-ty, tx))
    }
}

/// Computes the interior angle at a vertex formed by three points.
/// Returns the angle in radians between 0 and PI.
pub fn get_angle_at_vertex(p0: Point, p1: Point, p2: Point) -> f64 {
    let v1x = p0.0 - p1.0;
    let v1y = p0.1 - p1.1;
    let v2x = p2.0 - p1.0;
    let v2y = p2.1 - p1.1;

    let mag_v1 = v1x.hypot(v1y);
    let mag_v2 = v2x.hypot(v2y);
    let mag_prod = mag_v1 * mag_v2;

    // Degenerate case: collinear points
    if mag_prod < 1e-9 {
        return PI;
    }

    let dot = v1x * v2x + v1y * v2y;
    // Clamp to handle numerical errors
    let cos_theta = (-1.0_f64).max(1.0_f64).min(dot / mag_prod);

    cos_theta.acos()
}

pub fn remove_duplicates<T: Clone + PartialEq>(points: &[T]) -> Vec<T> {
    let mut result: Vec<T> = Vec::new();
    for p in points {
        if result.is_empty() || !result.contains(p) {
            result.push(p.clone());
        }
    }
    result
}

/// Determines if a polygon is wound in clockwise order using the cross product.
/// Uses only the first three points to determine overall winding direction.
pub fn is_clockwise(points: &[Point]) -> bool {
    if points.len() < 3 {
        return false;
    }

    let p1 = points[0];
    let p2 = points[1];
    let p3 = points[2];

    // Cross product of first edge against second determines winding
    let cross_product =
        (p2.0 - p1.0) * (p3.1 - p2.1) - (p2.1 - p1.1) * (p3.0 - p2.0);
    cross_product < 0.0
}

/// Determines if points along an arc traverse in clockwise direction relative to center.
/// Uses cumulative cross product of successive radius vectors.
pub fn is_arc_clockwise(points: &[Point], center: Point) -> bool {
    let (xc, yc) = center;
    let mut cross_product_sum = 0.0;

    for i in 0..points.len() - 1 {
        let (x0, y0) = points[i];
        let (x1, y1) = points[i + 1];
        // Vector from center to each point
        let v0x = x0 - xc;
        let v0y = y0 - yc;
        let v1x = x1 - xc;
        let v1y = y1 - yc;
        // Accumulate cross products
        cross_product_sum += v0x * v1y - v0y * v1x;
    }

    cross_product_sum < 0.0
}

/// Check if a container geometry fully encloses a content geometry.
pub fn does_enclose(container: &Geometry, content: &Geometry) -> bool {
    if container.is_empty() || content.is_empty() {
        return false;
    }

    let cont_rect = container.rect();
    let other_rect = content.rect();
    if !(cont_rect.0 <= other_rect.0
        && cont_rect.1 <= other_rect.1
        && cont_rect.2 >= other_rect.2
        && cont_rect.3 >= other_rect.3)
    {
        return false;
    }

    if container_intersects_content(container, content) {
        return false;
    }

    let other_segments = content.segments();
    if other_segments.is_empty() || other_segments[0].is_empty() {
        return false;
    }
    let test_point: Point = (other_segments[0][0].0, other_segments[0][0].1);

    let self_contours = split_into_contours(container);
    let all_contour_data = get_valid_contours_data(&self_contours);

    let mut winding_number = 0;
    for (geo, vertices, is_closed) in &all_contour_data {
        if !is_closed {
            continue;
        }
        let area = get_subpath_area_from_array(&geo.data, 0);
        if is_point_inside_polygon(test_point, vertices) {
            if area > 1e-9 {
                winding_number += 1;
            } else if area < -1e-9 {
                winding_number -= 1;
            }
        }
    }

    winding_number > 0
}

pub fn segment_length_from_row(row: &[f64; 8], start_point: Point3D) -> f64 {
    let cmd = Command::from_row(row).expect("invalid command");
    let sx = start_point.0;
    let sy = start_point.1;
    let (ex, ey, _) = cmd.end_point();

    match &cmd {
        Command::Move { .. } | Command::Line { .. } => {
            ((ex - sx).powi(2) + (ey - sy).powi(2)).sqrt()
        }
        Command::Arc {
            center_offset,
            clockwise,
            ..
        } => {
            let cx = sx + center_offset.0;
            let cy = sy + center_offset.1;
            let radius =
                (center_offset.0.powi(2) + center_offset.1.powi(2)).sqrt();
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
            angle_span.abs() * radius
        }
        Command::Bezier { .. } => {
            let segments = crate::bezier::linearize_bezier_from_array(
                row,
                start_point,
                0.1,
            );
            segments
                .iter()
                .map(|(p1, p2)| {
                    ((p2.0 - p1.0).powi(2) + (p2.1 - p1.1).powi(2)).sqrt()
                })
                .sum()
        }
    }
}

pub fn segment_length_from_row_flat(
    row: &[f64; 8],
    start_point: Point3D,
) -> f64 {
    let cmd = Command::from_row(row).expect("invalid command");
    let sx = start_point.0;
    let sy = start_point.1;
    let (ex, ey, _) = cmd.end_point();

    match &cmd {
        Command::Move { .. } | Command::Line { .. } => {
            ((ex - sx).powi(2) + (ey - sy).powi(2)).sqrt()
        }
        Command::Arc { .. } | Command::Bezier { .. } => {
            segment_length_from_row(row, start_point)
        }
    }
}

/// Computes a partial segment from a geometry row by interpolating at
/// parameter t along the segment. Returns None for MOVE commands.
pub fn partial_segment_from_row(
    row: &[f64; 8],
    start_point: Point3D,
    t: f64,
) -> Option<[f64; 8]> {
    let cmd = Command::from_row(row).expect("invalid command");
    let sx = start_point.0;
    let sy = start_point.1;
    let sz = start_point.2;
    let (ex, ey, ez) = cmd.end_point();

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
            let cw = *clockwise;
            let cx = sx + i_off;
            let cy = sy + j_off;
            let radius_start = i_off.hypot(j_off);
            let radius_end = (ex - cx).hypot(ey - cy);

            let start_angle = (sy - cy).atan2(sx - cx);
            let end_angle = (ey - cy).atan2(ex - cx);
            let mut angle_span = end_angle - start_angle;

            if angle_span.abs() < 1e-9 {
                angle_span = if cw { -2.0 * PI } else { 2.0 * PI };
            } else if cw {
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
                    center_offset: (i_off, j_off),
                    clockwise: cw,
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
        _ => None,
    }
}

fn container_intersects_content(
    container: &Geometry,
    content: &Geometry,
) -> bool {
    crate::intersect::check_intersection_from_array(
        &container.data,
        &content.data,
        false,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::constants::*;

    #[test]
    fn test_is_closed() {
        let data: [[f64; 8]; 4] = [
            [CMD_TYPE_MOVE, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [CMD_TYPE_LINE, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [CMD_TYPE_LINE, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [CMD_TYPE_LINE, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        ];
        assert!(is_closed(&data, 1e-6));
    }

    #[test]
    fn test_is_not_closed() {
        let data: [[f64; 8]; 3] = [
            [CMD_TYPE_MOVE, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [CMD_TYPE_LINE, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [CMD_TYPE_LINE, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        ];
        assert!(!is_closed(&data, 1e-6));
    }

    #[test]
    fn test_get_subpath_vertices() {
        let data: [[f64; 8]; 3] = [
            [CMD_TYPE_MOVE, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [CMD_TYPE_LINE, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [CMD_TYPE_LINE, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        ];
        let vertices = get_subpath_vertices_from_array(&data[..], 0);
        assert_eq!(vertices.len(), 3);
        assert_eq!(vertices[0], (0.0, 0.0));
    }

    #[test]
    fn test_get_subpath_area() {
        let data: [[f64; 8]; 4] = [
            [CMD_TYPE_MOVE, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [CMD_TYPE_LINE, 4.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [CMD_TYPE_LINE, 4.0, 4.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [CMD_TYPE_LINE, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        ];
        let area = get_subpath_area_from_array(&data[..], 0);
        assert!((area - 8.0).abs() < 1e-6);
    }

    #[test]
    fn test_get_area_from_array() {
        let data: [[f64; 8]; 4] = [
            [CMD_TYPE_MOVE, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [CMD_TYPE_LINE, 4.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [CMD_TYPE_LINE, 4.0, 4.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [CMD_TYPE_LINE, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        ];
        let area = get_area_from_array(&data[..]);
        assert!((area - 8.0).abs() < 1e-6);
    }

    #[test]
    fn test_get_path_winding_order() {
        let data: [[f64; 8]; 4] = [
            [CMD_TYPE_MOVE, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [CMD_TYPE_LINE, 4.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [CMD_TYPE_LINE, 4.0, 4.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [CMD_TYPE_LINE, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        ];
        let winding = get_path_winding_order_from_array(&data[..], 0);
        assert_eq!(winding, "ccw");
    }

    #[test]
    fn test_get_point_and_tangent_at_from_array() {
        let data: [[f64; 8]; 2] = [
            [CMD_TYPE_MOVE, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [CMD_TYPE_LINE, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        ];
        let result = get_point_and_tangent_at_from_array(&data[..], 1, 0.5);
        assert!(result.is_some());
        let (point, tangent) = result.unwrap();
        assert!((point.0 - 0.5).abs() < 1e-9);
        assert!((point.1 - 0.0).abs() < 1e-9);
        assert!((tangent.0 - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_get_outward_normal() {
        let data: [[f64; 8]; 4] = [
            [CMD_TYPE_MOVE, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [CMD_TYPE_LINE, 4.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [CMD_TYPE_LINE, 4.0, 4.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [CMD_TYPE_LINE, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        ];
        let normal = get_outward_normal_at_from_array(&data[..], 1, 0.5);
        assert!(normal.is_some());
    }

    #[test]
    fn test_get_angle_at_vertex() {
        let angle = get_angle_at_vertex((0.0, 0.0), (1.0, 0.0), (1.0, 1.0));
        assert!((angle - PI / 2.0).abs() < 1e-9);
    }

    #[test]
    fn test_remove_duplicates() {
        let points = vec![1, 2, 2, 3, 3, 3, 4];
        let result = remove_duplicates(&points);
        assert_eq!(result, vec![1, 2, 3, 4]);
    }

    #[test]
    fn test_is_clockwise() {
        let points: Vec<Point> = vec![(0.0, 0.0), (1.0, 0.0), (1.0, 1.0)];
        assert!(!is_clockwise(&points));
    }

    #[test]
    fn test_is_arc_clockwise() {
        let points: Vec<Point> = vec![(1.0, 0.0), (0.0, 1.0), (-1.0, 0.0)];
        let center: Point = (0.0, 0.0);
        assert!(!is_arc_clockwise(&points, center));
    }
}
