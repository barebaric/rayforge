//! Transform: Geometry offsetting and affine transformation.
//!
//! This module provides functions for offsetting (growing/shrinking) geometry
//! and applying affine transformations. The offsetting uses polygon boolean
//! operations to handle holes correctly.

use crate::path::geometry::Geometry;
use crate::path::intersect::check_intersection_from_array;
use crate::path::split::{get_valid_contours_data, split_into_contours};
use crate::shape::arc::linearize_arc;
use crate::shape::polygon::is_point_inside_polygon;
use crate::shape::polygon::{get_polygons_difference, offset_polygon};
use crate::types::{Command, Point, Point3D, Polygon, Rect};

const CLIPPER_SCALE: i64 = 10_000_000;

#[derive(Clone, Debug)]
struct ContourItem {
    geo: Geometry,
    verts: Vec<Point>,
    path: Vec<(i64, i64)>,
    rect: Rect,
    area: f64,
    #[allow(dead_code)]
    id: usize,
}

fn prepare_contour_items(
    contour_data: &[(Geometry, Vec<Point>, bool)],
) -> Vec<ContourItem> {
    let mut items = Vec::new();
    for (i, (geo, vertices, _is_closed)) in contour_data.iter().enumerate() {
        if vertices.len() < 2 {
            continue;
        }
        let mut verts = vertices.clone();
        let first = verts[0];
        let last = verts[verts.len() - 1];
        if (first.0 - last.0).abs() < 1e-9 && (first.1 - last.1).abs() < 1e-9 {
            verts.pop();
        }
        if verts.len() < 3 {
            continue;
        }
        let mut area = 0.0;
        let n = verts.len();
        for j in 0..n {
            let k = (j + 1) % n;
            area += verts[j].0 * verts[k].1;
            area -= verts[k].0 * verts[j].1;
        }
        area = area.abs() / 2.0;
        let scaled_path: Vec<(i64, i64)> = verts
            .iter()
            .map(|v| {
                (
                    (v.0 * CLIPPER_SCALE as f64) as i64,
                    (v.1 * CLIPPER_SCALE as f64) as i64,
                )
            })
            .collect();
        let rect = geo.rect();
        items.push(ContourItem {
            geo: geo.clone(),
            verts,
            path: scaled_path,
            rect,
            area,
            id: i,
        });
    }
    items
}

fn build_containment_hierarchy(items: &[ContourItem]) -> Vec<isize> {
    let n = items.len();
    let mut parent_map = vec![-1isize; n];

    for i in 0..n {
        let mut best_parent = -1isize;
        let mut best_parent_area = f64::INFINITY;

        for j in 0..n {
            if i == j {
                continue;
            }
            let r_i = &items[i].rect;
            let r_j = &items[j].rect;
            if !(r_j.0 <= r_i.0
                && r_j.1 <= r_i.1
                && r_j.2 >= r_i.2
                && r_j.3 >= r_i.3)
            {
                continue;
            }
            if items[j].area <= items[i].area {
                continue;
            }
            if check_intersection_from_array(
                &items[i].geo.data,
                &items[j].geo.data,
                false,
            ) {
                continue;
            }
            if !items[j].verts.is_empty()
                && is_point_inside_polygon(items[i].verts[0], &items[j].verts)
                && items[j].area < best_parent_area
            {
                best_parent_area = items[j].area;
                best_parent = j as isize;
            }
        }
        parent_map[i] = best_parent;
    }
    parent_map
}

fn calculate_nesting_depths(
    parent_map: &[isize],
    num_items: usize,
) -> Vec<i32> {
    let mut depths = vec![0i32; num_items];
    for i in 0..num_items {
        let mut d = 0;
        let mut curr = parent_map[i];
        let mut iterations = 0;
        while curr != -1 && iterations <= num_items {
            d += 1;
            curr = parent_map[curr as usize];
            iterations += 1;
        }
        depths[i] = d;
    }
    depths
}

fn group_solids_and_holes(
    depths: &[i32],
    parent_map: &[isize],
) -> std::collections::HashMap<usize, Vec<usize>> {
    let mut groups: std::collections::HashMap<usize, Vec<usize>> =
        std::collections::HashMap::new();
    for (i, &d) in depths.iter().enumerate() {
        if d % 2 == 0 {
            groups.entry(i).or_default();
        } else {
            let p = parent_map[i];
            if p != -1 {
                groups.entry(p as usize).or_default().push(i);
            }
        }
    }
    groups
}

fn offset_contour_group(
    solid_path: &[(i64, i64)],
    hole_paths: &[(i64, i64)],
    offset: f64,
) -> Vec<Polygon> {
    let scale = CLIPPER_SCALE as f64;
    let solid_poly: Polygon = solid_path
        .iter()
        .map(|(x, y)| (*x as f64 / scale, *y as f64 / scale))
        .collect();
    if solid_poly.len() < 3 {
        return vec![];
    }
    if hole_paths.is_empty() {
        return offset_polygon(&solid_poly, offset);
    }
    let mut hole_polys: Vec<Polygon> = Vec::new();
    let mut current_hole: Vec<Point> = Vec::new();
    let mut expected_len = 0usize;
    for &(x, y) in hole_paths {
        let px = x as f64 / scale;
        let py = y as f64 / scale;
        if current_hole.len() == expected_len {
            current_hole.push((px, py));
        } else {
            if (current_hole[0].0 - px).abs() < 1e-9
                && (current_hole[0].1 - py).abs() < 1e-9
            {
                current_hole.pop();
                if current_hole.len() >= 3 {
                    hole_polys.push(current_hole.clone());
                }
                current_hole.clear();
                expected_len = 0;
            } else {
                current_hole.push((px, py));
            }
        }
        expected_len += 1;
    }
    if !current_hole.is_empty() && current_hole.len() >= 3 {
        hole_polys.push(current_hole);
    }
    if hole_polys.is_empty() {
        return offset_polygon(&solid_poly, offset);
    }
    // Offset solid and holes separately, then subtract holes from solid.
    // For positive offset (grow): solid expands outward, hole contracts (inward).
    // For negative offset (shrink): solid contracts, hole expands.
    // The hole offset direction is always opposite to the solid offset.
    let offset_solids = offset_polygon(&solid_poly, offset);
    let mut final_polys = offset_solids;
    for hole in &hole_polys {
        let offset_holes = offset_polygon(hole, -offset);
        for offset_hole in &offset_holes {
            let mut new_result = Vec::new();
            for poly in final_polys.drain(..) {
                new_result.extend(get_polygons_difference(&poly, offset_hole));
            }
            final_polys = new_result;
        }
    }
    final_polys
}

/// Offsets the closed contours of a Geometry object by a given amount.
///
/// This function grows (positive offset) or shrinks (negative offset) the
/// area enclosed by closed paths.
///
/// This implementation processes logically distinct shapes (islands)
/// independently. Holes are associated with their enclosing solids and
/// offset together. Adjacent or overlapping solids remain separate, preserving
/// distinct toolpaths.
pub fn grow_geometry(geometry: &Geometry, offset: f64) -> Geometry {
    let raw_contours = split_into_contours(geometry);
    if raw_contours.is_empty() {
        return Geometry::new();
    }
    let contour_data = get_valid_contours_data(&raw_contours);
    if contour_data.is_empty() {
        return Geometry::new();
    }
    let closed_items = prepare_contour_items(&contour_data);
    if closed_items.is_empty() {
        return Geometry::new();
    }
    let parent_map = build_containment_hierarchy(&closed_items);
    let depths = calculate_nesting_depths(&parent_map, closed_items.len());
    let solid_groups = group_solids_and_holes(&depths, &parent_map);
    let mut new_geo = Geometry::new();
    for (solid_idx, hole_indices) in solid_groups.iter() {
        let solid_item = &closed_items[*solid_idx];
        let hole_paths: Vec<(i64, i64)> = hole_indices
            .iter()
            .flat_map(|&h_idx| closed_items[h_idx].path.clone())
            .collect();
        let offset_contours =
            offset_contour_group(&solid_item.path, &hole_paths, offset);
        for new_vertices in offset_contours {
            let points: Vec<(f64, f64, f64)> =
                new_vertices.iter().map(|(x, y)| (*x, *y, 0.0)).collect();
            let new_contour_geo = Geometry::from_points(&points, true);
            if !new_contour_geo.is_empty() {
                new_geo.extend(&new_contour_geo);
            }
        }
    }
    new_geo
}

/// Applies an affine transformation matrix to a geometry data array.
/// Handles uniform and non-uniform scaling (linearizing arcs for the latter).
/// Returns the transformed data. For non-uniform scaling, the returned data
/// may be longer than the input (arcs are linearized into lines).
pub fn apply_affine_transform_to_array(
    data: &[[f64; 8]],
    matrix: &[[f64; 4]; 4],
) -> Vec<[f64; 8]> {
    if data.is_empty() {
        return vec![];
    }

    let v_x = [matrix[0][0], matrix[1][0], matrix[2][0], matrix[3][0]];
    let v_y = [matrix[0][1], matrix[1][1], matrix[2][1], matrix[3][1]];
    let len_x_sq = v_x[0] * v_x[0] + v_x[1] * v_x[1];
    let len_y_sq = v_y[0] * v_y[0] + v_y[1] * v_y[1];
    let is_non_uniform = (len_x_sq - len_y_sq).abs() > 1e-9;

    if is_non_uniform {
        transform_array_non_uniform(data, matrix)
    } else {
        transform_array_uniform(data, matrix)
    }
}

fn transform_point(
    matrix: &[[f64; 4]; 4],
    x: f64,
    y: f64,
    z: f64,
) -> (f64, f64, f64) {
    let px =
        matrix[0][0] * x + matrix[0][1] * y + matrix[0][2] * z + matrix[0][3];
    let py =
        matrix[1][0] * x + matrix[1][1] * y + matrix[1][2] * z + matrix[1][3];
    let pz =
        matrix[2][0] * x + matrix[2][1] * y + matrix[2][2] * z + matrix[2][3];
    (px, py, pz)
}

fn transform_vec(matrix: &[[f64; 4]; 4], x: f64, y: f64) -> (f64, f64) {
    let vx = matrix[0][0] * x + matrix[0][1] * y;
    let vy = matrix[1][0] * x + matrix[1][1] * y;
    (vx, vy)
}

fn transform_array_uniform(
    data: &[[f64; 8]],
    matrix: &[[f64; 4]; 4],
) -> Vec<[f64; 8]> {
    let mut result: Vec<[f64; 8]> = Vec::with_capacity(data.len());
    for row in data {
        let cmd = Command::from_row(row).expect("invalid command");
        let (ex, ey, ez) = cmd.end_point();
        let (nx, ny, nz) = transform_point(matrix, ex, ey, ez);

        let transformed = match cmd {
            Command::Arc {
                end: _,
                center_offset,
                clockwise,
            } => {
                let (vi, vj) =
                    transform_vec(matrix, center_offset.0, center_offset.1);
                let det =
                    matrix[0][0] * matrix[1][1] - matrix[0][1] * matrix[1][0];
                let cw = if det < 0.0 { !clockwise } else { clockwise };
                Command::Arc {
                    end: (nx, ny, nz),
                    center_offset: (vi, vj),
                    clockwise: cw,
                }
            }
            Command::Bezier {
                end: _,
                control1,
                control2,
            } => {
                let (c1x, c1y, _) =
                    transform_point(matrix, control1.0, control1.1, 0.0);
                let (c2x, c2y, _) =
                    transform_point(matrix, control2.0, control2.1, 0.0);
                Command::Bezier {
                    end: (nx, ny, nz),
                    control1: (c1x, c1y),
                    control2: (c2x, c2y),
                }
            }
            Command::Move { .. } => Command::Move { end: (nx, ny, nz) },
            Command::Line { .. } => Command::Line { end: (nx, ny, nz) },
        };

        result.push(transformed.to_row());
    }
    result
}

fn transform_array_non_uniform(
    data: &[[f64; 8]],
    matrix: &[[f64; 4]; 4],
) -> Vec<[f64; 8]> {
    let mut result: Vec<[f64; 8]> = Vec::new();
    let mut last_pos: Point3D = (0.0, 0.0, 0.0);

    for row in data {
        let cmd = Command::from_row(row).expect("invalid command");
        let original_end = cmd.end_point();

        match cmd {
            Command::Arc { .. } => {
                let start_pt = last_pos;
                let segments = linearize_arc(row, start_pt, 0.1);
                for (_, p2) in segments {
                    let (tx, ty, tz) =
                        transform_point(matrix, p2.0, p2.1, p2.2);
                    let line_cmd = Command::Line { end: (tx, ty, tz) };
                    result.push(line_cmd.to_row());
                }
            }
            Command::Bezier {
                end: _,
                control1,
                control2,
            } => {
                let (nx, ny, nz) = transform_point(
                    matrix,
                    original_end.0,
                    original_end.1,
                    original_end.2,
                );
                let (c1x, c1y, _) =
                    transform_point(matrix, control1.0, control1.1, 0.0);
                let (c2x, c2y, _) =
                    transform_point(matrix, control2.0, control2.1, 0.0);
                let bezier_cmd = Command::Bezier {
                    end: (nx, ny, nz),
                    control1: (c1x, c1y),
                    control2: (c2x, c2y),
                };
                result.push(bezier_cmd.to_row());
            }
            _ => {
                let (nx, ny, nz) = transform_point(
                    matrix,
                    original_end.0,
                    original_end.1,
                    original_end.2,
                );
                let transformed = match cmd {
                    Command::Move { .. } => Command::Move { end: (nx, ny, nz) },
                    Command::Line { .. } => Command::Line { end: (nx, ny, nz) },
                    _ => unreachable!(),
                };
                result.push(transformed.to_row());
            }
        }

        last_pos = original_end;
    }
    result
}

/// Transforms a Geometry object to fit into an affine frame defined by
/// three points.
#[allow(clippy::too_many_arguments)]
pub fn map_geometry_to_frame(
    geometry: &Geometry,
    origin: Point,
    p_width: Point,
    p_height: Point,
    anchor_y: Option<f64>,
    stable_src_height: Option<f64>,
    anchor_x: Option<f64>,
    stable_src_width: Option<f64>,
) -> Geometry {
    if geometry.is_empty() {
        return Geometry::new();
    }

    let (min_x, min_y, max_x, max_y) = geometry.rect();
    let src_width = stable_src_width.unwrap_or(max_x - min_x);
    let src_height = stable_src_height.unwrap_or(max_y - min_y);

    let anchor_x_value = anchor_x.unwrap_or(min_x);
    let anchor_y_value = anchor_y.unwrap_or(min_y);

    if src_width < 1e-9 || src_height < 1e-9 {
        return Geometry::new();
    }

    let u_vec = (p_width.0 - origin.0, p_width.1 - origin.1);
    let v_vec = (p_height.0 - origin.0, p_height.1 - origin.1);

    let t1: [[f64; 4]; 4] = [
        [1.0, 0.0, 0.0, -anchor_x_value],
        [0.0, 1.0, 0.0, -anchor_y_value],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ];

    let t2: [[f64; 4]; 4] = [
        [1.0 / src_width, 0.0, 0.0, 0.0],
        [0.0, 1.0 / src_height, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ];

    let t3: [[f64; 4]; 4] = [
        [u_vec.0, v_vec.0, 0.0, origin.0],
        [u_vec.1, v_vec.1, 0.0, origin.1],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ];

    let final_matrix = mat4_mul(&t3, &mat4_mul(&t2, &t1));

    let mut transformed_geo = geometry.copy();
    transformed_geo.transform(&final_matrix);
    transformed_geo
}

fn mat4_mul(a: &[[f64; 4]; 4], b: &[[f64; 4]; 4]) -> [[f64; 4]; 4] {
    let mut result = [[0.0; 4]; 4];
    for i in 0..4 {
        for j in 0..4 {
            for k in 0..4 {
                result[i][j] += a[i][k] * b[k][j];
            }
        }
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rectangle_geo(x: f64, y: f64, w: f64, h: f64) -> Geometry {
        let points = [
            (x, y, 0.0),
            (x + w, y, 0.0),
            (x + w, y + h, 0.0),
            (x, y + h, 0.0),
        ];
        Geometry::from_points(&points[..], true)
    }

    #[test]
    fn test_grow_geometry_positive() {
        let rect = rectangle_geo(0.0, 0.0, 10.0, 10.0);
        let grown = grow_geometry(&rect, 1.0);
        assert!(grown.area() > rect.area());
    }

    #[test]
    fn test_grow_geometry_negative() {
        let rect = rectangle_geo(0.0, 0.0, 10.0, 10.0);
        let grown = grow_geometry(&rect, -1.0);
        assert!(grown.area() < rect.area());
    }

    #[test]
    fn test_grow_geometry_empty() {
        let geo = Geometry::new();
        let grown = grow_geometry(&geo, 1.0);
        assert!(grown.is_empty());
    }

    #[test]
    fn test_grow_geometry_zero_offset() {
        let rect = rectangle_geo(0.0, 0.0, 10.0, 10.0);
        let grown = grow_geometry(&rect, 0.0);
        let diff = (grown.area() - rect.area()).abs();
        assert!(diff < 1e-6);
    }

    #[test]
    fn test_prepare_contour_items() {
        let rect = rectangle_geo(0.0, 0.0, 10.0, 10.0);
        let contours = split_into_contours(&rect);
        let contour_data = get_valid_contours_data(&contours);
        let items = prepare_contour_items(&contour_data);
        assert_eq!(items.len(), 1);
        assert_eq!(items[0].area, 100.0);
    }

    #[test]
    fn test_calculate_nesting_depths() {
        let parent_map = vec![-1isize, 0, 1];
        let depths = calculate_nesting_depths(&parent_map, 3);
        assert_eq!(depths, vec![0, 1, 2]);
    }

    #[test]
    fn test_group_solids_and_holes() {
        let depths = vec![0, 1, 2, 1];
        let parent_map = vec![-1isize, 0, 1, 0];
        let groups = group_solids_and_holes(&depths, &parent_map);
        // depth 0 (solid) -> key 0 with holes [1, 3]
        // depth 1 (hole) -> belongs to parent 0
        // depth 2 (solid) -> key 2, no holes
        assert!(groups.contains_key(&0));
        assert!(groups.contains_key(&2));
        assert_eq!(groups[&0].len(), 2);
    }
}
