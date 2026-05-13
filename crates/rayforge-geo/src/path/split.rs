//! Split: Contour and component splitting for geometry data.
//!
//! Provides functions for splitting a Geometry into individual contours
//! (subpaths delimited by MOVE commands) and for separating logically
//! connected components using point-in-polygon containment and BFS.
//! Also provides contour analysis functions like reverse_contour,
//! normalize_winding_orders, filter_to_external_contours, etc.

use crate::path::analysis::{get_subpath_area_from_array, is_closed};
use crate::path::geometry::Geometry;
use crate::shape::polygon::is_point_inside_polygon;
use crate::types::{Command, Point, Rect};

/// Split a Geometry into individual contour geometries.
///
/// Each contour is a continuous subpath starting with a MOVE command.
/// The array is split at each MOVE boundary, creating separate Geometry
/// objects for each resulting segment.
pub fn split_into_contours(geometry: &Geometry) -> Vec<Geometry> {
    if geometry.data.is_empty() {
        return vec![];
    }
    let data = &geometry.data;

    let move_indices: Vec<usize> = data
        .iter()
        .enumerate()
        .filter(|(_, row)| {
            matches!(Command::from_row(row), Ok(Command::Move { .. }))
        })
        .map(|(i, _)| i)
        .collect();

    // No MOVE commands: treat entire data as one contour
    if move_indices.is_empty() {
        let mut new_geo = Geometry::new();
        new_geo.data = data.to_vec();
        new_geo.last_move_to = Command::from_row(&data[0])
            .expect("invalid command")
            .end_point();
        return vec![new_geo];
    }

    let mut contours: Vec<Geometry> = Vec::new();

    // Handle any leading data before the first MOVE
    if move_indices[0] != 0 {
        let mut new_geo = Geometry::new();
        new_geo.data = data[..move_indices[0]].to_vec();
        if !new_geo.data.is_empty() {
            new_geo.last_move_to = Command::from_row(&new_geo.data[0])
                .expect("invalid command")
                .end_point();
        }
        contours.push(new_geo);
    }

    // Split at each MOVE boundary
    for i in 0..move_indices.len() {
        let start = move_indices[i];
        let end = if i + 1 < move_indices.len() {
            move_indices[i + 1]
        } else {
            data.len()
        };
        let slice = &data[start..end];
        if !slice.is_empty() {
            let mut new_geo = Geometry::new();
            new_geo.data = slice.to_vec();
            new_geo.last_move_to = Command::from_row(&slice[0])
                .expect("invalid command")
                .end_point();
            contours.push(new_geo);
        }
    }

    contours.retain(|g| !g.is_empty());
    contours
}

/// Find connected components in an adjacency graph using BFS.
fn find_connected_components_bfs(
    num_contours: usize,
    adj: &[Vec<usize>],
) -> Vec<Vec<usize>> {
    let mut visited = vec![false; num_contours];
    let mut components: Vec<Vec<usize>> = Vec::new();

    for i in 0..num_contours {
        if visited[i] {
            continue;
        }
        let mut component = Vec::new();
        let mut queue = vec![i];
        visited[i] = true;
        while let Some(u) = queue.pop() {
            component.push(u);
            for &v in &adj[u] {
                if !visited[v] {
                    visited[v] = true;
                    queue.push(v);
                }
            }
        }
        components.push(component);
    }
    components
}

/// Extract valid contour data from a list of contour geometries.
///
/// Filters out empty, non-closed, or degenerate contours and returns
/// `(geometry, vertices, is_closed)` tuples for each valid contour.
pub fn get_valid_contours_data(
    contour_geometries: &[Geometry],
) -> Vec<(Geometry, Vec<Point>, bool)> {
    let mut result = Vec::new();
    for geo in contour_geometries {
        if geo.is_empty() {
            continue;
        }
        if geo.data.len() < 2 {
            continue;
        }
        if !matches!(Command::from_row(&geo.data[0]), Ok(Command::Move { .. }))
        {
            continue;
        }

        let closed = geo.is_closed(1e-6);
        let bbox = geo.rect();
        let bbox_area = (bbox.2 - bbox.0) * (bbox.3 - bbox.1);
        let is_closed_flag = closed && bbox_area > 1e-9;

        if !is_closed_flag {
            continue;
        }

        let vertices =
            crate::analysis::get_subpath_vertices_from_array(&geo.data, 0);

        result.push((geo.clone(), vertices, is_closed_flag));
    }
    result
}

/// Split a Geometry into logically connected components.
///
/// Uses point-in-polygon containment tests and BFS to group nested
/// contours (islands and holes) into separate component geometries.
pub fn split_into_components(geometry: &Geometry) -> Vec<Geometry> {
    if geometry.is_empty() {
        return vec![];
    }

    let contour_geometries = split_into_contours(geometry);
    if contour_geometries.len() <= 1 {
        return vec![geometry.copy()];
    }

    let all_contour_data = get_valid_contours_data(&contour_geometries);
    if all_contour_data.is_empty() {
        return vec![];
    }

    let any_closed = all_contour_data.iter().any(|(_, _, closed)| *closed);
    if !any_closed {
        return vec![geometry.copy()];
    }

    // Build adjacency based on point-in-polygon containment
    let num_contours = all_contour_data.len();
    let mut adj: Vec<Vec<usize>> = vec![vec![]; num_contours];

    for i in 0..num_contours {
        let (_, ref vertices_i, closed_i) = &all_contour_data[i];
        if !closed_i {
            continue;
        }
        for j in 0..num_contours {
            if i == j {
                continue;
            }
            let (_, ref vertices_j, _) = &all_contour_data[j];
            if vertices_j.is_empty() || vertices_i.is_empty() {
                continue;
            }
            if is_point_inside_polygon(vertices_j[0], vertices_i) {
                adj[i].push(j);
                adj[j].push(i);
            }
        }
    }

    let component_indices_list =
        find_connected_components_bfs(num_contours, &adj);

    let mut final_geometries: Vec<Geometry> = Vec::new();
    let mut stray_open = Geometry::new();
    stray_open.uniform_scalable = geometry.uniform_scalable;

    for indices in &component_indices_list {
        let mut component_geo = Geometry::new();
        component_geo.uniform_scalable = geometry.uniform_scalable;
        let mut has_closed = false;

        for &idx in indices {
            let (ref geo_data, _, closed) = &all_contour_data[idx];
            component_geo.extend(geo_data);
            if *closed {
                has_closed = true;
            }
        }

        if has_closed {
            final_geometries.push(component_geo);
        } else {
            stray_open.extend(&component_geo);
        }
    }

    if !stray_open.is_empty() {
        final_geometries.push(stray_open);
    }

    final_geometries
}

/// Reverse the direction of a single-contour Geometry.
pub fn reverse_contour(contour: &Geometry) -> Geometry {
    let data = &contour.data;
    if data.is_empty() {
        return contour.copy();
    }
    let first_cmd = Command::from_row(&data[0]).expect("invalid command");
    if !matches!(first_cmd, Command::Move { .. }) {
        return contour.copy();
    }

    let mut new_rows: Vec<[f64; 8]> = Vec::with_capacity(data.len());

    let last_cmd =
        Command::from_row(&data[data.len() - 1]).expect("invalid command");
    new_rows.push(
        Command::Move {
            end: last_cmd.end_point(),
        }
        .to_row(),
    );
    let mut last_point = last_cmd.end_point();

    for i in (1..data.len()).rev() {
        let end_cmd = Command::from_row(&data[i]).expect("invalid command");
        let start_cmd =
            Command::from_row(&data[i - 1]).expect("invalid command");
        let start_point = start_cmd.end_point();

        match end_cmd {
            Command::Line { .. } => {
                new_rows.push(Command::Line { end: start_point }.to_row());
            }
            Command::Arc {
                center_offset,
                clockwise,
                ..
            } => {
                let center_abs_x = start_point.0 + center_offset.0;
                let center_abs_y = start_point.1 + center_offset.1;
                let new_offset_x = center_abs_x - last_point.0;
                let new_offset_y = center_abs_y - last_point.1;
                new_rows.push(
                    Command::Arc {
                        end: start_point,
                        center_offset: (new_offset_x, new_offset_y),
                        clockwise: !clockwise,
                    }
                    .to_row(),
                );
            }
            Command::Bezier {
                control1, control2, ..
            } => {
                new_rows.push(
                    Command::Bezier {
                        end: start_point,
                        control1: control2,
                        control2: control1,
                    }
                    .to_row(),
                );
            }
            Command::Move { .. } => {}
        }

        last_point = start_point;
    }

    let mut new_geo = Geometry::new();
    new_geo.data = new_rows;
    new_geo.last_move_to = first_cmd.end_point();
    new_geo
}

/// Split contours into inner and outer groups based on the even-odd rule.
pub fn split_inner_and_outer_contours(
    contours: &[Geometry],
) -> (Vec<usize>, Vec<usize>) {
    if contours.is_empty() {
        return (vec![], vec![]);
    }

    let count = contours.len();

    struct ContourInfo {
        vertices: Vec<Point>,
        rect: Rect,
        test_point: Point,
    }

    let mut info_list: Vec<Option<ContourInfo>> = Vec::with_capacity(count);

    for c in contours {
        if c.is_empty() || c.data.is_empty() {
            info_list.push(None);
            continue;
        }

        let c_is_closed = is_closed(&c.data, 1e-6);
        if !c_is_closed {
            info_list.push(None);
            continue;
        }

        let segments = c.segments();
        if segments.is_empty() {
            info_list.push(None);
            continue;
        }

        let verts_3d = &segments[0];
        let verts_2d: Vec<Point> =
            verts_3d.iter().map(|p| (p.0, p.1)).collect();
        let rect = c.rect();
        let test_point = if verts_2d.is_empty() {
            (0.0, 0.0)
        } else {
            verts_2d[0]
        };

        let area = get_subpath_area_from_array(&c.data, 0);
        let area_ok = area.abs() > 1e-9;

        if !area_ok {
            info_list.push(None);
            continue;
        }

        info_list.push(Some(ContourInfo {
            vertices: verts_2d,
            rect,
            test_point,
        }));
    }

    let mut is_external = vec![true; count];

    for (i, info) in info_list.iter().enumerate() {
        let current = match info {
            Some(info) => info,
            None => continue,
        };

        let mut nesting_level = 0u32;
        let (tx, ty) = current.test_point;

        for (j, other_info) in info_list.iter().enumerate() {
            if i == j {
                continue;
            }
            let other = match other_info {
                Some(info) => info,
                None => continue,
            };

            let (o_min_x, o_min_y, o_max_x, o_max_y) = other.rect;
            if tx < o_min_x || tx > o_max_x || ty < o_min_y || ty > o_max_y {
                continue;
            }

            if is_point_inside_polygon(current.test_point, &other.vertices) {
                nesting_level += 1;
            }
        }

        is_external[i] = nesting_level.is_multiple_of(2);
    }

    let mut internal_indices: Vec<usize> = Vec::new();
    let mut external_indices: Vec<usize> = Vec::new();

    for (i, _) in contours.iter().enumerate() {
        if is_external[i] {
            external_indices.push(i);
        } else {
            internal_indices.push(i);
        }
    }

    (internal_indices, external_indices)
}

/// Close all contours in a geometry.
pub fn close_all_contours(geometry: &Geometry) -> Geometry {
    if geometry.is_empty() {
        return geometry.copy();
    }

    let contours = split_into_contours(geometry);
    if contours.is_empty() {
        return geometry.copy();
    }

    let mut result = Geometry::new();
    for mut contour in contours {
        if !contour.is_closed(1e-6) {
            contour.close_path();
            contour.sync_to_data();
        }
        result.extend(&contour);
    }

    result.last_move_to = geometry.last_move_to;
    result
}

/// Normalize winding orders of contours (CCW for solids, CW for holes).
pub fn normalize_winding_orders(contours: &[Geometry]) -> Vec<Geometry> {
    if contours.is_empty() {
        return vec![];
    }

    let count = contours.len();
    type ContourEntry = (Geometry, Vec<Point>, Rect, Point, bool);
    let mut contour_data: Vec<Option<ContourEntry>> = Vec::with_capacity(count);
    let mut normalized_contours: Vec<Geometry> = Vec::new();

    for c in contours {
        if c.is_empty() {
            contour_data.push(None);
            continue;
        }
        if c.data.is_empty() && c.pending_data.is_empty() {
            contour_data.push(None);
            continue;
        }

        let c_is_closed = is_closed(&c.data, 1e-6);

        if !c_is_closed {
            normalized_contours.push(c.copy());
            contour_data.push(None);
            continue;
        }

        let segments = c.segments();
        if segments.is_empty() {
            contour_data.push(None);
            continue;
        }

        let verts_3d = &segments[0];
        let verts_2d: Vec<Point> =
            verts_3d.iter().map(|p| (p.0, p.1)).collect();
        let rect = c.rect();
        let test_point = if verts_2d.is_empty() {
            (0.0, 0.0)
        } else {
            verts_2d[0]
        };

        contour_data.push(Some((c.copy(), verts_2d, rect, test_point, true)));
    }

    for (i, entry) in contour_data.iter().enumerate() {
        let current = match entry {
            Some(data) => data,
            None => continue,
        };

        let mut nesting_level = 0;
        let (tx, ty) = current.3;

        for (j, other_entry) in contour_data.iter().enumerate() {
            if i == j {
                continue;
            }
            let other = match other_entry {
                Some(data) => data,
                None => continue,
            };

            let (o_min_x, o_min_y, o_max_x, o_max_y) = other.2;
            if tx < o_min_x || tx > o_max_x || ty < o_min_y || ty > o_max_y {
                continue;
            }

            if is_point_inside_polygon(current.3, &other.1) {
                nesting_level += 1;
            }
        }

        let signed_area = get_subpath_area_from_array(&current.0.data, 0);
        let is_ccw = signed_area > 0.0;
        let is_nested_odd = nesting_level % 2 != 0;

        if (is_nested_odd && is_ccw) || (!is_nested_odd && !is_ccw) {
            normalized_contours.push(reverse_contour(&current.0));
        } else {
            normalized_contours.push(current.0.copy());
        }
    }

    normalized_contours
}

/// Filter to only external contours (solid filled areas).
pub fn filter_to_external_contours(contours: &[Geometry]) -> Vec<Geometry> {
    if contours.is_empty() {
        return vec![];
    }

    let normalized_contours = normalize_winding_orders(contours);

    let mut final_contours: Vec<Geometry> = Vec::new();
    for c in &normalized_contours {
        if !c.data.is_empty() {
            let area = get_subpath_area_from_array(&c.data, 0);
            if area > 1e-9 {
                final_contours.push(c.copy());
            }
        }
    }
    final_contours
}

/// Remove inner edges (holes) from a geometry, keeping only external contours.
pub fn remove_inner_edges(geometry: &Geometry) -> Geometry {
    if geometry.is_empty() {
        return Geometry::new();
    }

    let all_contours = split_into_contours(geometry);
    if all_contours.is_empty() {
        return Geometry::new();
    }

    let mut closed_contours: Vec<Geometry> = Vec::new();
    let mut open_contours: Vec<Geometry> = Vec::new();

    for contour in all_contours {
        if contour.is_closed(1e-6) {
            closed_contours.push(contour);
        } else {
            open_contours.push(contour);
        }
    }

    let external_closed = filter_to_external_contours(&closed_contours);

    let mut final_geo = Geometry::new();
    for contour in &external_closed {
        final_geo.extend(contour);
    }
    for contour in &open_contours {
        final_geo.extend(contour);
    }

    final_geo.last_move_to = geometry.last_move_to;
    final_geo
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_split_empty() {
        let geo = Geometry::new();
        let result = split_into_contours(&geo);
        assert!(result.is_empty());
    }

    #[test]
    fn test_split_single_contour() {
        let mut geo = Geometry::new();
        geo.move_to(0.0, 0.0, 0.0);
        geo.line_to(10.0, 0.0, 0.0);
        geo.line_to(10.0, 10.0, 0.0);
        geo.sync_to_data();
        let result = split_into_contours(&geo);
        assert_eq!(result.len(), 1);
    }

    #[test]
    fn test_split_two_contours() {
        let mut geo = Geometry::new();
        geo.move_to(0.0, 0.0, 0.0);
        geo.line_to(10.0, 0.0, 0.0);
        geo.move_to(20.0, 20.0, 0.0);
        geo.line_to(30.0, 20.0, 0.0);
        geo.sync_to_data();
        let result = split_into_contours(&geo);
        assert_eq!(result.len(), 2);
    }

    #[test]
    fn test_bfs_components() {
        let adj = vec![vec![1], vec![0, 2], vec![1]];
        let components = find_connected_components_bfs(3, &adj);
        assert_eq!(components.len(), 1);
    }

    #[test]
    fn test_bfs_disconnected() {
        let adj = vec![vec![], vec![], vec![]];
        let components = find_connected_components_bfs(3, &adj);
        assert_eq!(components.len(), 3);
    }
}
