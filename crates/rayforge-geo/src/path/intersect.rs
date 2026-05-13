//! Intersect: Geometry intersection detection.
//!
//! Provides functions for checking self-intersection and cross-intersection
//! of geometry command arrays. Arcs and Bezier curves are linearized into
//! line segments for testing, and bounding box culling is used for performance.

use crate::constants::*;
use crate::shape::arc::linearize_arc;
use crate::shape::bezier::linearize_bezier_from_array;
use crate::shape::line::get_line_segment_intersection;
use crate::types::Point3D;

/// Returns a list of linearized line segments for a given row in a data array.
/// Arcs and Beziers are converted to approximating line segments.
fn get_segments_for_row(
    data: &[[f64; 8]],
    index: usize,
) -> Vec<(Point3D, Point3D)> {
    let row = data[index];
    let cmd_type = row[COL_TYPE] as i32;
    let end_point = (row[COL_X], row[COL_Y], row[COL_Z]);

    let start_point = if index > 0 {
        let prev = data[index - 1];
        (prev[COL_X], prev[COL_Y], prev[COL_Z])
    } else {
        (0.0, 0.0, 0.0)
    };

    if cmd_type == CMD_TYPE_LINE as i32 {
        vec![(start_point, end_point)]
    } else if cmd_type == CMD_TYPE_ARC as i32 {
        linearize_arc(&row, start_point, 0.1)
    } else if cmd_type == CMD_TYPE_BEZIER as i32 {
        linearize_bezier_from_array(&row, start_point, 0.1)
    } else {
        vec![]
    }
}

/// Pre-computed segments and bounding box for a single command row.
struct RowSegments {
    index: usize,
    segments: Vec<(Point3D, Point3D)>,
    bbox: (f64, f64, f64, f64),
}

/// Pre-compute linearized segments and bounding boxes for all draw commands.
fn precompute_row_segments(data: &[[f64; 8]]) -> Vec<RowSegments> {
    let mut rows = Vec::new();
    for i in 0..data.len() {
        let cmd_type = data[i][COL_TYPE] as i32;
        if cmd_type != CMD_TYPE_LINE as i32
            && cmd_type != CMD_TYPE_ARC as i32
            && cmd_type != CMD_TYPE_BEZIER as i32
        {
            continue;
        }
        let segments = get_segments_for_row(data, i);
        if segments.is_empty() {
            continue;
        }
        let mut min_x = f64::INFINITY;
        let mut max_x = f64::NEG_INFINITY;
        let mut min_y = f64::INFINITY;
        let mut max_y = f64::NEG_INFINITY;
        for (p1, p2) in &segments {
            for &pt in &[p1, p2] {
                if pt.0 < min_x {
                    min_x = pt.0;
                }
                if pt.0 > max_x {
                    max_x = pt.0;
                }
                if pt.1 < min_y {
                    min_y = pt.1;
                }
                if pt.1 > max_y {
                    max_y = pt.1;
                }
            }
        }
        rows.push(RowSegments {
            index: i,
            segments,
            bbox: (min_x, min_y, max_x, max_y),
        });
    }
    rows
}

/// Core intersection test between two geometry data arrays.
///
/// For self-intersection checks (`is_self_check=true`), adjacent segments
/// sharing a vertex are not counted as intersecting (they share an endpoint
/// by construction).
fn data_intersect(
    data1: &[[f64; 8]],
    data2: &[[f64; 8]],
    is_self_check: bool,
    fail_on_t_junction: bool,
) -> bool {
    let rows1 = precompute_row_segments(data1);
    let rows2 = precompute_row_segments(data2);

    for ri1 in &rows1 {
        for ri2 in &rows2 {
            // Skip self-comparisons for self-intersection and adjacent rows
            if is_self_check && ri2.index <= ri1.index {
                continue;
            }

            // Bounding box culling
            if ri1.bbox.2 < ri2.bbox.0 || ri1.bbox.0 > ri2.bbox.2 {
                continue;
            }
            if ri1.bbox.3 < ri2.bbox.1 || ri1.bbox.1 > ri2.bbox.3 {
                continue;
            }

            for &(seg1_p1, seg1_p2) in &ri1.segments {
                for &(seg2_p1, seg2_p2) in &ri2.segments {
                    let intersection = get_line_segment_intersection(
                        (seg1_p1.0, seg1_p1.1),
                        (seg1_p2.0, seg1_p2.1),
                        (seg2_p1.0, seg2_p1.1),
                        (seg2_p2.0, seg2_p2.1),
                    );

                    if let Some(pt) = intersection {
                        // For adjacent segments in a self-intersection check,
                        // the shared vertex is not a real intersection
                        if is_self_check && ri2.index == ri1.index + 1 {
                            let shared_vertex = [
                                data1[ri1.index][COL_X],
                                data1[ri1.index][COL_Y],
                            ];
                            let dsq = (pt.0 - shared_vertex[0]).powi(2)
                                + (pt.1 - shared_vertex[1]).powi(2);
                            if dsq < 1e-12 {
                                continue;
                            }
                            return true;
                        }

                        let at_end1 = (pt.0 - seg1_p1.0).powi(2)
                            + (pt.1 - seg1_p1.1).powi(2)
                            < 1e-12
                            || (pt.0 - seg1_p2.0).powi(2)
                                + (pt.1 - seg1_p2.1).powi(2)
                                < 1e-12;
                        let at_end2 = (pt.0 - seg2_p1.0).powi(2)
                            + (pt.1 - seg2_p1.1).powi(2)
                            < 1e-12
                            || (pt.0 - seg2_p2.0).powi(2)
                                + (pt.1 - seg2_p2.1).powi(2)
                                < 1e-12;

                        if is_self_check
                            && (at_end1 || at_end2)
                            && !fail_on_t_junction
                        {
                            continue;
                        }
                        return true;
                    }
                }
            }
        }
    }
    false
}

/// Check if a path self-intersects.
/// Each subpath (delimited by MOVE commands) is checked independently.
pub fn check_self_intersection_from_array(
    data: &[[f64; 8]],
    fail_on_t_junction: bool,
) -> bool {
    let mut move_indices: Vec<usize> = Vec::new();
    for (i, row) in data.iter().enumerate() {
        if (row[COL_TYPE] as i32) == CMD_TYPE_MOVE as i32 {
            move_indices.push(i);
        }
    }

    if move_indices.is_empty() {
        return data_intersect(data, data, true, fail_on_t_junction);
    }

    for i in 0..move_indices.len() {
        let start = move_indices[i];
        let end = if i + 1 < move_indices.len() {
            move_indices[i + 1]
        } else {
            data.len()
        };

        if end - start > 1 {
            let subpath = &data[start..end];
            if data_intersect(subpath, subpath, true, fail_on_t_junction) {
                return true;
            }
        }
    }
    false
}

/// Check if two geometry data arrays intersect each other.
pub fn check_intersection_from_array(
    data1: &[[f64; 8]],
    data2: &[[f64; 8]],
    fail_on_t_junction: bool,
) -> bool {
    data_intersect(data1, data2, false, fail_on_t_junction)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_no_self_intersection() {
        let data = [
            [CMD_TYPE_MOVE, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [CMD_TYPE_LINE, 10.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [CMD_TYPE_LINE, 10.0, 10.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [CMD_TYPE_LINE, 0.0, 10.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        ];
        assert!(!check_self_intersection_from_array(&data, false));
    }

    #[test]
    fn test_self_intersection_found() {
        let data = [
            [CMD_TYPE_MOVE, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [CMD_TYPE_LINE, 10.0, 10.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [CMD_TYPE_LINE, 0.0, 10.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [CMD_TYPE_LINE, 10.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        ];
        assert!(check_self_intersection_from_array(&data, false));
    }

    #[test]
    fn test_intersection_between_paths() {
        let data1 = [
            [CMD_TYPE_MOVE, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [CMD_TYPE_LINE, 10.0, 10.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        ];
        let data2 = [
            [CMD_TYPE_MOVE, 0.0, 10.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [CMD_TYPE_LINE, 10.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        ];
        assert!(check_intersection_from_array(&data1, &data2, false));
    }

    #[test]
    fn test_no_intersection_between_paths() {
        let data1 = [
            [CMD_TYPE_MOVE, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [CMD_TYPE_LINE, 10.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        ];
        let data2 = [
            [CMD_TYPE_MOVE, 0.0, 10.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [CMD_TYPE_LINE, 10.0, 10.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        ];
        assert!(!check_intersection_from_array(&data1, &data2, false));
    }
}
