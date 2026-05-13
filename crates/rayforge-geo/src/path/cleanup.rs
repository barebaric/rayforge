//! Cleanup: Deduplication and gap closing for geometry data.
//!
//! Provides functions for cleaning geometry command arrays by removing
//! duplicate segments and closing small gaps between connected paths.

use crate::types::Command;

/// Check if two points (as 3-element arrays) are equal within a tolerance.
pub fn are_points_equal(p1: &[f64; 3], p2: &[f64; 3], tolerance: f64) -> bool {
    for i in 0..3 {
        if (p1[i] - p2[i]).abs() > tolerance {
            return false;
        }
    }
    true
}

/// Extract a hashable key for a segment at the given index.
/// Returns None for MOVE commands.
pub fn get_segment_key(row: &[f64; 8]) -> Option<(u32, [f64; 3], [f64; 4])> {
    let cmd = Command::from_row(row).ok()?;
    match cmd {
        Command::Move { .. } => None,
        Command::Line { end } => Some((2, [end.0, end.1, end.2], [0.0; 4])),
        Command::Arc {
            end,
            center_offset,
            clockwise,
        } => {
            let params = [
                center_offset.0,
                center_offset.1,
                if clockwise { 1.0 } else { 0.0 },
                0.0,
            ];
            Some((3, [end.0, end.1, end.2], params))
        }
        Command::Bezier {
            end,
            control1,
            control2,
        } => {
            let params = [control1.0, control1.1, control2.0, control2.1];
            Some((4, [end.0, end.1, end.2], params))
        }
    }
}

/// Check if two segment keys represent identical segments within tolerance.
pub fn are_segments_equal(
    k1: &(u32, [f64; 3], [f64; 4]),
    k2: &(u32, [f64; 3], [f64; 4]),
    tolerance: f64,
) -> bool {
    if k1.0 != k2.0 {
        return false;
    }
    if !are_points_equal(&k1.1, &k2.1, tolerance) {
        return false;
    }
    if k1.0 == 2 {
        return true;
    }
    if k1.0 == 3 {
        return are_points_equal(
            &[k1.2[0], k1.2[1], 0.0],
            &[k2.2[0], k2.2[1], 0.0],
            tolerance,
        ) && (k1.2[2] - k2.2[2]).abs() < tolerance;
    }
    if k1.0 == 4 {
        let p1 = [k1.2[0], k1.2[1], k1.2[2]];
        let p2 = [k2.2[0], k2.2[1], k2.2[2]];
        return are_points_equal(&p1, &p2, tolerance)
            && (k1.2[3] - k2.2[3]).abs() < tolerance;
    }
    false
}

/// Remove duplicate segments from geometry command data.
///
/// A segment is considered duplicate if it has the same type, endpoint,
/// and parameters as another segment within the same subpath. MOVE commands
/// reset duplicate detection for new paths.
pub fn remove_duplicate_segments(
    data: &[[f64; 8]],
    tolerance: f64,
) -> Vec<[f64; 8]> {
    if data.is_empty() {
        return data.to_vec();
    }

    let mut result: Vec<[f64; 8]> = Vec::new();
    let mut seen_segments: Vec<(u32, [f64; 3], [f64; 4])> = Vec::new();

    for row in data {
        let cmd = Command::from_row(row).expect("invalid command");

        if matches!(cmd, Command::Move { .. }) {
            seen_segments.clear();
            result.push(*row);
            continue;
        }

        if let Some(key) = get_segment_key(row) {
            let is_dup = seen_segments
                .iter()
                .any(|sk| are_segments_equal(&key, sk, tolerance));
            if is_dup {
                continue;
            }
            seen_segments.push(key);
        }
        result.push(*row);
    }

    result
}

/// Close small gaps in a geometry data array to form clean, connected paths.
///
/// Two operations are performed:
/// 1. Within each subpath (between MOVE commands), if the end point is within
///    tolerance of the start point, snap the end point to the start.
/// 2. If a MOVE command starts within tolerance of the previous subpath's end,
///    convert it to a LINE command (bridging the gap).
pub fn close_geometry_gaps_from_array(
    data: &[[f64; 8]],
    tolerance: f64,
) -> Vec<[f64; 8]> {
    if data.len() < 2 {
        return data.to_vec();
    }

    let tol_sq = tolerance * tolerance;

    // Find MOVE command indices to identify subpath boundaries
    let mut move_indices: Vec<usize> = Vec::new();
    for (i, row) in data.iter().enumerate() {
        let cmd = Command::from_row(row).expect("invalid command");
        if matches!(cmd, Command::Move { .. }) {
            move_indices.push(i);
        }
    }

    let mut modified: Vec<[f64; 8]> = data.to_vec();

    // Build ranges for each subpath
    let sub_ranges: Vec<(usize, usize)> = if move_indices.is_empty() {
        vec![(0, data.len())]
    } else {
        let mut ranges = Vec::new();
        let mut prev = 0;
        for &mi in &move_indices[1..] {
            ranges.push((prev, mi));
            prev = mi;
        }
        ranges.push((prev, data.len()));
        ranges
    };

    // Close gaps within each subpath
    for &(start, end) in &sub_ranges {
        if end - start >= 2 {
            let s = Command::from_row(&modified[start])
                .expect("invalid command")
                .end_point();
            let e_cmd =
                Command::from_row(&modified[end - 1]).expect("invalid command");
            let e = e_cmd.end_point();
            let dsq =
                (s.0 - e.0).powi(2) + (s.1 - e.1).powi(2) + (s.2 - e.2).powi(2);
            if dsq < tol_sq {
                let new_cmd = match e_cmd {
                    Command::Move { .. } => Command::Move { end: s },
                    Command::Line { .. } => Command::Line { end: s },
                    Command::Arc {
                        center_offset,
                        clockwise,
                        ..
                    } => Command::Arc {
                        end: s,
                        center_offset,
                        clockwise,
                    },
                    Command::Bezier {
                        control1, control2, ..
                    } => Command::Bezier {
                        end: s,
                        control1,
                        control2,
                    },
                };
                modified[end - 1] = new_cmd.to_row();
            }
        }
    }

    // Convert MOVE to LINE when at the same position as previous endpoint
    let mut final_rows: Vec<[f64; 8]> = Vec::new();
    let mut last_end: Option<(f64, f64, f64)> = None;

    for &row in &modified {
        let cmd = Command::from_row(&row).expect("invalid command");
        let end_pt = cmd.end_point();

        if matches!(cmd, Command::Move { .. }) {
            if let Some(prev) = last_end {
                let dsq = (end_pt.0 - prev.0).powi(2)
                    + (end_pt.1 - prev.1).powi(2)
                    + (end_pt.2 - prev.2).powi(2);
                if dsq < tol_sq {
                    let line = Command::Line { end: prev };
                    final_rows.push(line.to_row());
                } else {
                    final_rows.push(row);
                }
            } else {
                final_rows.push(row);
            }
        } else {
            final_rows.push(row);
        }
        last_end = Some(end_pt);
    }

    final_rows
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::constants::*;

    #[test]
    fn test_remove_duplicate_lines() {
        let data: Vec<[f64; 8]> = vec![
            [CMD_TYPE_MOVE, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [CMD_TYPE_LINE, 10.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [CMD_TYPE_LINE, 10.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        ];
        let result = remove_duplicate_segments(&data, 1e-6);
        assert_eq!(result.len(), 2);
    }

    #[test]
    fn test_remove_duplicate_different() {
        let data: Vec<[f64; 8]> = vec![
            [CMD_TYPE_MOVE, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [CMD_TYPE_LINE, 10.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [CMD_TYPE_LINE, 10.0, 10.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        ];
        let result = remove_duplicate_segments(&data, 1e-6);
        assert_eq!(result.len(), 3);
    }

    #[test]
    fn test_close_gaps_simple() {
        let data: Vec<[f64; 8]> = vec![
            [CMD_TYPE_MOVE, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [CMD_TYPE_LINE, 10.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [CMD_TYPE_LINE, 10.0, 10.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [CMD_TYPE_LINE, 0.1, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0],
        ];
        let result = close_geometry_gaps_from_array(&data, 0.2);
        assert_eq!(result.last().unwrap()[COL_X], 0.0);
        assert_eq!(result.last().unwrap()[COL_Y], 0.0);
    }
}
