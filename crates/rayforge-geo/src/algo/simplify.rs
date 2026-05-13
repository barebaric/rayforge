use crate::types::Point3D;

/// Simplify a sequence of 3D points using the Ramer-Douglas-Peucker algorithm.
///
/// Uses an iterative stack-based approach (not recursion) to avoid stack overflow
/// on large point sequences. The first and last points are always preserved.
/// Only the X and Y coordinates are used for distance calculation; the Z coordinate
/// is preserved in the output.
pub fn simplify_polyline(points: &[Point3D], tolerance: f64) -> Vec<Point3D> {
    let n = points.len();
    if n < 3 {
        return points.to_vec();
    }

    // Boolean mask of points to keep
    let tol_sq = tolerance * tolerance;
    let mut keep = vec![false; n];
    keep[0] = true;
    keep[n - 1] = true;

    // Iterative stack to avoid recursion depth issues
    let mut stack: Vec<(usize, usize)> = vec![(0, n - 1)];

    while let Some((start, end)) = stack.pop() {
        if end - start < 2 {
            continue;
        }

        let p_start = (points[start].0, points[start].1);
        let p_end = (points[end].0, points[end].1);
        let chord_vec = (p_end.0 - p_start.0, p_end.1 - p_start.1);
        let chord_len_sq =
            chord_vec.0 * chord_vec.0 + chord_vec.1 * chord_vec.1;

        let mut max_dist_sq = 0.0_f64;
        let mut max_idx = start;

        if chord_len_sq < 1e-12 {
            // Start and End are practically identical; use Euclidean distance from start
            for (i, p) in points.iter().enumerate().take(end).skip(start + 1) {
                let d_sq =
                    (p.0 - p_start.0).powi(2) + (p.1 - p_start.1).powi(2);
                if d_sq > max_dist_sq {
                    max_dist_sq = d_sq;
                    max_idx = i;
                }
            }
        } else {
            // Vectorized perpendicular distance: |Cross(v_start_to_pt, chord)| / |chord|
            for (i, p) in points.iter().enumerate().take(end).skip(start + 1) {
                let cross = (p.0 - p_start.0) * chord_vec.1
                    - (p.1 - p_start.1) * chord_vec.0;
                let d_sq = (cross * cross) / chord_len_sq;
                if d_sq > max_dist_sq {
                    max_dist_sq = d_sq;
                    max_idx = i;
                }
            }
        }

        if max_dist_sq > tol_sq {
            keep[max_idx] = true;
            stack.push((start, max_idx));
            stack.push((max_idx, end));
        }
    }

    points
        .iter()
        .enumerate()
        .filter(|(i, _)| keep[*i])
        .map(|(_, p)| *p)
        .collect()
}

/// Simplify geometry command data using the Ramer-Douglas-Peucker algorithm.
///
/// Operates directly on `[f64; 8]` command rows (columns 1=X, 2=Y used for
/// distance; all columns preserved in output).
pub fn simplify_data(data: &[[f64; 8]], tolerance: f64) -> Vec<[f64; 8]> {
    let n = data.len();
    if n < 3 {
        return data.to_vec();
    }

    let tol_sq = tolerance * tolerance;
    let mut keep = vec![false; n];
    keep[0] = true;
    keep[n - 1] = true;

    let mut stack: Vec<(usize, usize)> = vec![(0, n - 1)];

    while let Some((start, end)) = stack.pop() {
        if end - start < 2 {
            continue;
        }

        let p_start = (data[start][1], data[start][2]);
        let p_end = (data[end][1], data[end][2]);
        let chord_vec = (p_end.0 - p_start.0, p_end.1 - p_start.1);
        let chord_len_sq =
            chord_vec.0 * chord_vec.0 + chord_vec.1 * chord_vec.1;

        let mut max_dist_sq = 0.0_f64;
        let mut max_idx = start;

        if chord_len_sq < 1e-12 {
            for (i, row) in data.iter().enumerate().take(end).skip(start + 1) {
                let d_sq =
                    (row[1] - p_start.0).powi(2) + (row[2] - p_start.1).powi(2);
                if d_sq > max_dist_sq {
                    max_dist_sq = d_sq;
                    max_idx = i;
                }
            }
        } else {
            for (i, row) in data.iter().enumerate().take(end).skip(start + 1) {
                let cross = (row[1] - p_start.0) * chord_vec.1
                    - (row[2] - p_start.1) * chord_vec.0;
                let d_sq = (cross * cross) / chord_len_sq;
                if d_sq > max_dist_sq {
                    max_dist_sq = d_sq;
                    max_idx = i;
                }
            }
        }

        if max_dist_sq > tol_sq {
            keep[max_idx] = true;
            stack.push((start, max_idx));
            stack.push((max_idx, end));
        }
    }

    data.iter()
        .enumerate()
        .filter(|(i, _)| keep[*i])
        .map(|(_, r)| *r)
        .collect()
}

/// Simplify a sequence of 3D points using RDP, returning the simplified points.
/// Input is a flat Nx3 array (rows of [x, y, z]).
pub fn simplify_polyline_to_array(
    points: &[Point3D],
    tolerance: f64,
) -> Vec<Point3D> {
    simplify_polyline(points, tolerance)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_simplify_trivial() {
        let points = vec![(0.0, 0.0, 0.0), (10.0, 10.0, 0.0)];
        let result = simplify_polyline(&points, 0.001);
        assert_eq!(result.len(), 2);
    }

    #[test]
    fn test_simplify_collinear() {
        let points = vec![
            (0.0, 0.0, 0.0),
            (1.0, 1.0, 0.0),
            (2.0, 2.0, 0.0),
            (3.0, 3.0, 0.0),
            (10.0, 10.0, 0.0),
        ];
        let result = simplify_polyline(&points, 0.001);
        assert_eq!(result.len(), 2);
        assert_eq!(result[0], (0.0, 0.0, 0.0));
        assert_eq!(result[result.len() - 1], (10.0, 10.0, 0.0));
    }

    #[test]
    fn test_simplify_keeps_corner() {
        let points = vec![(0.0, 0.0, 0.0), (5.0, 5.0, 0.0), (10.0, 0.0, 0.0)];
        let result = simplify_polyline(&points, 0.1);
        assert_eq!(result.len(), 3);
    }

    #[test]
    fn test_simplify_data_trivial() {
        let data: Vec<[f64; 8]> = vec![
            [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [2.0, 10.0, 10.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        ];
        let result = simplify_data(&data, 0.001);
        assert_eq!(result.len(), 2);
    }
}
