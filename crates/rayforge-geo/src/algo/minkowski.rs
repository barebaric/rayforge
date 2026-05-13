//! Minkowski: Minkowski sum/difference operations for polygons.
//!
//! This module provides functions for calculating Minkowski sums and differences
//! of polygons, which are used in packing and nesting algorithms for computing
//! No-Fit Polygons (NFP) and Inner-Fit Polygons (IFP).

use crate::shape::polygon::{get_polygon_bounds, get_polygon_convex_hull};
use crate::types::{IntPolygon, Point, Polygon};

const CLIPPER_SCALE: i64 = 10_000_000;

pub fn convolve_two_segments(
    a1: (i64, i64),
    a2: (i64, i64),
    b1: (i64, i64),
    b2: (i64, i64),
) -> Vec<(i64, i64)> {
    vec![
        (a1.0 + b2.0, a1.1 + b2.1),
        (a1.0 + b1.0, a1.1 + b1.1),
        (a2.0 + b1.0, a2.1 + b1.1),
        (a2.0 + b2.0, a2.1 + b2.1),
    ]
}

pub fn convolve_point_sequences(
    seq_a: &IntPolygon,
    seq_b: &IntPolygon,
) -> Vec<IntPolygon> {
    let mut parallelograms: Vec<IntPolygon> = Vec::new();
    if seq_a.len() < 2 || seq_b.len() < 2 {
        return parallelograms;
    }
    let n = seq_a.len();
    let m = seq_b.len();
    for i in 0..n {
        let p_a1 = seq_a[(i + n - 1) % n];
        let p_a2 = seq_a[i];
        for j in 0..m {
            let p_b1 = seq_b[(j + m - 1) % m];
            let p_b2 = seq_b[j];
            parallelograms.push(convolve_two_segments(p_a1, p_a2, p_b1, p_b2));
        }
    }
    parallelograms
}

pub fn calculate_input_scale(polygons: &[Polygon], max_int: i64) -> f64 {
    if polygons.is_empty() {
        return 0.1 * (max_int as f64);
    }
    let mut max_abs = 0.0f64;
    for poly in polygons {
        for &(x, y) in poly {
            max_abs = max_abs.max(x.abs()).max(y.abs());
        }
    }
    if max_abs < 1.0 {
        max_abs = 1.0;
    }
    0.1 * (max_int as f64) / max_abs
}

pub fn get_polygon_minkowski_sum_convex(
    poly_a: &IntPolygon,
    poly_b: &IntPolygon,
) -> Vec<IntPolygon> {
    if poly_a.is_empty() || poly_b.is_empty() {
        return vec![];
    }
    let mut all_points: Vec<Point> = Vec::new();
    for p1 in poly_a {
        for p2 in poly_b {
            all_points
                .push((p1.0 as f64 + p2.0 as f64, p1.1 as f64 + p2.1 as f64));
        }
    }
    let hull = get_polygon_convex_hull(&all_points);
    if hull.len() < 3 {
        return vec![];
    }
    vec![hull.iter().map(|(x, y)| (*x as i64, *y as i64)).collect()]
}

fn to_clipper(polygon: &Polygon, scale: i64) -> IntPolygon {
    polygon
        .iter()
        .map(|(x, y)| ((x * scale as f64) as i64, (y * scale as f64) as i64))
        .collect()
}

fn from_clipper(polygon: &IntPolygon, scale: i64) -> Polygon {
    polygon
        .iter()
        .map(|(x, y)| (*x as f64 / scale as f64, *y as f64 / scale as f64))
        .collect()
}

/// Calculate the No-Fit Polygon (NFP) for two polygons.
///
/// Assumes polygons are convex for performance.
pub fn get_no_fit_polygon(
    stationary: &Polygon,
    orbiting: &Polygon,
) -> Vec<Polygon> {
    if stationary.is_empty() || orbiting.is_empty() {
        return vec![];
    }
    let scale = CLIPPER_SCALE;
    let static_path = to_clipper(stationary, scale);
    let orbiting_path = to_clipper(orbiting, scale);
    let orbiting_negated: IntPolygon =
        orbiting_path.iter().map(|(x, y)| (-*x, -*y)).collect();
    let nfp_paths =
        get_polygon_minkowski_sum_convex(&static_path, &orbiting_negated);
    let mut results = Vec::new();
    let first_pt = orbiting_path[0];
    for path in nfp_paths {
        let shifted: Polygon = path
            .iter()
            .map(|(x, y)| {
                (*x as f64 + first_pt.0 as f64, *y as f64 + first_pt.1 as f64)
            })
            .collect();
        results.push(from_clipper(
            &shifted
                .iter()
                .map(|(x, y)| (*x as i64, *y as i64))
                .collect(),
            scale,
        ));
    }
    results
}

/// Calculate the Inner-Fit Polygon (IFP) using a simple formula based on
/// bounding boxes, which is exact for axis-aligned rectangles and a robust
/// approximation for other convex shapes.
pub fn get_inner_fit_polygon(
    container: &Polygon,
    part: &Polygon,
) -> Vec<Polygon> {
    if container.is_empty() || part.is_empty() {
        return vec![];
    }
    let c_rect = get_polygon_bounds(container);
    let p_rect = get_polygon_bounds(part);
    let p_width = p_rect.2 - p_rect.0;
    let p_height = p_rect.3 - p_rect.1;
    let c_width = c_rect.2 - c_rect.0;
    let c_height = c_rect.3 - c_rect.1;
    if p_width > c_width + 1e-9 || p_height > c_height + 1e-9 {
        return vec![];
    }
    let ifp_min_x = c_rect.0 - p_rect.0;
    let ifp_max_x = c_rect.2 - p_rect.2;
    let ifp_min_y = c_rect.1 - p_rect.1;
    let ifp_max_y = c_rect.3 - p_rect.3;
    if ifp_min_x > ifp_max_x || ifp_min_y > ifp_max_y {
        return vec![];
    }
    let ifp = vec![
        (ifp_min_x, ifp_min_y),
        (ifp_max_x, ifp_min_y),
        (ifp_max_x, ifp_max_y),
        (ifp_min_x, ifp_max_y),
    ];
    vec![ifp]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_calculate_input_scale_empty() {
        let scale = calculate_input_scale(&[] as &[Polygon], 2147483647i64);
        assert!((scale - 214748364.7).abs() < 1e-4);
    }

    #[test]
    fn test_calculate_input_scale_simple() {
        let polys = vec![vec![(0.0, 0.0), (1.0, 0.0), (1.0, 1.0)]];
        let scale = calculate_input_scale(&polys, 2147483647i64);
        assert!((scale - 214748364.7).abs() < 1e-4);
    }

    #[test]
    fn test_calculate_input_scale_small() {
        let polys = vec![vec![(0.0, 0.0), (0.5, 0.0)]];
        let scale = calculate_input_scale(&polys, 2147483647i64);
        assert!((scale - 214748364.7).abs() < 1e-4);
    }

    #[test]
    fn test_get_polygon_minkowski_sum_convex_simple() {
        let poly_a = vec![(0i64, 0i64), (1i64, 0i64), (1i64, 1i64)];
        let poly_b = vec![(0i64, 0i64), (1i64, 0i64)];
        let result = get_polygon_minkowski_sum_convex(&poly_a, &poly_b);
        assert!(!result.is_empty());
        assert!(result[0].len() >= 3);
    }

    #[test]
    fn test_get_polygon_minkowski_sum_convex_empty() {
        let result = get_polygon_minkowski_sum_convex(&vec![], &vec![]);
        assert!(result.is_empty());
        let result = get_polygon_minkowski_sum_convex(
            &vec![(0i64, 0i64)],
            &vec![(0i64, 0i64)],
        );
        assert!(result.is_empty());
    }

    #[test]
    fn test_calculate_nfp_simple() {
        let stationary =
            vec![(0.0, 0.0), (10.0, 0.0), (10.0, 10.0), (0.0, 10.0)];
        let orbiting = vec![(2.0, 2.0), (6.0, 2.0), (6.0, 6.0), (2.0, 6.0)];
        let nfp = get_no_fit_polygon(&stationary, &orbiting);
        assert!(!nfp.is_empty());
    }

    #[test]
    fn test_calculate_nfp_empty() {
        let result = get_no_fit_polygon(&vec![], &vec![(0.0, 0.0), (1.0, 0.0)]);
        assert!(result.is_empty());
        let result = get_no_fit_polygon(&vec![(0.0, 0.0), (1.0, 0.0)], &vec![]);
        assert!(result.is_empty());
    }

    #[test]
    fn test_calculate_ifp_simple() {
        let container =
            vec![(0.0, 0.0), (10.0, 0.0), (10.0, 10.0), (0.0, 10.0)];
        let part = vec![(0.0, 0.0), (2.0, 0.0), (2.0, 2.0), (0.0, 2.0)];
        let ifp = get_inner_fit_polygon(&container, &part);
        assert!(!ifp.is_empty());
        assert_eq!(ifp[0].len(), 4);
        assert_eq!(ifp[0][0], (-0.0, -0.0));
        assert_eq!(ifp[0][1], (8.0, -0.0));
        assert_eq!(ifp[0][2], (8.0, 8.0));
        assert_eq!(ifp[0][3], (-0.0, 8.0));
    }

    #[test]
    fn test_calculate_ifp_part_larger_than_container() {
        let container = vec![(0.0, 0.0), (5.0, 0.0), (5.0, 5.0), (0.0, 5.0)];
        let part = vec![(0.0, 0.0), (10.0, 0.0), (10.0, 10.0), (0.0, 10.0)];
        let ifp = get_inner_fit_polygon(&container, &part);
        assert!(ifp.is_empty());
    }

    #[test]
    fn test_calculate_ifp_empty() {
        let result =
            get_inner_fit_polygon(&vec![], &vec![(0.0, 0.0), (1.0, 0.0)]);
        assert!(result.is_empty());
        let result =
            get_inner_fit_polygon(&vec![(0.0, 0.0), (1.0, 0.0)], &vec![]);
        assert!(result.is_empty());
    }

    #[test]
    fn test_to_clipper_roundtrip() {
        let poly = vec![(0.0, 0.0), (1.0, 0.0), (1.0, 1.0)];
        let scaled = to_clipper(&poly, CLIPPER_SCALE);
        let restored = from_clipper(&scaled, CLIPPER_SCALE);
        for (a, b) in poly.iter().zip(restored.iter()) {
            assert!((a.0 - b.0).abs() < 1e-9);
            assert!((a.1 - b.1).abs() < 1e-9);
        }
    }
}
