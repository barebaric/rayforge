//! Smooth: Polyline smoothing using Gaussian filtering.
//!
//! Provides functions for computing Gaussian kernels and applying them
//! to open or closed polylines with optional corner preservation.

use crate::analysis::get_angle_at_vertex;
use crate::types::Point3D;

/// Compute a normalized Gaussian kernel based on smoothing amount.
///
/// `amount` ranges from 0 (none) to 200 (very heavy). The sigma is derived
/// as `(amount / 100.0) * 5.0 + 0.1`, and the radius is `ceil(sigma * 3)`.
/// Returns `(kernel, sigma)` where `kernel` is a normalized list of weights
/// that sums to 1.0.
pub fn compute_gaussian_kernel(amount: i32) -> (Vec<f64>, f64) {
    if amount == 0 {
        return (vec![1.0], 0.0);
    }

    let sigma = (amount as f64 / 100.0) * 5.0 + 0.1;
    let radius = (sigma * 3.0).ceil() as i32;
    let size = (2 * radius + 1) as usize;
    let mut kernel = vec![0.0; size];
    let mut kernel_sum = 0.0;

    for (i, k) in kernel.iter_mut().enumerate() {
        let x = i as f64 - radius as f64;
        let val = (-0.5 * (x / sigma).powi(2)).exp();
        *k = val;
        kernel_sum += val;
    }

    let norm: Vec<f64> = kernel.iter().map(|k| k / kernel_sum).collect();
    (norm, sigma)
}

/// Apply a Gaussian kernel to an open list of 3D points. Endpoints are preserved.
pub fn smooth_sub_segment(points: &[Point3D], kernel: &[f64]) -> Vec<Point3D> {
    let num_pts = points.len();
    if num_pts < 3 {
        return points.to_vec();
    }

    let kernel_radius = (kernel.len() - 1) / 2;
    let mut smoothed = vec![points[0]];

    for i in 1..(num_pts - 1) {
        let mut new_x = 0.0;
        let mut new_y = 0.0;
        for (k_idx, k_weight) in kernel.iter().enumerate() {
            let p_idx = (i as i32 - kernel_radius as i32 + k_idx as i32)
                .clamp(0, num_pts as i32 - 1) as usize;
            let pt = points[p_idx];
            new_x += pt.0 * k_weight;
            new_y += pt.1 * k_weight;
        }
        smoothed.push((new_x, new_y, points[i].2));
    }

    smoothed.push(points[num_pts - 1]);
    smoothed
}

/// Apply a wrapping Gaussian filter to a closed loop of points.
pub fn smooth_circularly(points: &[Point3D], kernel: &[f64]) -> Vec<Point3D> {
    let num_pts = points.len();
    if num_pts < 3 {
        return points.to_vec();
    }

    let kernel_radius = (kernel.len() - 1) / 2;
    let mut smoothed = Vec::with_capacity(num_pts);

    for i in 0..num_pts {
        let mut new_x = 0.0;
        let mut new_y = 0.0;
        for (k_idx, k_weight) in kernel.iter().enumerate() {
            let p_idx = (i as i32 - kernel_radius as i32 + k_idx as i32)
                .rem_euclid(num_pts as i32) as usize;
            let pt = points[p_idx];
            new_x += pt.0 * k_weight;
            new_y += pt.1 * k_weight;
        }
        smoothed.push((new_x, new_y, points[i].2));
    }

    if !smoothed.is_empty() {
        smoothed.push(smoothed[0]);
    }
    smoothed
}

/// Resample a polyline so that no segment is longer than `max_segment_length`.
/// New points are added by linear interpolation along existing segments.
pub fn resample_polyline(
    points: &[Point3D],
    max_segment_length: f64,
    is_closed: bool,
) -> Vec<Point3D> {
    if points.is_empty() {
        return vec![];
    }

    let mut new_points = vec![points[0]];
    let num_segments = if is_closed {
        points.len()
    } else {
        points.len() - 1
    };

    for i in 0..num_segments {
        let p1 = points[i];
        let p2 = points[(i + 1) % points.len()];
        let dx = p2.0 - p1.0;
        let dy = p2.1 - p1.1;
        let dist = dx.hypot(dy);

        if dist > max_segment_length {
            let num_sub = (dist / max_segment_length).ceil() as i32;
            for j in 1..num_sub {
                let t = j as f64 / num_sub as f64;
                let px = p1.0 * (1.0 - t) + p2.0 * t;
                let py = p1.1 * (1.0 - t) + p2.1 * t;
                new_points.push((px, py, p1.2));
            }
        }

        if !(is_closed && i == num_segments - 1) {
            new_points.push(p2);
        }
    }

    new_points
}

/// Smooth a polyline using Gaussian filtering with optional corner preservation.
///
/// Angles sharper than `corner_angle_threshold` are preserved as anchors
/// and not smoothed. The polyline is first resampled to ensure sufficient
/// point density for the Gaussian kernel.
pub fn smooth_polyline(
    points: &[Point3D],
    amount: i32,
    corner_angle_threshold: f64,
    is_closed: Option<bool>,
) -> Vec<Point3D> {
    if points.len() < 3 || amount == 0 {
        return points.to_vec();
    }

    let (kernel, sigma) = compute_gaussian_kernel(amount);
    if kernel.len() <= 1 {
        return points.to_vec();
    }

    // Auto-detect closed paths if not specified
    let is_closed = is_closed.unwrap_or_else(|| {
        if points.len() >= 3 {
            let tol = 1e-6;
            (points[0].0 - points[points.len() - 1].0)
                .hypot(points[0].1 - points[points.len() - 1].1)
                < tol
        } else {
            false
        }
    });

    // Remove the duplicate endpoint for closed paths before resampling
    let work_points = if is_closed {
        &points[..points.len() - 1]
    } else {
        points
    };
    let max_len = (0.1_f64).max(sigma / 4.0);
    let prepared = resample_polyline(work_points, max_len, is_closed);
    let num_points = prepared.len();

    if num_points < 3 {
        return points.to_vec();
    }

    // Identify corners (sharp angles) to preserve
    let corner_threshold_rad = corner_angle_threshold.to_radians();
    let mut anchor_indices: Vec<usize> = Vec::new();

    if !is_closed {
        anchor_indices.push(0);
        anchor_indices.push(num_points - 1);
    }

    for i in 0..num_points {
        let p_prev = prepared[(i + num_points - 1) % num_points];
        let p_curr = prepared[i];
        let p_next = prepared[(i + 1) % num_points];
        let angle = get_angle_at_vertex(
            (p_prev.0, p_prev.1),
            (p_curr.0, p_curr.1),
            (p_next.0, p_next.1),
        );

        if angle < corner_threshold_rad
            && !approx_equal(angle, corner_threshold_rad)
            && !anchor_indices.contains(&i)
        {
            anchor_indices.push(i);
        }
    }

    anchor_indices.sort_unstable();
    let mut final_points: Vec<Point3D> = Vec::new();

    if is_closed {
        if anchor_indices.is_empty() {
            return smooth_circularly(&prepared, &kernel);
        }

        let num_anchors = anchor_indices.len();
        for i in 0..num_anchors {
            let start_idx = anchor_indices[i];
            let end_idx = anchor_indices[(i + 1) % num_anchors];
            let sub_seg: Vec<Point3D> = if start_idx < end_idx {
                prepared[start_idx..=end_idx].to_vec()
            } else {
                let mut seg: Vec<Point3D> = prepared[start_idx..].to_vec();
                seg.extend_from_slice(&prepared[..=end_idx]);
                seg
            };
            let smoothed_sub = smooth_sub_segment(&sub_seg, &kernel);
            for p in smoothed_sub.iter().take(smoothed_sub.len() - 1) {
                final_points.push(*p);
            }
        }

        if !final_points.is_empty() {
            final_points.push(final_points[0]);
        }
        final_points
    } else {
        if anchor_indices.len() < 2 {
            return smooth_sub_segment(&prepared, &kernel);
        }

        let mut last_anchor = anchor_indices[0];
        for &anchor_idx in anchor_indices.iter().skip(1) {
            let sub_seg: Vec<Point3D> =
                prepared[last_anchor..=anchor_idx].to_vec();
            let smoothed_sub = smooth_sub_segment(&sub_seg, &kernel);
            for p in smoothed_sub.iter().take(smoothed_sub.len() - 1) {
                final_points.push(*p);
            }
            last_anchor = anchor_idx;
        }

        final_points.push(prepared[num_points - 1]);
        final_points
    }
}

fn approx_equal(a: f64, b: f64) -> bool {
    (a - b).abs() < 1e-12
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_compute_gaussian_kernel_zero() {
        let (kernel, sigma) = compute_gaussian_kernel(0);
        assert_eq!(kernel, vec![1.0]);
        assert!((sigma - 0.0).abs() < 1e-9);
    }

    #[test]
    fn test_compute_gaussian_kernel_nonzero() {
        let (kernel, sigma) = compute_gaussian_kernel(50);
        assert!(kernel.len() > 1);
        assert!(sigma > 0.0);
        let sum: f64 = kernel.iter().sum();
        assert!((sum - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_smooth_sub_segment_trivial() {
        let points = vec![(0.0, 0.0, 0.0), (1.0, 1.0, 0.0)];
        let kernel = vec![1.0];
        let result = smooth_sub_segment(&points, &kernel);
        assert_eq!(result.len(), 2);
    }

    #[test]
    fn test_resample_polyline() {
        let points = vec![(0.0, 0.0, 0.0), (10.0, 0.0, 0.0)];
        let result = resample_polyline(&points, 1.0, false);
        assert_eq!(result.len(), 11);
    }

    #[test]
    fn test_resample_polyline_closed() {
        let points = vec![(0.0, 0.0, 0.0), (10.0, 0.0, 0.0), (0.0, 0.0, 0.0)];
        let result = resample_polyline(&points, 1.0, true);
        assert!(result.len() > 3);
    }
}
