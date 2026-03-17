"""
Polyline smoothing using Gaussian filtering.

This module provides smoothing functions that can be reused by both
transformers and other components like contour detection.
"""

import math
from typing import List, Optional, Tuple

from .analysis import get_angle_at_vertex
from .linearize import resample_polyline
from .types import Point3D


def compute_gaussian_kernel(amount: int) -> Tuple[List[float], float]:
    """
    Compute a Gaussian kernel based on smoothing amount.

    Args:
        amount: Smoothing strength from 0 (none) to 100 (heavy).

    Returns:
        Tuple of (kernel, sigma) where kernel is a normalized list of weights.
    """
    if amount == 0:
        return [1.0], 0.0

    sigma = (amount / 100.0) * 5.0 + 0.1
    radius = math.ceil(sigma * 3)
    size = 2 * radius + 1
    kernel = [0.0] * size
    kernel_sum = 0.0

    for i in range(size):
        x = i - radius
        val = math.exp(-0.5 * (x / sigma) ** 2)
        kernel[i] = val
        kernel_sum += val

    return [k / kernel_sum for k in kernel], sigma


def smooth_sub_segment(
    points: List[Point3D],
    kernel: List[float],
) -> List[Point3D]:
    """
    Apply Gaussian kernel to an open list of points.

    Endpoints are preserved.

    Args:
        points: List of 3D points to smooth.
        kernel: Normalized Gaussian kernel.

    Returns:
        Smoothed list of points with same length.
    """
    num_pts = len(points)
    if num_pts < 3:
        return points

    kernel_radius = (len(kernel) - 1) // 2
    smoothed = [points[0]]

    for i in range(1, num_pts - 1):
        new_x, new_y = 0.0, 0.0
        for k_idx, k_weight in enumerate(kernel):
            p_idx = max(0, min(num_pts - 1, i - kernel_radius + k_idx))
            point = points[p_idx]
            new_x += point[0] * k_weight
            new_y += point[1] * k_weight
        smoothed.append((new_x, new_y, points[i][2]))

    smoothed.append(points[-1])
    return smoothed


def smooth_circularly(
    points: List[Point3D],
    kernel: List[float],
) -> List[Point3D]:
    """
    Apply wrapping Gaussian filter to a closed loop.

    Args:
        points: List of 3D points forming a closed loop.
        kernel: Normalized Gaussian kernel.

    Returns:
        Smoothed list with the first point appended at the end to close.
    """
    num_pts = len(points)
    if num_pts < 3:
        return points

    kernel_radius = (len(kernel) - 1) // 2
    smoothed = []

    for i in range(num_pts):
        new_x, new_y = 0.0, 0.0
        for k_idx, k_weight in enumerate(kernel):
            p_idx = (i - kernel_radius + k_idx + num_pts) % num_pts
            point = points[p_idx]
            new_x += point[0] * k_weight
            new_y += point[1] * k_weight
        smoothed.append((new_x, new_y, points[i][2]))

    if smoothed:
        smoothed.append(smoothed[0])
    return smoothed


def smooth_polyline(
    points: List[Point3D],
    amount: int,
    corner_angle_threshold: float = 45.0,
    is_closed: Optional[bool] = None,
) -> List[Point3D]:
    """
    Smooth a polyline using Gaussian filtering with corner preservation.

    Angles sharper than the threshold are preserved as anchors.

    Args:
        points: List of 3D points to smooth.
        amount: Smoothing strength from 0 (none) to 100 (heavy).
        corner_angle_threshold: Internal angles smaller than this (in degrees)
                               are preserved as corners.
        is_closed: Whether the polyline is closed. If None, auto-detected.

    Returns:
        Smoothed list of points.
    """
    if len(points) < 3 or amount == 0:
        return points

    kernel, sigma = compute_gaussian_kernel(amount)
    if len(kernel) <= 1:
        return points

    if is_closed is None:
        tol = 1e-6
        is_closed = (
            len(points) >= 3
            and math.hypot(
                points[0][0] - points[-1][0], points[0][1] - points[-1][1]
            )
            < tol
        )

    work_points = points[:-1] if is_closed else points
    max_len = max(0.1, sigma / 4.0)
    prepared_points = resample_polyline(work_points, max_len, is_closed)
    num_points = len(prepared_points)

    if num_points < 3:
        return points

    corner_threshold_rad = math.radians(corner_angle_threshold)
    anchor_indices = set()

    if not is_closed:
        anchor_indices.update([0, num_points - 1])

    for i in range(num_points):
        p_prev = prepared_points[(i - 1 + num_points) % num_points]
        p_curr = prepared_points[i]
        p_next = prepared_points[(i + 1) % num_points]
        angle = get_angle_at_vertex(p_prev, p_curr, p_next)

        if angle < corner_threshold_rad and not math.isclose(
            angle, corner_threshold_rad
        ):
            anchor_indices.add(i)

    sorted_anchors = sorted(list(anchor_indices))
    final_points: List[Point3D] = []

    if is_closed:
        if not sorted_anchors:
            return smooth_circularly(prepared_points, kernel)

        num_anchors = len(sorted_anchors)
        for i in range(num_anchors):
            start_idx = sorted_anchors[i]
            end_idx = sorted_anchors[(i + 1) % num_anchors]
            if start_idx < end_idx:
                sub_seg = prepared_points[start_idx : end_idx + 1]
            else:
                sub_seg = (
                    prepared_points[start_idx:]
                    + prepared_points[: end_idx + 1]
                )

            smoothed_sub = smooth_sub_segment(sub_seg, kernel)
            final_points.extend(smoothed_sub[:-1])

        if final_points:
            final_points.append(final_points[0])
        return final_points
    else:
        if len(sorted_anchors) < 2:
            return smooth_sub_segment(prepared_points, kernel)

        last_anchor_idx = sorted_anchors[0]
        for i in range(1, len(sorted_anchors)):
            anchor_idx = sorted_anchors[i]
            sub_seg = prepared_points[last_anchor_idx : anchor_idx + 1]
            smoothed_sub = smooth_sub_segment(sub_seg, kernel)
            final_points.extend(smoothed_sub[:-1])
            last_anchor_idx = anchor_idx

        final_points.append(prepared_points[-1])
        return final_points
