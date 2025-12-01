import math
from typing import List, Tuple, Any
import numpy as np


def linearize_arc(
    arc_cmd: Any,
    start_point: Tuple[float, float, float],
    resolution: float = 0.1,
) -> List[Tuple[Tuple[float, float, float], Tuple[float, float, float]]]:
    """
    Converts an arc command into a list of line segments.

    This function is generic and relies on duck typing for the `arc_cmd`
    object, which is expected to have `.end`, `.center_offset`, and
    `.clockwise` attributes.

    Args:
        arc_cmd: An object representing the arc (e.g., ops.ArcToCommand or
                 geometry.ArcToCommand).
        start_point: The (x, y, z) starting point of the arc.
        resolution: percentage of arc length of each step

    Returns:
        A list of tuples, where each tuple represents a line segment
        as ((start_x, start_y, start_z), (end_x, end_y, end_z)).
    """
    segments: List[
        Tuple[Tuple[float, float, float], Tuple[float, float, float]]
    ] = []
    p0 = start_point
    p1 = arc_cmd.end
    z0, z1 = p0[2], p1[2]

    center = (
        p0[0] + arc_cmd.center_offset[0],
        p0[1] + arc_cmd.center_offset[1],
    )

    radius_start = math.dist(p0[:2], center)
    radius_end = math.dist(p1[:2], center)

    # If the start point is the center, it's just a line to the end.
    if radius_start == 0:
        return [(p0, p1)]

    start_angle = math.atan2(p0[1] - center[1], p0[0] - center[0])
    end_angle = math.atan2(p1[1] - center[1], p1[0] - center[0])
    angle_range = end_angle - start_angle
    if arc_cmd.clockwise:
        if angle_range > 0:
            angle_range -= 2 * math.pi
    else:
        if angle_range < 0:
            angle_range += 2 * math.pi

    # Use the average radius to get a better estimate for arc length
    avg_radius = (radius_start + radius_end) / 2
    arc_len = abs(angle_range * avg_radius)
    num_segments = max(2, int(arc_len / resolution))

    prev_pt = p0
    for i in range(1, num_segments + 1):
        t = i / num_segments
        # Interpolate radius and angle to handle imperfectly defined arcs
        radius = radius_start + (radius_end - radius_start) * t
        angle = start_angle + angle_range * t
        z = z0 + (z1 - z0) * t
        next_pt = (
            center[0] + radius * math.cos(angle),
            center[1] + radius * math.sin(angle),
            z,
        )
        segments.append((prev_pt, next_pt))
        prev_pt = next_pt
    return segments


def linearize_bezier(
    p0: Tuple[float, ...],
    c1: Tuple[float, ...],
    c2: Tuple[float, ...],
    p1: Tuple[float, ...],
    num_steps: int,
) -> List[Tuple[Tuple[float, ...], Tuple[float, ...]]]:
    """
    Converts a cubic Bézier curve into a list of line segments.
    This function is generic and supports points of any dimension (e.g., 2D
    or 3D).

    Args:
        p0: The starting point of the curve.
        c1: The first control point.
        c2: The second control point.
        p1: The ending point of the curve.
        num_steps: The number of line segments to approximate the curve with.

    Returns:
        A list of tuples, where each tuple is a line segment represented by
        (start_point, end_point).
    """
    if num_steps < 1:
        return []

    points_np = np.array([p0, c1, c2, p1])
    t_values = np.linspace(0, 1, num_steps + 1)

    # Evaluate the cubic Bézier formula for each value of t
    interpolated_points_np = np.array(
        [
            (1 - t) ** 3 * points_np[0]
            + 3 * (1 - t) ** 2 * t * points_np[1]
            + 3 * (1 - t) * t**2 * points_np[2]
            + t**3 * points_np[3]
            for t in t_values
        ]
    )
    # Convert back to a list of tuples
    interpolated_points = [tuple(p) for p in interpolated_points_np]

    # Create segments from the list of points
    return [
        (interpolated_points[i], interpolated_points[i + 1])
        for i in range(num_steps)
    ]


def resample_polyline(
    points: List[Tuple[float, float, float]],
    max_segment_length: float,
    is_closed: bool,
) -> List[Tuple[float, float, float]]:
    """
    Resamples a polyline, adding points to increase its density such that
    no segment is longer than `max_segment_length`.
    """
    if not points:
        return []

    new_points = [points[0]]
    num_segments = len(points) if is_closed else len(points) - 1

    for i in range(num_segments):
        p1 = points[i]
        p2 = points[(i + 1) % len(points)]  # Wraps for closed paths
        dist = math.hypot(p2[0] - p1[0], p2[1] - p1[1])

        if dist > max_segment_length:
            # If a segment is too long, subdivide it.
            num_sub = math.ceil(dist / max_segment_length)
            for j in range(1, int(num_sub)):
                t = j / num_sub
                # Linear interpolation to create new points.
                px = p1[0] * (1 - t) + p2[0] * t
                py = p1[1] * (1 - t) + p2[1] * t
                new_points.append((px, py, p1[2]))

        # Add the original endpoint, avoiding duplication for closed paths.
        if not (is_closed and i == num_segments - 1):
            new_points.append(p2)

    return new_points
