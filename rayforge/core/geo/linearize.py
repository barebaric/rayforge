import math
from typing import List, Tuple, Any


def linearize_arc(
    arc_cmd: Any, start_point: Tuple[float, float, float]
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
    # Use ~0.5mm segments for linearization
    num_segments = max(2, int(arc_len / 0.5))

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
