import math
from typing import List, Tuple, Any, Optional

from .linearize import linearize_arc


def get_subpath_vertices(
    commands: List[Any], start_cmd_index: int
) -> List[Tuple[float, float]]:
    """
    Extracts all 2D vertices for a single continuous subpath starting at a
    given MoveToCommand index, linearizing any arcs.
    """
    from .geometry import MoveToCommand, LineToCommand, ArcToCommand

    vertices: List[Tuple[float, float]] = []
    if start_cmd_index >= len(commands):
        return []
    last_pos_3d = commands[start_cmd_index].end or (0.0, 0.0, 0.0)
    vertices.append(last_pos_3d[:2])

    for i in range(start_cmd_index + 1, len(commands)):
        cmd = commands[i]
        if isinstance(cmd, MoveToCommand):
            # End of the subpath
            break
        if not isinstance(cmd, (LineToCommand, ArcToCommand)) or not cmd.end:
            continue

        if isinstance(cmd, LineToCommand):
            vertices.append(cmd.end[:2])
        elif isinstance(cmd, ArcToCommand):
            segments = linearize_arc(cmd, last_pos_3d)
            for _, p2 in segments:
                vertices.append(p2[:2])
        last_pos_3d = cmd.end

    return vertices


def get_path_winding_order(commands: List[Any], segment_index: int) -> str:
    """
    Determines winding order ('cw', 'ccw', 'unknown') for the subpath at a
    given index.
    """
    from .geometry import MoveToCommand

    # Find the start of the subpath for the given segment
    subpath_start_index = -1
    for i in range(segment_index, -1, -1):
        if isinstance(commands[i], MoveToCommand):
            subpath_start_index = i
            break
    if subpath_start_index == -1:
        return "unknown"

    vertices = get_subpath_vertices(commands, subpath_start_index)
    if len(vertices) < 3:
        return "unknown"  # Not a closed polygon

    # Shoelace formula to calculate signed area
    area = 0.0
    for i in range(len(vertices)):
        p1 = vertices[i]
        p2 = vertices[(i + 1) % len(vertices)]
        area += (p1[0] * p2[1]) - (p2[0] * p1[1])

    # Convention: positive area is CCW, negative is CW in a Y-up system
    if abs(area) < 1e-9:
        return "unknown"
    elif area > 0:
        return "ccw"
    else:
        return "cw"


def get_point_and_tangent_at(
    commands: List[Any], segment_index: int, t: float
) -> Optional[Tuple[Tuple[float, float], Tuple[float, float]]]:
    """
    Calculates the 2D point and tangent vector at a parameter 't' along a
    segment.
    """
    from .geometry import LineToCommand, ArcToCommand, MovingCommand

    cmd = commands[segment_index]
    if not isinstance(cmd, (LineToCommand, ArcToCommand)) or not cmd.end:
        return None

    # Find the start point of this segment
    start_pos_3d: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    for i in range(segment_index - 1, -1, -1):
        prev_cmd = commands[i]
        if isinstance(prev_cmd, MovingCommand) and prev_cmd.end:
            start_pos_3d = prev_cmd.end
            break

    p0 = start_pos_3d[:2]
    p1 = cmd.end[:2]

    if isinstance(cmd, LineToCommand):
        point = (p0[0] + t * (p1[0] - p0[0]), p0[1] + t * (p1[1] - p0[1]))
        tangent_vec = (p1[0] - p0[0], p1[1] - p0[1])
    elif isinstance(cmd, ArcToCommand):
        center = (
            p0[0] + cmd.center_offset[0],
            p0[1] + cmd.center_offset[1],
        )
        start_angle = math.atan2(p0[1] - center[1], p0[0] - center[0])
        end_angle = math.atan2(p1[1] - center[1], p1[0] - center[0])
        angle_range = end_angle - start_angle

        if cmd.clockwise:
            if angle_range > 0:
                angle_range -= 2 * math.pi
        else:
            if angle_range < 0:
                angle_range += 2 * math.pi

        current_angle = start_angle + t * angle_range
        radius_start = math.dist(p0, center)
        radius_end = math.dist(p1, center)
        radius = radius_start + t * (radius_end - radius_start)

        point = (
            center[0] + radius * math.cos(current_angle),
            center[1] + radius * math.sin(current_angle),
        )

        radius_vec = (point[0] - center[0], point[1] - center[1])
        if cmd.clockwise:
            tangent_vec = (radius_vec[1], -radius_vec[0])
        else:
            tangent_vec = (-radius_vec[1], radius_vec[0])
    else:
        return None

    norm = math.hypot(tangent_vec[0], tangent_vec[1])
    if norm < 1e-9:
        return point, (1.0, 0.0)

    normalized_tangent = (tangent_vec[0] / norm, tangent_vec[1] / norm)
    return point, normalized_tangent


def get_outward_normal_at(
    commands: List[Any], segment_index: int, t: float
) -> Optional[Tuple[float, float]]:
    """
    Calculates the outward-pointing normal vector for a point on a closed
    path.
    """
    winding = get_path_winding_order(commands, segment_index)
    if winding == "unknown":
        return None

    result = get_point_and_tangent_at(commands, segment_index, t)
    if not result:
        return None

    _, tangent = result
    tx, ty = tangent

    if winding == "ccw":
        return (ty, -tx)
    else:  # winding == "cw"
        return (-ty, tx)
