from typing import List, Tuple, Any, Optional
from .primitives import (
    find_closest_point_on_line_segment,
    find_closest_point_on_arc,
    get_arc_bounding_box,
)


def get_bounding_rect(
    commands: List[Any],
    include_travel: bool = False,
) -> Tuple[float, float, float, float]:
    """
    Returns a rectangle (x1, y1, x2, y2) that encloses the
    occupied area in the XY plane. This function is generic and works with
    both ops.Command and geo.Command lists.
    """
    occupied_points: List[Tuple[float, float, float]] = []
    last_point: Optional[Tuple[float, float, float]] = None
    for cmd in commands:
        # Use duck-typing on class names to remain generic
        cmd_type_name = cmd.__class__.__name__
        if (
            cmd_type_name == "MoveToCommand"
            and hasattr(cmd, "end")
            and cmd.end
        ):
            if include_travel:
                if last_point is not None:
                    occupied_points.append(last_point)
                occupied_points.append(cmd.end)
            last_point = cmd.end
        elif (
            cmd_type_name
            in ("LineToCommand", "ArcToCommand", "ScanLinePowerCommand")
            and hasattr(cmd, "end")
            and cmd.end
        ):
            start_point = last_point  # Capture start point for this command

            if start_point is not None:
                occupied_points.append(start_point)
            occupied_points.append(cmd.end)

            # For arcs, we must also consider the curve's extent.
            if cmd_type_name == "ArcToCommand" and start_point:
                arc_box = get_arc_bounding_box(
                    start_pos=start_point[:2],
                    end_pos=cmd.end[:2],
                    center_offset=cmd.center_offset,
                    clockwise=cmd.clockwise,
                )
                # By adding the min and max corners of the arc's true
                # bounding box, we ensure the final min/max calculation
                # will correctly encompass the arc's full curve.
                occupied_points.append((arc_box[0], arc_box[1], 0.0))
                occupied_points.append((arc_box[2], arc_box[3], 0.0))

            last_point = cmd.end

    if not occupied_points:
        return 0.0, 0.0, 0.0, 0.0

    xs = [p[0] for p in occupied_points if p]
    ys = [p[1] for p in occupied_points if p]
    if not xs or not ys:
        return 0.0, 0.0, 0.0, 0.0
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    return min_x, min_y, max_x, max_y


def get_total_distance(commands: List[Any]) -> float:
    """
    Calculates the total 2D path length for all moving commands in a list.
    """
    total = 0.0
    last: Optional[Tuple[float, float, float]] = None
    for cmd in commands:
        total += cmd.distance(last)
        # Update last point if the command was a move
        if hasattr(cmd, "end") and cmd.end is not None:
            last = cmd.end
    return total


def find_closest_point_on_path(
    commands: List[Any], x: float, y: float
) -> Optional[Tuple[int, float, Tuple[float, float]]]:
    """
    Finds the closest point on an entire path to a given 2D coordinate.
    """
    min_dist_sq = float("inf")
    closest_info: Optional[Tuple[int, float, Tuple[float, float]]] = None

    last_pos_3d: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    for i, cmd in enumerate(commands):
        # Use duck-typing on class names
        cmd_type_name = cmd.__class__.__name__

        if cmd_type_name == "MoveToCommand":
            if cmd.end:
                last_pos_3d = cmd.end
            continue
        if (
            cmd_type_name not in ("LineToCommand", "ArcToCommand")
            or not cmd.end
        ):
            continue

        start_pos = last_pos_3d

        if cmd_type_name == "LineToCommand":
            t, pt, dist_sq = find_closest_point_on_line_segment(
                start_pos[:2], cmd.end[:2], x, y
            )
            if dist_sq < min_dist_sq:
                min_dist_sq = dist_sq
                closest_info = (i, t, pt)

        elif cmd_type_name == "ArcToCommand":
            result = find_closest_point_on_arc(cmd, start_pos, x, y)
            if result:
                t_arc, pt_arc, dist_sq_arc = result
                if dist_sq_arc < min_dist_sq:
                    min_dist_sq = dist_sq_arc
                    closest_info = (i, t_arc, pt_arc)

        if hasattr(cmd, "end") and cmd.end:
            last_pos_3d = cmd.end

    return closest_info
