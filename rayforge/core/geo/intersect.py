import math
from typing import List, Tuple

from .geometry import (
    Command,
    LineToCommand,
    ArcToCommand,
    MoveToCommand,
    MovingCommand,
)
from .linearize import linearize_arc
from .primitives import line_segment_intersection


def _get_command_start_point(
    commands: List[Command], index: int
) -> Tuple[float, float, float]:
    """Finds the start point of the command at the given index."""
    for i in range(index - 1, -1, -1):
        prev_cmd = commands[i]
        if isinstance(prev_cmd, MovingCommand) and prev_cmd.end:
            return prev_cmd.end
    return 0.0, 0.0, 0.0


def _get_segments_for_command(
    commands: List[Command], index: int
) -> List[Tuple[Tuple[float, float, float], Tuple[float, float, float]]]:
    """
    Returns a list of linearized line segments for a given command.
    For a LineToCommand, this is a list with one segment.
    For an ArcToCommand, this is a list of its linearized segments.
    """
    cmd = commands[index]
    if not isinstance(cmd, (LineToCommand, ArcToCommand)) or not cmd.end:
        return []

    start_point = _get_command_start_point(commands, index)

    if isinstance(cmd, LineToCommand):
        return [(start_point, cmd.end)]
    elif isinstance(cmd, ArcToCommand):
        return linearize_arc(cmd, start_point)
    return []


def _commands_intersect(
    commands1: List[Command],
    commands2: List[Command],
    is_self_check: bool = False,
    fail_on_t_junction: bool = False,
) -> bool:
    """Core logic to check for intersections between two command lists."""
    for i in range(len(commands1)):
        cmd1 = commands1[i]
        if not isinstance(cmd1, (LineToCommand, ArcToCommand)) or not cmd1.end:
            continue

        # Check against subsequent segments. i + 1 is used to check adjacent
        # segments, which is necessary for cases like an arc being crossed
        # by the following line segment.
        start_idx_j = i + 1 if is_self_check else 0
        for j in range(start_idx_j, len(commands2)):
            cmd2 = commands2[j]
            if (
                not isinstance(cmd2, (LineToCommand, ArcToCommand))
                or not cmd2.end
            ):
                continue

            segments1 = _get_segments_for_command(commands1, i)
            segments2 = _get_segments_for_command(commands2, j)

            for seg1_p1, seg1_p2 in segments1:
                for seg2_p1, seg2_p2 in segments2:
                    intersection = line_segment_intersection(
                        seg1_p1[:2], seg1_p2[:2], seg2_p1[:2], seg2_p2[:2]
                    )

                    if intersection:
                        is_adjacent_check = is_self_check and (j == i + 1)

                        # Case 1: Adjacent segments in a self-check.
                        if is_adjacent_check:
                            # An intersection is only ignored if it's their
                            # shared vertex. Any other intersection is a true
                            # self-intersection.
                            shared_vertex = cmd1.end[:2]
                            is_at_shared_vertex = all(
                                math.isclose(a, b, abs_tol=1e-9)
                                for a, b in zip(intersection, shared_vertex)
                            )
                            if is_at_shared_vertex:
                                continue  # Ignore the normal connection point.
                            else:
                                return True  # It's a true self-intersection.

                        # Case 2: Non-adjacent segments. Here, an intersection
                        # at any vertex is considered a T-junction or a shared
                        # corner, which can be ignored.
                        is_at_endpoint1 = all(
                            math.isclose(a, b, abs_tol=1e-9)
                            for a, b in zip(intersection, seg1_p1[:2])
                        ) or all(
                            math.isclose(a, b, abs_tol=1e-9)
                            for a, b in zip(intersection, seg1_p2[:2])
                        )

                        is_at_endpoint2 = all(
                            math.isclose(a, b, abs_tol=1e-9)
                            for a, b in zip(intersection, seg2_p1[:2])
                        ) or all(
                            math.isclose(a, b, abs_tol=1e-9)
                            for a, b in zip(intersection, seg2_p2[:2])
                        )

                        is_at_vertex = is_at_endpoint1 or is_at_endpoint2

                        if (
                            is_self_check
                            and is_at_vertex
                            and not fail_on_t_junction
                        ):
                            continue

                        # It's a "real" crossing intersection.
                        return True
    return False


def check_self_intersection(
    commands: List[Command], fail_on_t_junction: bool = False
) -> bool:
    """
    Checks if a path defined by a list of commands self-intersects.

    This function correctly handles geometries with multiple disjoint subpaths,
    only checking for self-intersections within each subpath.
    """
    subpaths: List[List[Command]] = []
    current_subpath: List[Command] = []
    for cmd in commands:
        if isinstance(cmd, MoveToCommand):
            if len(current_subpath) > 1:
                subpaths.append(current_subpath)
            current_subpath = [cmd]
        elif isinstance(cmd, (LineToCommand, ArcToCommand)):
            if not current_subpath:
                current_subpath.append(MoveToCommand((0.0, 0.0, 0.0)))
            current_subpath.append(cmd)

    if len(current_subpath) > 1:
        subpaths.append(current_subpath)

    for subpath_commands in subpaths:
        if _commands_intersect(
            subpath_commands,
            subpath_commands,
            is_self_check=True,
            fail_on_t_junction=fail_on_t_junction,
        ):
            return True

    return False


def check_intersection(
    commands1: List[Command],
    commands2: List[Command],
    fail_on_t_junction: bool = False,
) -> bool:
    """Checks if two paths defined by command lists intersect."""
    # Note: fail_on_t_junction is not used for non-self checks, as a T-junction
    # between two separate paths is always a valid intersection.
    return _commands_intersect(
        commands1,
        commands2,
        is_self_check=False,
        fail_on_t_junction=fail_on_t_junction,
    )
