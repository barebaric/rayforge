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
) -> bool:
    """Core logic to check for intersections between two command lists."""
    for i in range(len(commands1)):
        cmd1 = commands1[i]
        if not isinstance(cmd1, (LineToCommand, ArcToCommand)):
            continue

        start_idx_j = i + 1 if is_self_check else 0
        for j in range(start_idx_j, len(commands2)):
            cmd2 = commands2[j]
            if not isinstance(cmd2, (LineToCommand, ArcToCommand)):
                continue

            # Get the original (pre-linearization) endpoints for adjacency
            # check
            start1 = _get_command_start_point(commands1, i)
            end1 = cmd1.end
            start2 = _get_command_start_point(commands2, j)
            end2 = cmd2.end
            if not end1 or not end2:
                continue

            # Determine if the two original commands share a vertex
            shared_vertex = None
            if is_self_check:
                p1, p2, p3, p4 = start1[:2], end1[:2], start2[:2], end2[:2]
                if all(
                    math.isclose(a, b, abs_tol=1e-9) for a, b in zip(p2, p3)
                ):
                    shared_vertex = p2
                elif all(
                    math.isclose(a, b, abs_tol=1e-9) for a, b in zip(p1, p4)
                ):
                    shared_vertex = p1
                elif all(
                    math.isclose(a, b, abs_tol=1e-9) for a, b in zip(p1, p3)
                ):
                    shared_vertex = p1
                elif all(
                    math.isclose(a, b, abs_tol=1e-9) for a, b in zip(p2, p4)
                ):
                    shared_vertex = p2

            segments1 = _get_segments_for_command(commands1, i)
            segments2 = _get_segments_for_command(commands2, j)

            for seg1_p1, seg1_p2 in segments1:
                for seg2_p1, seg2_p2 in segments2:
                    intersection = line_segment_intersection(
                        seg1_p1[:2], seg1_p2[:2], seg2_p1[:2], seg2_p2[:2]
                    )

                    if intersection:
                        if shared_vertex:
                            # If adjacent, ignore intersections at shared
                            # vertex
                            dist_sq = (
                                intersection[0] - shared_vertex[0]
                            ) ** 2 + (intersection[1] - shared_vertex[1]) ** 2
                            if dist_sq < 1e-12:
                                continue  # It's a connection, not a crossing

                        # Found a valid intersection
                        return True
    return False


def check_self_intersection(commands: List[Command]) -> bool:
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
            # Start new subpath with the MoveTo for context
            current_subpath = [cmd]
        elif isinstance(cmd, (LineToCommand, ArcToCommand)):
            if not current_subpath:  # Path starts with a drawing command
                current_subpath.append(MoveToCommand((0.0, 0.0, 0.0)))
            current_subpath.append(cmd)

    if len(current_subpath) > 1:
        subpaths.append(current_subpath)

    for subpath_commands in subpaths:
        if _commands_intersect(
            subpath_commands, subpath_commands, is_self_check=True
        ):
            return True

    return False


def check_intersection(
    commands1: List[Command], commands2: List[Command]
) -> bool:
    """Checks if two paths defined by command lists intersect."""
    return _commands_intersect(commands1, commands2, is_self_check=False)
