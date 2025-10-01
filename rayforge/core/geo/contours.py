from __future__ import annotations
import math
from typing import List, TYPE_CHECKING
from .analysis import get_subpath_area
from .primitives import is_point_in_polygon

if TYPE_CHECKING:
    from .geometry import Geometry, Command


def close_geometry_gaps(
    geometry: Geometry, tolerance: float = 1e-6
) -> Geometry:
    """
    Closes small gaps in a Geometry object to form clean, connected paths.

    This function creates a new Geometry object with the modifications. The
    process is two-fold:
    1.  It iterates through each subpath (contour) and checks if the
        start and end points are within the given tolerance. If so, it
        snaps the end point to the start point, creating a perfectly
        closed shape.
    2.  It then checks for gaps between separate subpaths. If a `MoveTo`
        command starts very close to where the previous path ended, it
        replaces the `MoveTo` with a `LineTo`, effectively stitching the
        two paths together.

    Args:
        geometry: The input Geometry object.
        tolerance: The maximum distance between two points to be
                    considered "the same".

    Returns:
        A new, modified Geometry object.
    """
    from .geometry import MoveToCommand, MovingCommand, LineToCommand

    if len(geometry.commands) < 2:
        return geometry.copy()

    # Work on a copy to avoid modifying the original
    new_geo = geometry.copy()

    # Pass 1: Close gaps within each contour (intra-contour)
    contour_blocks: List[List[Command]] = []
    if new_geo.commands:
        current_block: List[Command] = []
        for cmd in new_geo.commands:
            if isinstance(cmd, MoveToCommand):
                if current_block:
                    contour_blocks.append(current_block)
                current_block = [cmd]
            else:
                if not current_block:  # Path starts with drawing cmd
                    current_block.append(MoveToCommand((0.0, 0.0, 0.0)))
                current_block.append(cmd)
        if current_block:
            contour_blocks.append(current_block)

    for block in contour_blocks:
        if len(block) < 2:
            continue
        start_cmd = block[0]
        end_cmd = block[-1]
        if (
            isinstance(start_cmd, MoveToCommand)
            and isinstance(end_cmd, MovingCommand)
            and start_cmd.end
            and end_cmd.end
        ):
            if math.dist(start_cmd.end, end_cmd.end) < tolerance:
                # Snap the end point to the start point. This modifies the
                # command object within the new_geo.commands list.
                end_cmd.end = start_cmd.end

    # Pass 2: Connect adjacent contours (inter-contour) using the modified list
    final_commands: List[Command] = []
    last_end_point: tuple[float, float, float] | None = None
    for cmd in new_geo.commands:
        if isinstance(cmd, MoveToCommand) and cmd.end:
            if (
                last_end_point is not None
                and math.dist(cmd.end, last_end_point) < tolerance
            ):
                # This MoveTo is a small jump; replace with a LineTo
                # to the exact previous endpoint to close the gap.
                final_commands.append(LineToCommand(last_end_point))
                # The logical position remains last_end_point
            else:
                final_commands.append(cmd)
                last_end_point = cmd.end
        elif isinstance(cmd, MovingCommand) and cmd.end:
            final_commands.append(cmd)
            last_end_point = cmd.end
        else:
            final_commands.append(cmd)  # Non-moving command

    new_geo.commands = final_commands
    return new_geo


def reverse_contour(contour: Geometry) -> Geometry:
    """Reverses the direction of a single-contour Geometry object."""
    from .geometry import (
        Geometry,
        MoveToCommand,
        LineToCommand,
        ArcToCommand,
        MovingCommand,
    )

    if contour.is_empty() or not isinstance(
        contour.commands[0], MoveToCommand
    ):
        return contour.copy()

    new_geo = Geometry()
    moving_cmds = [
        cmd for cmd in contour.commands if isinstance(cmd, MovingCommand)
    ]
    if not moving_cmds:
        return contour.copy()

    # The new path starts at the old path's end
    new_geo.move_to(*moving_cmds[-1].end)
    last_point = moving_cmds[-1].end

    # Iterate backwards through the moving commands
    for i in range(len(moving_cmds) - 1, 0, -1):
        end_cmd = moving_cmds[i]
        start_cmd = moving_cmds[i - 1]
        start_point = start_cmd.end

        if isinstance(end_cmd, LineToCommand):
            new_geo.line_to(*start_point)
        elif isinstance(end_cmd, ArcToCommand):
            # To reverse an arc, we swap start/end points and flip the flag.
            # The center offset must be recalculated from the new start point.
            center_abs = (
                start_point[0] + end_cmd.center_offset[0],
                start_point[1] + end_cmd.center_offset[1],
            )
            new_offset = (
                center_abs[0] - last_point[0],
                center_abs[1] - last_point[1],
            )
            new_geo.arc_to(
                x=start_point[0],
                y=start_point[1],
                z=start_point[2],
                i=new_offset[0],
                j=new_offset[1],
                clockwise=not end_cmd.clockwise,
            )
        last_point = start_point

    return new_geo


def normalize_winding_orders(contours: List[Geometry]) -> List[Geometry]:
    """
    Analyzes a list of contours and enforces the correct winding order
    (CCW for solids, CW for holes) based on their nesting level.

    This is crucial for ensuring that filtering algorithms based on the
    even-odd rule work correctly, especially with vector data from sources
    that do not guarantee winding order.
    """
    if not contours:
        return []

    # Pre-calculate vertices for each contour to avoid repeated computation
    contour_vertices = [c.segments()[0] for c in contours]
    contour_points_2d = [[p[:2] for p in v] for v in contour_vertices]

    normalized_contours: List[Geometry] = []
    for i, contour_to_normalize in enumerate(contours):
        if contour_to_normalize.is_empty():
            continue

        test_point = contour_points_2d[i][0]
        nesting_level = 0
        for j, other_points in enumerate(contour_points_2d):
            if i == j:
                continue
            # Use the raw point-in-polygon test, which is independent of
            # winding order. This breaks the circular dependency.
            if is_point_in_polygon(test_point, other_points):
                nesting_level += 1

        signed_area = get_subpath_area(contour_to_normalize.commands, 0)
        is_ccw = signed_area > 0
        is_nested_odd = nesting_level % 2 != 0

        # An outer shape (even nesting) should be CCW.
        # A hole (odd nesting) should be CW.
        # If the current state is wrong, reverse the contour.
        if (is_nested_odd and is_ccw) or (not is_nested_odd and not is_ccw):
            normalized_contours.append(reverse_contour(contour_to_normalize))
        else:
            normalized_contours.append(contour_to_normalize)

    return normalized_contours


def split_into_contours(geometry: Geometry) -> List[Geometry]:
    """
    Splits a Geometry object into a list of separate, single-contour
    Geometry objects. Each new object represents one continuous subpath
    that starts with a MoveToCommand.
    """
    if geometry.is_empty():
        return []

    from .geometry import Geometry, MoveToCommand

    contours: List[Geometry] = []
    current_geo: Geometry | None = None

    for cmd in geometry.commands:
        if isinstance(cmd, MoveToCommand):
            # A MoveTo command always starts a new contour.
            current_geo = Geometry()
            contours.append(current_geo)

        if current_geo is None:
            # This handles geometries that might not start with a
            # MoveToCommand. The first drawing command will implicitly
            # start the first contour.
            current_geo = Geometry()
            contours.append(current_geo)

        current_geo.add(cmd)

    # Filter out any empty geometries that might have been created
    return [c for c in contours if not c.is_empty()]


def filter_to_external_contours(contours: List[Geometry]) -> List[Geometry]:
    """
    Filters a list of single-contour geometries, returning only those
    that represent external paths (i.e., solid filled areas).

    This function is robust to the initial winding order of the input contours.
    It automatically normalizes all paths according to the even-odd fill rule
    and returns only the contours that represent solid material (those with
    a final CCW winding order).

    Args:
        contours: A list of Geometry objects, where each object is assumed
                  to represent a single, closed contour.

    Returns:
        A new list of Geometry objects containing only the external contours.
    """
    if not contours:
        return []

    # First, ensure all winding orders are correct relative to each other.
    normalized_contours = normalize_winding_orders(contours)

    # After normalization, any "external" or "solid" area will have a CCW
    # winding order (positive area). Holes will be CW (negative area).
    # We simply need to keep the CCW ones.
    return [
        c
        for c in normalized_contours
        if get_subpath_area(c.commands, 0) > 1e-9
    ]
