from __future__ import annotations
import math
import logging
from typing import List, Tuple, TYPE_CHECKING
from .analysis import get_subpath_area
from .primitives import is_point_in_polygon
from .split import split_into_contours

if TYPE_CHECKING:
    from .geometry import Geometry, Command

logger = logging.getLogger(__name__)


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


def split_inner_and_outer_contours(
    contours: List[Geometry],
) -> Tuple[List[Geometry], List[Geometry]]:
    """
    Splits a list of single-contour Geometries into two lists: external
    contours (solids) and internal ones (holes).

    This function robustly partitions the list into two groups based on the
    even-odd fill rule.

    Args:
        contours: A list of Geometry objects, where each object is assumed
                  to represent a single, closed contour.

    Returns:
        A tuple containing two lists: (internal_contours, external_contours).
    """
    if not contours:
        return [], []

    # filter_to_external_contours correctly identifies all contours that are
    # "solid" based on the even-odd rule.
    external_contours = filter_to_external_contours(contours)
    external_set = set(external_contours)

    # All other contours are, by definition, "internal" (holes).
    internal_contours = [c for c in contours if c not in external_set]

    return internal_contours, external_contours


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

    count = len(contours)

    # 1. Pre-calculate data to avoid re-computing per iteration
    # Store: (geometry, start_point_2d, bounding_box)
    contour_data = []

    for c in contours:
        if c.is_empty():
            contour_data.append(None)
            continue

        segments = c.segments()
        if not segments:
            contour_data.append(None)
            continue

        # Get vertices for point-in-poly check
        verts_3d = segments[0]
        verts_2d = [p[:2] for p in verts_3d]

        # Get Bounding Box (min_x, min_y, max_x, max_y)
        rect = c.rect()

        # We only need one test point to determine nesting
        test_point = verts_2d[0]

        contour_data.append(
            {
                "geo": c,
                "verts": verts_2d,
                "rect": rect,
                "test_point": test_point,
            }
        )

    normalized_contours: List[Geometry] = []

    for i in range(count):
        current = contour_data[i]
        if current is None:
            continue

        nesting_level = 0
        tx, ty = current["test_point"]

        # Optimization: Filter candidates by Bounding Box first
        # We check if 'current' is inside 'other'
        for j in range(count):
            if i == j:
                continue

            other = contour_data[j]
            if other is None:
                continue

            # Bounding Box Check:
            # If current.x is outside other.bbox, it strictly cannot be
            # inside other.
            o_min_x, o_min_y, o_max_x, o_max_y = other["rect"]

            if tx < o_min_x or tx > o_max_x or ty < o_min_y or ty > o_max_y:
                continue

            # Detailed Check:
            # Use the raw point-in-polygon test
            if is_point_in_polygon(current["test_point"], other["verts"]):
                nesting_level += 1

        signed_area = get_subpath_area(current["geo"].commands, 0)
        is_ccw = signed_area > 0
        is_nested_odd = nesting_level % 2 != 0

        # An outer shape (even nesting) should be CCW.
        # A hole (odd nesting) should be CW.
        # If the current state is wrong, reverse the contour.
        if (is_nested_odd and is_ccw) or (not is_nested_odd and not is_ccw):
            normalized_contours.append(reverse_contour(current["geo"]))
        else:
            normalized_contours.append(current["geo"])

    return normalized_contours


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


def remove_inner_edges(geometry: Geometry) -> Geometry:
    """
    Filters a geometry, keeping all open paths and only the external-most
    closed paths (contours).

    This function first splits the input geometry into individual contours.
    It then separates these contours into two groups: open paths and closed
    paths. The closed paths are filtered to remove any inner contours (holes),
    and finally, the remaining external closed paths are recombined with the
    original open paths into a new Geometry object.

    Args:
        geometry: The input Geometry object to filter.

    Returns:
        A new Geometry object containing only the external contours and all
        original open paths.
    """
    from .geometry import Geometry  # For creating the new object

    if geometry.is_empty():
        return Geometry()

    all_contours = split_into_contours(geometry)
    if not all_contours:
        return Geometry()

    closed_contours: List[Geometry] = []
    open_contours: List[Geometry] = []

    for contour in all_contours:
        # Use a reasonably small tolerance for checking if a path is closed.
        if contour.is_closed(tolerance=1e-6):
            closed_contours.append(contour)
        else:
            open_contours.append(contour)

    # Filter the closed contours to get only the external ones
    external_closed_contours = filter_to_external_contours(closed_contours)

    # Reassemble the final geometry
    final_geo = Geometry()
    for contour in external_closed_contours:
        final_geo.commands.extend(contour.commands)
    for contour in open_contours:
        final_geo.commands.extend(contour.commands)

    # Preserve the last_move_to from the original, as it's the most
    # sensible value, although its direct relevance might be diminished.
    final_geo.last_move_to = geometry.last_move_to

    return final_geo
