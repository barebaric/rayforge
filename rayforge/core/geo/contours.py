from __future__ import annotations
import logging
from typing import List, Tuple, TYPE_CHECKING
import numpy as np
from .analysis import get_subpath_area_from_array
from .primitives import is_point_in_polygon
from .split import split_into_contours
from .constants import (
    CMD_TYPE_MOVE,
    CMD_TYPE_LINE,
    CMD_TYPE_ARC,
    COL_TYPE,
    COL_X,
    COL_Y,
    COL_Z,
    COL_I,
    COL_J,
    COL_CW,
)


if TYPE_CHECKING:
    from .geometry import Geometry

logger = logging.getLogger(__name__)


def close_geometry_gaps_from_array(
    data: np.ndarray, tolerance: float = 1e-6
) -> np.ndarray:
    """
    Closes small gaps in a geometry array to form clean, connected paths.

    Args:
        data: The input geometry numpy array.
        tolerance: The maximum distance between two points to be
                    considered "the same".

    Returns:
        A new, modified numpy array.
    """
    if data is None or len(data) < 2:
        return data if data is not None else np.array([])

    # Pass 1: Close gaps within each contour (intra-contour)
    # This pass modifies a copy of the array.
    modified_data = data.copy()
    move_indices = np.where(modified_data[:, COL_TYPE] == CMD_TYPE_MOVE)[0]
    sub_arrays = np.split(modified_data, move_indices[1:])

    for sub in sub_arrays:
        if len(sub) < 2:
            continue
        start_pt = sub[0, COL_X : COL_Z + 1]
        end_pt = sub[-1, COL_X : COL_Z + 1]
        dist_sq = np.sum((start_pt - end_pt) ** 2)
        if dist_sq < tolerance * tolerance:
            # Snap the end point to the start point.
            sub[-1, COL_X : COL_Z + 1] = start_pt
    # Reassemble the array after modifications
    modified_data = np.vstack(sub_arrays)

    # Pass 2: Connect adjacent contours (inter-contour)
    # This pass builds a new list of rows, as it can change command types.
    final_rows: List[np.ndarray] = []
    last_end_point: np.ndarray | None = None
    for row in modified_data:
        cmd_type = row[COL_TYPE]
        end_point = row[COL_X : COL_Z + 1]

        if cmd_type == CMD_TYPE_MOVE:
            if last_end_point is not None:
                dist_sq = np.sum((end_point - last_end_point) ** 2)
                if dist_sq < tolerance * tolerance:
                    # This MoveTo is a small jump; replace with a LineTo
                    # to the exact previous endpoint to close the gap.
                    new_row = row.copy()
                    new_row[COL_TYPE] = CMD_TYPE_LINE
                    new_row[COL_X : COL_Z + 1] = last_end_point
                    final_rows.append(new_row)
                    # The logical position remains last_end_point
                else:
                    final_rows.append(row)
                    last_end_point = end_point
            else:
                final_rows.append(row)
                last_end_point = end_point
        else:
            final_rows.append(row)
            last_end_point = end_point

    if not final_rows:
        return np.array([])
    return np.array(final_rows)


def close_geometry_gaps(
    geometry: Geometry, tolerance: float = 1e-6
) -> Geometry:
    """
    Closes small gaps in a Geometry object to form clean, connected paths.

    This function creates a new Geometry object with the modifications.

    Args:
        geometry: The input Geometry object.
        tolerance: The maximum distance between two points to be
                    considered "the same".

    Returns:
        A new, modified Geometry object.
    """
    new_geo = geometry.copy()
    new_geo._sync_to_numpy()
    if new_geo.is_empty() or new_geo._data is None:
        return new_geo

    new_geo._data = close_geometry_gaps_from_array(
        new_geo._data, tolerance=tolerance
    )
    return new_geo


def reverse_contour(contour: Geometry) -> Geometry:
    """Reverses the direction of a single-contour Geometry object."""
    from .geometry import Geometry

    contour._sync_to_numpy()
    data = contour._data
    if data is None or len(data) == 0:
        return contour.copy()

    if data[0, COL_TYPE] != CMD_TYPE_MOVE:
        return contour.copy()  # Can only reverse single contours

    new_rows = []

    # New path starts at the old path's end
    last_row = data[-1]
    new_rows.append(
        [
            CMD_TYPE_MOVE,
            last_row[COL_X],
            last_row[COL_Y],
            last_row[COL_Z],
            0.0,
            0.0,
            0.0,
        ]
    )
    last_point = last_row[COL_X : COL_Z + 1]

    # Iterate backwards through rows
    for i in range(len(data) - 1, 0, -1):
        end_row = data[i]
        start_row = data[i - 1]
        start_point = start_row[COL_X : COL_Z + 1]
        cmd_type = end_row[COL_TYPE]

        if cmd_type == CMD_TYPE_LINE:
            new_rows.append(
                [
                    CMD_TYPE_LINE,
                    start_point[0],
                    start_point[1],
                    start_point[2],
                    0,
                    0,
                    0,
                ]
            )
        elif cmd_type == CMD_TYPE_ARC:
            center_abs_x = start_point[0] + end_row[COL_I]
            center_abs_y = start_point[1] + end_row[COL_J]
            new_offset_x = center_abs_x - last_point[0]
            new_offset_y = center_abs_y - last_point[1]
            new_cw = 1.0 - end_row[COL_CW]  # Flip clockwise flag
            new_rows.append(
                [
                    CMD_TYPE_ARC,
                    start_point[0],
                    start_point[1],
                    start_point[2],
                    new_offset_x,
                    new_offset_y,
                    new_cw,
                ]
            )

        last_point = start_point

    new_geo = Geometry()
    new_geo._data = np.array(new_rows)
    new_geo.last_move_to = (
        new_rows[0][COL_X],
        new_rows[0][COL_Y],
        new_rows[0][COL_Z],
    )
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
        c._sync_to_numpy()  # Ensure data is available
        if c.data is None:
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

        current_data = current["geo"].data
        if current_data is None:
            continue
        signed_area = get_subpath_area_from_array(current_data, 0)
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
    final_contours = []
    for c in normalized_contours:
        c._sync_to_numpy()
        data = c.data
        if data is not None and get_subpath_area_from_array(data, 0) > 1e-9:
            final_contours.append(c)
    return final_contours


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
        final_geo.extend(contour)
    for contour in open_contours:
        final_geo.extend(contour)

    # Preserve the last_move_to from the original, as it's the most
    # sensible value, although its direct relevance might be diminished.
    final_geo.last_move_to = geometry.last_move_to

    return final_geo
