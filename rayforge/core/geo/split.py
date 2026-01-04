from __future__ import annotations
import logging
from typing import List, TYPE_CHECKING, Set
import numpy as np
from .constants import CMD_TYPE_MOVE, COL_TYPE

if TYPE_CHECKING:
    from .geometry import Geometry

logger = logging.getLogger(__name__)


def split_into_contours(geometry: "Geometry") -> List["Geometry"]:
    """
    Splits a Geometry object into a list of separate, single-contour
    Geometry objects. Each new object represents one continuous subpath
    that starts with a MoveToCommand.
    """
    geometry._sync_to_numpy()
    if geometry.is_empty() or geometry._data is None:
        return []

    from .geometry import Geometry

    contours: List[Geometry] = []
    data = geometry._data

    move_indices = np.where(data[:, COL_TYPE] == CMD_TYPE_MOVE)[0]

    # If there are no MoveTo commands, or if the first command is not a MoveTo,
    # the entire array is treated as a single contour.
    if len(move_indices) == 0:
        if len(data) > 0:
            new_geo = Geometry()
            new_geo._data = data.copy()
            contours.append(new_geo)
    else:
        # Handle the first segment if it doesn't start with a move
        if move_indices[0] != 0:
            new_geo = Geometry()
            new_geo._data = data[: move_indices[0]].copy()
            contours.append(new_geo)

        # Split the array at each MoveTo index (excluding the first one)
        # This creates sub-arrays, each starting with a MoveTo.
        split_arrays = np.split(data, move_indices[1:])
        for arr in split_arrays:
            if len(arr) > 0:
                new_geo = Geometry()
                new_geo._data = arr.copy()
                contours.append(new_geo)

    # Filter out any empty geometries that might have been created
    return [c for c in contours if not c.is_empty()]


def _find_connected_components_bfs(
    num_contours: int, adj: List[List[int]]
) -> List[List[int]]:
    """Finds connected components in the graph using BFS."""
    visited: Set[int] = set()
    components: List[List[int]] = []
    for i in range(num_contours):
        if i not in visited:
            component = []
            q = [i]
            visited.add(i)
            while q:
                u = q.pop(0)
                component.append(u)
                for v in adj[u]:
                    if v not in visited:
                        visited.add(v)
                        q.append(v)
            components.append(component)
    return components


def split_into_components(geometry: "Geometry") -> List["Geometry"]:
    """
    Analyzes the geometry and splits it into a list of separate,
    logically connected shapes (components).
    """
    from .geometry import Geometry
    from .primitives import is_point_in_polygon

    logger.debug("Starting to split_into_components")
    if geometry.is_empty():
        logger.debug("Geometry is empty, returning empty list.")
        return []

    contour_geometries = split_into_contours(geometry)
    if len(contour_geometries) <= 1:
        logger.debug("<= 1 contour, returning a copy of the whole.")
        return [geometry.copy()]

    all_contour_data = geometry._get_valid_contours_data(contour_geometries)
    if not all_contour_data:
        logger.debug("No valid contours found after filtering.")
        return []

    if not any(c["is_closed"] for c in all_contour_data):
        logger.debug("No closed paths found. Returning single component.")
        return [geometry.copy()]

    num_contours = len(all_contour_data)
    adj: List[List[int]] = [[] for _ in range(num_contours)]
    for i in range(num_contours):
        if not all_contour_data[i]["is_closed"]:
            continue
        for j in range(num_contours):
            if i == j:
                continue
            data_i = all_contour_data[i]
            data_j = all_contour_data[j]
            if is_point_in_polygon(data_j["vertices"][0], data_i["vertices"]):
                adj[i].append(j)
                adj[j].append(i)

    component_indices_list = _find_connected_components_bfs(num_contours, adj)
    logger.debug(f"Found {len(component_indices_list)} raw components.")

    final_geometries: List[Geometry] = []
    stray_open_geo = Geometry()
    for i, indices in enumerate(component_indices_list):
        component_geo = Geometry()
        has_closed_path = False
        for idx in indices:
            contour = all_contour_data[idx]
            component_geo.extend(contour["geo"])
            if contour["is_closed"]:
                has_closed_path = True

        if has_closed_path:
            final_geometries.append(component_geo)
        else:
            stray_open_geo.extend(component_geo)

    if not stray_open_geo.is_empty():
        logger.debug(
            "Found stray open paths, creating a final component for them."
        )
        final_geometries.append(stray_open_geo)

    return final_geometries
