from __future__ import annotations
import logging
from typing import List, TYPE_CHECKING, Optional, Set

if TYPE_CHECKING:
    from .geometry import Geometry

logger = logging.getLogger(__name__)


def split_into_contours(geometry: "Geometry") -> List["Geometry"]:
    """
    Splits a Geometry object into a list of separate, single-contour
    Geometry objects. Each new object represents one continuous subpath
    that starts with a MoveToCommand.
    """
    if geometry.is_empty():
        return []

    from .geometry import Geometry, MoveToCommand

    contours: List[Geometry] = []
    current_geo: Optional["Geometry"] = None

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
            component_geo.commands.extend(contour["geo"].commands)
            if contour["is_closed"]:
                has_closed_path = True

        if has_closed_path:
            final_geometries.append(component_geo)
        else:
            stray_open_geo.commands.extend(component_geo.commands)

    if not stray_open_geo.is_empty():
        logger.debug(
            "Found stray open paths, creating a final component for them."
        )
        final_geometries.append(stray_open_geo)

    return final_geometries
