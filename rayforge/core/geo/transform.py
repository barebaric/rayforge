import math
import logging
from typing import Tuple, Optional, TYPE_CHECKING, TypeVar, List, Dict
import numpy as np
import pyclipper
from .constants import (
    CMD_TYPE_LINE,
    CMD_TYPE_ARC,
    CMD_TYPE_BEZIER,
    COL_TYPE,
    COL_X,
    COL_Y,
    COL_Z,
    COL_I,
    COL_J,
    COL_CW,
    COL_C1X,
    COL_C1Y,
    COL_C2X,
    COL_C2Y,
)
from .linearize import linearize_arc

if TYPE_CHECKING:
    from .geometry import Geometry

# Define a TypeVar to make the function generic over Geometry and its
# subclasses.
T_Geometry = TypeVar("T_Geometry", bound="Geometry")
logger = logging.getLogger(__name__)


CLIPPER_SCALE = int(1e7)


def _prepare_contour_items(
    contour_data: List[Dict], scale: int = CLIPPER_SCALE
) -> List[Dict]:
    """
    Prepares contour data items for hierarchy analysis.

    Args:
        contour_data: List of contour data dicts from get_valid_contours_data.
        scale: Scale factor for clipper integer coordinates.

    Returns:
        List of dicts with 'geo', 'verts', 'path', 'rect', 'area', 'id'.
    """
    items = []

    for data in contour_data:
        if not data["is_closed"]:
            continue

        data["geo"]._sync_to_numpy()

        verts = list(data["vertices"])
        if (
            len(verts) > 1
            and math.isclose(verts[0][0], verts[-1][0])
            and math.isclose(verts[0][1], verts[-1][1])
        ):
            verts.pop()

        if len(verts) < 3:
            continue

        area = 0.0
        for i in range(len(verts)):
            j = (i + 1) % len(verts)
            area += verts[i][0] * verts[j][1]
            area -= verts[j][0] * verts[i][1]
        area = abs(area) / 2.0

        scaled_path = [(int(v[0] * scale), int(v[1] * scale)) for v in verts]

        items.append(
            {
                "geo": data["geo"],
                "verts": verts,
                "path": scaled_path,
                "rect": data["geo"].rect(),
                "area": area,
                "id": len(items),
            }
        )

    return items


def _build_containment_hierarchy(
    items: List[Dict],
    check_intersection_fn,
    is_point_in_polygon_fn,
) -> List[int]:
    """
    Builds a parent map representing containment hierarchy.

    Each contour maps to its immediate parent (smallest enclosing container).

    Args:
        items: List of contour item dicts.
        check_intersection_fn: Function to check if two geometries intersect.
        is_point_in_polygon_fn: Function to check if point is in polygon.

    Returns:
        List where parent_map[i] = index of immediate parent, or -1 if none.
    """
    parent_map = [-1] * len(items)

    for i, item_i in enumerate(items):
        best_parent = -1
        best_parent_area = float("inf")

        for j, item_j in enumerate(items):
            if i == j:
                continue

            r_i = item_i["rect"]
            r_j = item_j["rect"]
            if not (
                r_j[0] <= r_i[0]
                and r_j[1] <= r_i[1]
                and r_j[2] >= r_i[2]
                and r_j[3] >= r_i[3]
            ):
                continue

            if item_j["area"] <= item_i["area"]:
                continue

            if check_intersection_fn(item_i["geo"].data, item_j["geo"].data):
                continue

            if is_point_in_polygon_fn(item_i["verts"][0], item_j["verts"]):
                if item_j["area"] < best_parent_area:
                    best_parent_area = item_j["area"]
                    best_parent = j

        parent_map[i] = best_parent

    return parent_map


def _calculate_nesting_depths(
    parent_map: List[int], num_items: int
) -> List[int]:
    """
    Calculates nesting depth for each item based on parent hierarchy.

    Args:
        parent_map: List mapping each item to its parent index (-1 if none).
        num_items: Total number of items.

    Returns:
        List of depths (0 = top-level, 1 = one level deep, etc.).
    """
    depths = [0] * num_items
    for i in range(num_items):
        d = 0
        curr = parent_map[i]
        while curr != -1:
            d += 1
            curr = parent_map[curr]
            if d > num_items:
                break
        depths[i] = d
    return depths


def _group_solids_and_holes(
    depths: List[int], parent_map: List[int]
) -> Dict[int, List[int]]:
    """
    Groups holes with their parent solids.

    Even depth = Solid (0, 2, ...)
    Odd depth = Hole (1, 3, ...)

    Args:
        depths: Nesting depth for each item.
        parent_map: Parent index for each item.

    Returns:
        Dict mapping solid index -> list of hole indices.
    """
    groups: Dict[int, List[int]] = {}

    for i, d in enumerate(depths):
        if d % 2 == 0:
            if i not in groups:
                groups[i] = []
        else:
            p = parent_map[i]
            if p != -1:
                if p not in groups:
                    groups[p] = []
                groups[p].append(i)

    return groups


def _offset_contour_group(
    solid_path: List[Tuple[int, int]],
    hole_paths: List[List[Tuple[int, int]]],
    offset: float,
    scale: int = CLIPPER_SCALE,
) -> List[List[Tuple[float, float]]]:
    """
    Offsets a solid with its holes using pyclipper.

    Args:
        solid_path: Scaled integer coordinates for solid (will be CCW).
        hole_paths: List of scaled integer paths for holes (will be CW).
        offset: Offset distance (positive = grow, negative = shrink).
        scale: Scale factor used for integer conversion.

    Returns:
        List of offset contours as float coordinate tuples.
    """
    if not pyclipper.Orientation(solid_path):  # type: ignore
        solid_path = list(reversed(solid_path))

    paths_to_offset = [solid_path]

    for hole_path in hole_paths:
        if pyclipper.Orientation(hole_path):  # type: ignore
            hole_path = list(reversed(hole_path))
        paths_to_offset.append(hole_path)

    try:
        pco = pyclipper.PyclipperOffset()  # type: ignore
        pco.AddPaths(
            paths_to_offset,
            pyclipper.JT_MITER,  # type: ignore
            pyclipper.ET_CLOSEDPOLYGON,  # type: ignore
        )
        solution = pco.Execute(offset * scale)
    except Exception as e:
        logger.error(f"Offset failed: {e}")
        return []

    result = []
    for contour in solution:
        if len(contour) < 3:
            continue
        result.append([(p[0] / scale, p[1] / scale) for p in contour])

    return result


def grow_geometry(geometry: T_Geometry, offset: float) -> T_Geometry:
    """
    Offsets the closed contours of a Geometry object by a given amount.

    This function grows (positive offset) or shrinks (negative offset) the
    area enclosed by closed paths.

    This implementation processes logically distinct shapes (islands)
    independently. Holes are associated with their enclosing solids and
    offset together. Adjacent or overlapping solids remain separate, preserving
    distinct toolpaths.

    Args:
        geometry: The input Geometry object.
        offset: The distance to offset the geometry. Positive values expand
                the shape, negative values contract it.

    Returns:
        A new Geometry object of the same type as the input, containing
        the offset shape(s).
    """
    from .contours import get_valid_contours_data
    from .intersect import check_intersection_from_array
    from .primitives import is_point_in_polygon
    from .split import split_into_contours

    new_geo = type(geometry)()

    raw_contours = split_into_contours(geometry)
    if not raw_contours:
        return new_geo

    contour_data = get_valid_contours_data(raw_contours)
    if not contour_data:
        return new_geo

    logger.debug(f"Running grow_geometry with offset: {offset}")

    closed_items = _prepare_contour_items(contour_data)
    if not closed_items:
        return new_geo

    parent_map = _build_containment_hierarchy(
        closed_items, check_intersection_from_array, is_point_in_polygon
    )

    depths = _calculate_nesting_depths(parent_map, len(closed_items))

    solid_groups = _group_solids_and_holes(depths, parent_map)

    for solid_idx, hole_indices in solid_groups.items():
        solid_item = closed_items[solid_idx]
        hole_paths = [closed_items[h_idx]["path"] for h_idx in hole_indices]

        offset_contours = _offset_contour_group(
            solid_item["path"], hole_paths, offset
        )

        for new_vertices in offset_contours:
            new_contour_geo = type(geometry).from_points(
                [(v[0], v[1], 0.0) for v in new_vertices], close=True
            )
            if not new_contour_geo.is_empty():
                new_geo.extend(new_contour_geo)

    logger.debug("Grow_geometry finished")
    return new_geo


def map_geometry_to_frame(
    geometry: T_Geometry,
    origin: Tuple[float, float],
    p_width: Tuple[float, float],
    p_height: Tuple[float, float],
    anchor_y: Optional[float] = None,
    stable_src_height: Optional[float] = None,
) -> T_Geometry:
    """
    Transforms a Geometry object to fit into an affine frame defined by three
    points.

    This function scales, rotates, and translates the input geometry from its
    natural bounding box to fit perfectly within the parallelogram defined by
    the origin, width point, and height point.

    Args:
        geometry: The Geometry object to transform. A copy is made.
        origin: The (x, y) coordinate for the bottom-left corner of the
                target frame.
        p_width: The (x, y) coordinate for the bottom-right corner of the
                 target frame, defining the local X-axis.
        p_height: The (x, y) coordinate for the top-left corner of the
                  target frame, defining the local Y-axis.
        anchor_y: Optional y-coordinate to use as vertical anchor instead of
                  the bounding box minimum. Useful for text where the baseline
                  should remain fixed. If None, uses min_y from bounding box.
        stable_src_height: Optional stable source height to use for scaling
                          instead of the bounding box height. Useful for text
                          where the height should remain stable regardless of
                          descenders. If None, uses max_y - min_y from
                          bounding box.

    Returns:
        A new, transformed Geometry object.
    """
    if geometry.is_empty():
        return type(geometry)()

    # 1. Get the source geometry's bounding box
    min_x, min_y, max_x, max_y = geometry.rect()
    src_width = max_x - min_x
    src_height = (
        stable_src_height if stable_src_height is not None else (max_y - min_y)
    )

    # Use anchor_y if provided, otherwise use bounding box min_y
    anchor_y_value = anchor_y if anchor_y is not None else min_y

    # Handle degenerate source geometry
    if src_width < 1e-9 or src_height < 1e-9:
        return type(geometry)()

    # 2. Calculate target frame vectors
    u_vec = (p_width[0] - origin[0], p_width[1] - origin[1])
    v_vec = (p_height[0] - origin[0], p_height[1] - origin[1])

    # 3. Build the transformation matrix
    # This matrix maps the source bounding box [min_x, min_y] -> [max_x, max_y]
    # to the target parallelogram.
    # We compose matrices: M = T_frame @ T_scale @ T_translate

    # T_translate: moves source rect's origin to (0,0)
    t1 = np.array(
        [
            [1, 0, 0, -min_x],
            [0, 1, 0, -anchor_y_value],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ]
    )

    # T_scale: scales source rect (now at origin) to a 1x1 unit square
    t2 = np.array(
        [
            [1 / src_width, 0, 0, 0],
            [0, 1 / src_height, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ]
    )

    # T_frame: maps the 1x1 unit square to the target parallelogram
    t3 = np.array(
        [
            [u_vec[0], v_vec[0], 0, origin[0]],
            [u_vec[1], v_vec[1], 0, origin[1]],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ]
    )

    # Combine transformations: M applies T1, then T2, then T3.
    final_matrix = t3 @ t2 @ t1

    # 4. Apply the transformation
    # We work on a copy to avoid modifying the original geometry
    transformed_geo = geometry.copy()
    return transformed_geo.transform(final_matrix)


class _MockArcCmd:
    """Helper to adapt array data for linearize_arc."""

    __slots__ = ("end", "center_offset", "clockwise")

    def __init__(self, end, center_offset, clockwise):
        self.end = end
        self.center_offset = center_offset
        self.clockwise = clockwise


class _MockBezierCmd:
    """Helper to adapt array data for linearization."""

    __slots__ = ("end", "c1", "c2")

    def __init__(self, end, c1, c2):
        self.end = end
        self.c1 = c1
        self.c2 = c2


def apply_affine_transform_to_array(
    data: np.ndarray, matrix: np.ndarray
) -> np.ndarray:
    """
    Applies an affine transformation to the geometry array.
    Handles uniform and non-uniform scaling (linearizing arcs for the latter).
    """
    if data is None or data.shape[0] == 0:
        return data

    # Check for non-uniform scaling
    # Compare squared lengths to correctly handle uniform reflections
    # (e.g., scale(1, -1) is uniform, just a flip)
    v_x = matrix @ np.array([1, 0, 0, 0])
    v_y = matrix @ np.array([0, 1, 0, 0])
    len_x_sq = np.sum(v_x[:2] ** 2)
    len_y_sq = np.sum(v_y[:2] ** 2)
    is_non_uniform = not np.isclose(len_x_sq, len_y_sq)

    if is_non_uniform:
        logger.debug(
            "Non-uniform scaling detected (x_scale=%f, y_scale=%f).",
            np.sqrt(len_x_sq),
            np.sqrt(len_y_sq),
        )
        return _transform_array_non_uniform(data, matrix)
    else:
        return _transform_array_uniform(data, matrix)


def _transform_array_uniform(
    data: np.ndarray, matrix: np.ndarray
) -> np.ndarray:
    # XYZ transform
    # data is (N, 8). Columns 1,2,3 are X,Y,Z.
    points = data[:, COL_X : COL_Z + 1]
    ones = np.ones((points.shape[0], 1))
    pts_homo = np.hstack([points, ones])

    transformed_pts = pts_homo @ matrix.T
    data[:, COL_X : COL_Z + 1] = transformed_pts[:, :3]

    # Arc IJ transform (Rotation/Scale only)
    is_arc = data[:, COL_TYPE] == CMD_TYPE_ARC
    if np.any(is_arc):
        vecs = data[is_arc, COL_I : COL_J + 1]
        # Add Z=0 for 3D rotation, though offsets are usually 2D.
        vecs_3d = np.hstack([vecs, np.zeros((vecs.shape[0], 1))])

        rot_scale_matrix = matrix[:3, :3]
        transformed_vecs = vecs_3d @ rot_scale_matrix.T

        data[is_arc, COL_I : COL_J + 1] = transformed_vecs[:, :2]

        # Check determinant for flip
        # Calculate 2D determinant of top-left 2x2
        det = matrix[0, 0] * matrix[1, 1] - matrix[0, 1] * matrix[1, 0]
        if det < 0:
            # Flip clockwise flag
            data[is_arc, COL_CW] = np.where(
                data[is_arc, COL_CW] > 0.5, 0.0, 1.0
            )

    # Bezier control point transform (Full transform)
    is_bezier = data[:, COL_TYPE] == CMD_TYPE_BEZIER
    if np.any(is_bezier):
        # Transform C1
        c1_pts = data[is_bezier, COL_C1X : COL_C1Y + 1]
        c1_homo = np.hstack(
            [
                c1_pts,
                np.zeros((c1_pts.shape[0], 1)),
                np.ones((c1_pts.shape[0], 1)),
            ]
        )
        trans_c1 = c1_homo @ matrix.T
        data[is_bezier, COL_C1X : COL_C1Y + 1] = trans_c1[:, :2]

        # Transform C2
        c2_pts = data[is_bezier, COL_C2X : COL_C2Y + 1]
        c2_homo = np.hstack(
            [
                c2_pts,
                np.zeros((c2_pts.shape[0], 1)),
                np.ones((c2_pts.shape[0], 1)),
            ]
        )
        trans_c2 = c2_homo @ matrix.T
        data[is_bezier, COL_C2X : COL_C2Y + 1] = trans_c2[:, :2]

    return data


def _transform_array_non_uniform(
    data: np.ndarray, matrix: np.ndarray
) -> np.ndarray:
    new_rows = []
    last_point = (0.0, 0.0, 0.0)

    for row in data:
        cmd_type = row[COL_TYPE]
        original_end = (row[COL_X], row[COL_Y], row[COL_Z])

        if cmd_type == CMD_TYPE_ARC:
            start_pt = last_point

            mock_cmd = _MockArcCmd(
                end=original_end,
                center_offset=(row[COL_I], row[COL_J]),
                clockwise=bool(row[COL_CW]),
            )
            # Arcs must be linearized for non-uniform scaling
            segments = linearize_arc(mock_cmd, start_pt)
            for _, p2 in segments:
                p_vec = np.array([p2[0], p2[1], p2[2], 1.0])
                trans_p = matrix @ p_vec

                new_rows.append(
                    [
                        CMD_TYPE_LINE,
                        trans_p[0],
                        trans_p[1],
                        trans_p[2],
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                    ]
                )
        elif cmd_type == CMD_TYPE_BEZIER:
            # Bezier curves are affine-invariant and can be transformed
            # directly without linearization.
            new_row = row.copy()

            # Transform endpoint (X, Y, Z)
            p_vec = np.array(
                [original_end[0], original_end[1], original_end[2], 1.0]
            )
            trans_p = matrix @ p_vec
            new_row[COL_X] = trans_p[0]
            new_row[COL_Y] = trans_p[1]
            new_row[COL_Z] = trans_p[2]

            # Transform C1 (X, Y)
            c1_orig = np.array([row[COL_C1X], row[COL_C1Y], 0.0, 1.0])
            trans_c1 = matrix @ c1_orig
            new_row[COL_C1X] = trans_c1[0]
            new_row[COL_C1Y] = trans_c1[1]

            # Transform C2 (X, Y)
            c2_orig = np.array([row[COL_C2X], row[COL_C2Y], 0.0, 1.0])
            trans_c2 = matrix @ c2_orig
            new_row[COL_C2X] = trans_c2[0]
            new_row[COL_C2Y] = trans_c2[1]

            new_rows.append(new_row)

        else:  # CMD_TYPE_MOVE, CMD_TYPE_LINE, etc.
            # Transform end point
            p_vec = np.array(
                [original_end[0], original_end[1], original_end[2], 1.0]
            )
            trans_p = matrix @ p_vec

            new_row = row.copy()
            new_row[COL_X] = trans_p[0]
            new_row[COL_Y] = trans_p[1]
            new_row[COL_Z] = trans_p[2]
            new_rows.append(new_row)

        last_point = original_end

    if not new_rows:
        return np.array([]).reshape(
            0, data.shape[1]
        )  # Return empty array with correct number of columns
    return np.array(new_rows)
