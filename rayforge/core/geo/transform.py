import math
import logging
from typing import Tuple, Optional, TYPE_CHECKING, TypeVar
import numpy as np
import pyclipper
from .constants import (
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
from .linearize import linearize_arc

if TYPE_CHECKING:
    from .geometry import Geometry

# Define a TypeVar to make the function generic over Geometry and its
# subclasses.
T_Geometry = TypeVar("T_Geometry", bound="Geometry")
logger = logging.getLogger(__name__)


def _solve_2x2_system(
    a1: float, b1: float, c1: float, a2: float, b2: float, c2: float
) -> Optional[Tuple[float, float]]:
    """
    Solves a 2x2 system of linear equations:
    a1*x + b1*y = c1
    a2*x + b2*y = c2
    """
    det = a1 * b2 - a2 * b1
    if abs(det) < 1e-9:
        return None  # No unique solution (lines are parallel)
    x = (c1 * b2 - c2 * b1) / det
    y = (a1 * c2 - a2 * c1) / det
    return x, y


def grow_geometry(geometry: T_Geometry, offset: float) -> T_Geometry:
    """
    Offsets the closed contours of a Geometry object by a given amount.

    This function grows (positive offset) or shrinks (negative offset) the
    area enclosed by closed paths. Arcs are linearized into polylines for the
    offsetting process. Open paths are currently ignored and not included
    in the output. This implementation uses the pyclipper library to handle
    complex cases, including self-intersections.

    Args:
        geometry: The input Geometry object.
        offset: The distance to offset the geometry. Positive values expand
                the shape, negative values contract it.

    Returns:
        A new Geometry object of the same type as the input, containing
        the offset shape(s).
    """
    new_geo = type(geometry)()
    contour_geometries = geometry.split_into_contours()
    contour_data = geometry._get_valid_contours_data(contour_geometries)

    logger.debug(f"Running grow_geometry with offset: {offset}")

    # Pyclipper works with integers, so we need to scale our coordinates.
    CLIPPER_SCALE = 1e7
    pco = pyclipper.PyclipperOffset()  # type: ignore

    paths_to_offset = []
    for i, data in enumerate(contour_data):
        logger.debug(f"Processing contour #{i} for pyclipper")
        if not data["is_closed"]:
            logger.debug("Contour is not closed, skipping.")
            continue

        vertices = data["vertices"]

        # If the last vertex is a duplicate of the first for closed paths,
        # remove it.
        if (
            len(vertices) > 1
            and math.isclose(vertices[0][0], vertices[-1][0])
            and math.isclose(vertices[0][1], vertices[-1][1])
        ):
            vertices.pop()

        if len(vertices) < 3:
            logger.debug("Contour has < 3 vertices, skipping.")
            continue

        scaled_vertices = [
            (int(v[0] * CLIPPER_SCALE), int(v[1] * CLIPPER_SCALE))
            for v in vertices
        ]
        paths_to_offset.append(scaled_vertices)

    pco.AddPaths(
        paths_to_offset,
        pyclipper.JT_MITER,  # type: ignore
        pyclipper.ET_CLOSEDPOLYGON,  # type: ignore
    )
    solution = pco.Execute(offset * CLIPPER_SCALE)

    logger.debug(f"Pyclipper generated {len(solution)} offset contours.")

    for new_contour_scaled in solution:
        if len(new_contour_scaled) < 3:
            continue

        new_vertices = [
            (p[0] / CLIPPER_SCALE, p[1] / CLIPPER_SCALE)
            for p in new_contour_scaled
        ]

        new_contour_geo = type(geometry).from_points(
            [(v[0], v[1], 0.0) for v in new_vertices], close=True
        )

        if not new_contour_geo.is_empty():
            new_geo.extend(new_contour_geo)

    logger.debug("Grow_geometry finished")
    return new_geo


class _MockArcCmd:
    """Helper to adapt array data for linearize_arc."""

    __slots__ = ("end", "center_offset", "clockwise")

    def __init__(self, end, center_offset, clockwise):
        self.end = end
        self.center_offset = center_offset
        self.clockwise = clockwise


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
    v_x = matrix @ np.array([1, 0, 0, 0])
    v_y = matrix @ np.array([0, 1, 0, 0])
    len_x = np.linalg.norm(v_x[:2])
    len_y = np.linalg.norm(v_y[:2])
    is_non_uniform = not np.isclose(len_x, len_y)

    if is_non_uniform:
        return _transform_array_non_uniform(data, matrix)
    else:
        return _transform_array_uniform(data, matrix)


def _transform_array_uniform(
    data: np.ndarray, matrix: np.ndarray
) -> np.ndarray:
    # XYZ transform
    # data is (N, 7). Columns 1,2,3 are X,Y,Z.
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
                    ]
                )
        else:
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

    return np.array(new_rows)
