import math
import logging
from typing import List, Tuple, Optional, TYPE_CHECKING, TypeVar
import numpy as np
from .analysis import get_subpath_area

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
    in the output.

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

    for i, data in enumerate(contour_data):
        logger.debug(f"Processing contour #{i}")
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

        contour_geo = data["geo"]
        original_signed_area = get_subpath_area(contour_geo.commands, 0)
        logger.debug(f"Original signed area: {original_signed_area}")

        new_vertices: List[Tuple[float, float]] = []
        for j in range(len(vertices)):
            p_prev = vertices[(j - 1 + len(vertices)) % len(vertices)]
            p_curr = vertices[j]
            p_next = vertices[(j + 1) % len(vertices)]

            v_in = (p_curr[0] - p_prev[0], p_curr[1] - p_prev[1])
            v_out = (p_next[0] - p_curr[0], p_next[1] - p_curr[1])

            mag_in = math.hypot(*v_in)
            mag_out = math.hypot(*v_out)

            if mag_in < 1e-9 or mag_out < 1e-9:
                continue

            v_in_norm = (v_in[0] / mag_in, v_in[1] / mag_in)
            v_out_norm = (v_out[0] / mag_out, v_out[1] / mag_out)

            # Normal vector points 90 degrees left of the segment's direction
            n_in = (v_in_norm[1], -v_in_norm[0])
            n_out = (v_out_norm[1], -v_out_norm[0])

            # A point on each offset line
            p1_offset = (
                p_curr[0] + offset * n_in[0],
                p_curr[1] + offset * n_in[1],
            )
            p2_offset = (
                p_curr[0] + offset * n_out[0],
                p_curr[1] + offset * n_out[1],
            )

            a1, b1 = v_in_norm[1], -v_in_norm[0]
            c1 = a1 * p1_offset[0] + b1 * p1_offset[1]

            a2, b2 = v_out_norm[1], -v_out_norm[0]
            c2 = a2 * p2_offset[0] + b2 * p2_offset[1]

            intersection = _solve_2x2_system(a1, b1, c1, a2, b2, c2)

            if intersection:
                new_vertices.append(intersection)
            else:
                new_vertices.append(
                    (
                        p_curr[0] + offset * n_out[0],
                        p_curr[1] + offset * n_out[1],
                    )
                )

        logger.debug(
            f"Generated {len(new_vertices)} new vertices: {new_vertices}"
        )

        # When shrinking, a large offset can cause the polygon to invert.
        # This is detected by checking if new vertices "cross over" the
        # centroid.
        if offset < 0 and len(new_vertices) == len(vertices):
            centroid = np.mean(vertices, axis=0)
            v_old = np.array(vertices[0])
            v_new = np.array(new_vertices[0])
            vec_old = v_old - centroid
            vec_new = v_new - centroid
            if np.dot(vec_old, vec_new) < 0:
                # Shrinking caused polygon inversion. Discarding contour.
                continue

        if len(new_vertices) < 3:
            # Not enough new vertices to form a polygon, skipping.
            continue

        new_contour_geo = type(geometry).from_points(
            [(v[0], v[1], 0.0) for v in new_vertices], close=True
        )

        if new_contour_geo.is_empty():
            # Generated contour geometry is empty, skipping.
            continue

        new_signed_area = get_subpath_area(new_contour_geo.commands, 0)

        if abs(new_signed_area) < 1e-9:
            # New area is degenerate, discarding.
            continue

        original_sign = math.copysign(1, original_signed_area)
        new_sign = math.copysign(1, new_signed_area)

        if new_sign != original_sign:
            # Winding order flipped. Discarding contour.
            continue

        new_geo.commands.extend(new_contour_geo.commands)

    logger.debug("Grow_geometry finished")
    return new_geo
