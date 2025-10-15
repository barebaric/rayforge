import math
import logging
from typing import Tuple, Optional, TYPE_CHECKING, TypeVar
import pyclipper

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
            new_geo.commands.extend(new_contour_geo.commands)

    logger.debug("Grow_geometry finished")
    return new_geo
