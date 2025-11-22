import math
import logging
from typing import Tuple, Optional, TYPE_CHECKING, TypeVar, List
import numpy as np
import pyclipper

from .linearize import linearize_arc

if TYPE_CHECKING:
    from .geometry import Geometry, Command

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


def _transform_commands_non_uniform(
    commands: List["Command"], matrix: "np.ndarray"
) -> List["Command"]:
    """
    Handles transformation when non-uniform scaling is present.
    Arcs must be linearized as they become elliptical.
    """
    # Local import to avoid circular dependency
    from .geometry import LineToCommand, ArcToCommand, MovingCommand

    transformed_commands: List["Command"] = []
    last_point_untransformed: Optional[Tuple[float, float, float]] = None

    for cmd in commands:
        original_cmd_end = cmd.end if isinstance(cmd, MovingCommand) else None

        if isinstance(cmd, ArcToCommand):
            start_point = last_point_untransformed or (0.0, 0.0, 0.0)
            segments = linearize_arc(cmd, start_point)
            for p1, p2 in segments:
                point_vec = np.array([p2[0], p2[1], p2[2], 1.0])
                transformed_vec = matrix @ point_vec
                transformed_commands.append(
                    LineToCommand(tuple(transformed_vec[:3]))
                )
        elif isinstance(cmd, MovingCommand):
            point_vec = np.array([*cmd.end, 1.0])
            transformed_vec = matrix @ point_vec
            cmd.end = tuple(transformed_vec[:3])

            if isinstance(cmd, ArcToCommand):
                # Recalculate offset (vector transform)
                offset_vec_3d = np.array(
                    [cmd.center_offset[0], cmd.center_offset[1], 0]
                )
                rot_scale_matrix = matrix[:3, :3]
                new_offset_vec_3d = rot_scale_matrix @ offset_vec_3d
                cmd.center_offset = (
                    new_offset_vec_3d[0],
                    new_offset_vec_3d[1],
                )
            transformed_commands.append(cmd)
        else:
            transformed_commands.append(cmd)

        if original_cmd_end is not None:
            last_point_untransformed = original_cmd_end

    return transformed_commands


def _transform_commands_uniform(
    commands: List["Command"], matrix: "np.ndarray"
) -> List["Command"]:
    """
    Handles transformation for uniform scaling, rotation, and translation.
    Uses vectorized numpy operations for high performance.
    Updates commands in-place where possible.
    """
    # Local import to avoid circular dependency
    from .geometry import ArcToCommand, MovingCommand

    points: List[Tuple[float, float, float]] = []
    cmd_indices: List[int] = []
    arc_offsets: List[Tuple[float, float, float]] = []
    arc_indices: List[int] = []

    for i, cmd in enumerate(commands):
        if isinstance(cmd, MovingCommand) and cmd.end:
            points.append(cmd.end)
            cmd_indices.append(i)
            if isinstance(cmd, ArcToCommand):
                # 2D offsets to 3D vectors
                arc_offsets.append((*cmd.center_offset, 0.0))
                arc_indices.append(i)

    if points:
        # Batch transform points
        pts_array = np.array(points)
        ones = np.ones((pts_array.shape[0], 1))
        pts_homo = np.hstack([pts_array, ones])
        transformed_pts = pts_homo @ matrix.T
        res_pts = transformed_pts[:, :3].tolist()

        for i, original_idx in enumerate(cmd_indices):
            commands[original_idx].end = tuple(res_pts[i])

        # Batch transform arc offsets (rotation/scale only)
        if arc_offsets:
            vec_array = np.array(arc_offsets)
            rot_scale_matrix = matrix[:3, :3]
            transformed_offsets = vec_array @ rot_scale_matrix.T
            res_offsets = transformed_offsets.tolist()

            for i, original_idx in enumerate(arc_indices):
                off = res_offsets[i]
                cmd_to_update = commands[original_idx]
                if isinstance(cmd_to_update, ArcToCommand):
                    cmd_to_update.center_offset = (off[0], off[1])

    return commands


def apply_affine_transform(
    commands: List["Command"], matrix: "np.ndarray"
) -> List["Command"]:
    """
    Applies an affine transformation matrix to a list of commands.
    Automatically selects between a fast vectorized path for uniform transforms
    and a linearization path for non-uniform scaling.

    Args:
        commands: The list of commands to transform.
        matrix: A 4x4 numpy affine transformation matrix.

    Returns:
        The list of transformed commands (may be a new list or modified
        original).
    """
    if not commands:
        return commands

    v_x = matrix @ np.array([1, 0, 0, 0])
    v_y = matrix @ np.array([0, 1, 0, 0])
    len_x = np.linalg.norm(v_x[:2])
    len_y = np.linalg.norm(v_y[:2])
    is_non_uniform = not np.isclose(len_x, len_y)

    if is_non_uniform:
        return _transform_commands_non_uniform(commands, matrix)
    else:
        return _transform_commands_uniform(commands, matrix)
