"""Path-level operations on :class:`~rayforge.core.geo.geometry.Geometry`
objects and raw command arrays.

Many functions accept a ``data`` argument — a list of 8-element float
lists (the same layout as :attr:`Geometry.data`).
"""

from typing import Optional, Tuple, List, Union, TypeAlias

from numpy.typing import NDArray
import numpy as np

from rayforge.core.geo import Point, Point3D
from rayforge.core.geo.geometry import Geometry

_ArrayLike: TypeAlias = Union[List[List[float]], NDArray[np.float64]]


class PyCommand:
    """Typed view over a single geometry command row.

    Use ``isinstance`` checks to dispatch on the variant::

        if isinstance(cmd, PyCommand.Move):
            ...
        elif isinstance(cmd, PyCommand.Arc):
            ...
    """

    class Move:
        """A move-to command (start a new sub-path)."""
        end: Point3D
        """Target position ``(x, y, z)``."""

    class Line:
        """A line-to command (draw a straight line)."""
        end: Point3D
        """Target position ``(x, y, z)``."""

    class Arc:
        """An arc-to command (draw a circular arc)."""
        end: Point3D
        """Target position ``(x, y, z)``."""
        center_offset: Point
        """Offset from start to arc center ``(i, j)``."""
        clockwise: bool
        """``True`` if the arc runs clockwise."""

    class Bezier:
        """A cubic Bezier curve command."""
        end: Point3D
        """Target position ``(x, y, z)``."""
        control1: Point
        """First control point ``(c1x, c1y)``."""
        control2: Point
        """Second control point ``(c2x, c2y)``."""


def grow_geometry(geometry: Geometry, offset: float) -> Geometry:
    """Offset a geometry outward or inward.

    Args:
        geometry: Source geometry.
        offset: Positive grows outward, negative shrinks inward.

    Returns:
        A new offset :class:`Geometry`.
    """
    ...


def split_into_contours(geometry: Geometry) -> List[Geometry]:
    """Split into individual closed contours.

    Args:
        geometry: The geometry to split.

    Returns:
        One :class:`Geometry` per contour.
    """
    ...


def split_into_components(geometry: Geometry) -> List[Geometry]:
    """Split into disconnected components.

    Args:
        geometry: The geometry to split.

    Returns:
        One :class:`Geometry` per component.
    """
    ...


def get_bounding_rect_from_array(
    data: _ArrayLike,
) -> Tuple[float, float, float, float]:
    """Compute the 2D bounding box from a raw command array.

    Args:
        data: Command rows (each 8 floats).

    Returns:
        ``(x_min, y_min, x_max, y_max)``.
    """
    ...


def get_total_distance_from_array(data: List[List[float]]) -> float:
    """Compute total path length from a raw command array.

    Args:
        data: Command rows (each 8 floats).

    Returns:
        Sum of segment lengths.
    """
    ...


def extract_overcut_rows(
    data: Optional[_ArrayLike],
    max_length: float,
) -> Optional[NDArray[np.float64]]:
    """Extract overcut rows from command data.

    Args:
        data: Command rows, or ``None``.
        max_length: Maximum overcut length.

    Returns:
        An ``(N, 8)`` NumPy array of extracted rows, or ``None`` if
        *data* is ``None``.
    """
    ...


def get_subpath_vertices_from_array(
    data: _ArrayLike,
    subpath_index: int,
) -> List[Point]:
    """Get vertices of a specific sub-path from a raw command array.

    Args:
        data: Command rows (each 8 floats).
        subpath_index: Zero-based sub-path index.

    Returns:
        Ordered list of ``(x, y)`` vertices.
    """
    ...


def get_subpath_area_from_array(
    data: _ArrayLike,
    subpath_index: int,
) -> float:
    """Compute the signed area of a specific sub-path.

    Args:
        data: Command rows (each 8 floats).
        subpath_index: Zero-based sub-path index.

    Returns:
        Signed area — positive for CCW, negative for CW.
    """
    ...


def get_area_from_array(data: List[List[float]]) -> float:
    """Compute the total signed area of all sub-paths.

    Args:
        data: Command rows (each 8 floats).

    Returns:
        Signed area.
    """
    ...


def get_path_winding_order_from_array(
    data: _ArrayLike,
    start_cmd_index: int,
) -> str:
    """Determine winding order of a sub-path.

    Args:
        data: Command rows (each 8 floats).
        start_cmd_index: Index of the move-to command that starts the
            sub-path.

    Returns:
        ``"cw"``, ``"ccw"``, or ``"unknown"``.
    """
    ...


def get_point_tangent_at(
    data: _ArrayLike,
    row_index: int,
    t: float,
) -> Optional[Tuple[Point, Point]]:
    """Get position and tangent at parameter *t* on a command row.

    Args:
        data: Command rows (each 8 floats).
        row_index: Index of the command row.
        t: Parameter in ``[0, 1]``.

    Returns:
        ``((px, py), (tx, ty))`` or ``None`` on failure.
    """
    ...


def optimize_path_from_array(
    data: Optional[_ArrayLike],
    tolerance: float,
    fit_arcs: bool,
) -> NDArray[np.float64]:
    """Optimize a path by fitting curves and simplifying.

    Args:
        data: Command rows, or ``None`` for an empty result.
        tolerance: Maximum allowed deviation.
        fit_arcs: Whether to attempt arc fitting.

    Returns:
        An ``(N, 8)`` NumPy array of optimized command rows.
    """
    ...


def does_enclose(
    container: Geometry, content: Geometry,
) -> bool:
    """Check whether *container* fully encloses *content*.

    Args:
        container: The enclosing geometry.
        content: The geometry that may be enclosed.

    Returns:
        ``True`` if every point of *content* is inside *container*.

    Raises:
        RuntimeError: If the underlying computation fails.
    """
    ...


def fit_arcs(
    data: Optional[_ArrayLike],
    tolerance: float,
    progress_callback: Optional[object] = ...,
) -> Optional[List[List[float]]]:
    """Fit circular arcs to linear segments in the command data.

    Args:
        data: Command rows, or ``None`` to return ``None``.
        tolerance: Maximum allowed deviation.
        progress_callback: Optional callable receiving a float ``1.0``
            on completion.

    Returns:
        Command rows with arcs fitted, or ``None`` if *data* was
        ``None``.

    Raises:
        RuntimeError: If the fitting algorithm fails.
    """
    ...


def reverse_contour(contour: Geometry) -> Geometry:
    """Reverse the direction of a closed contour.

    Args:
        contour: The contour to reverse.

    Returns:
        A new :class:`Geometry` with reversed winding.

    Raises:
        RuntimeError: If the underlying computation fails.
    """
    ...


def split_inner_and_outer_contours(
    contours: List[Geometry],
) -> Tuple[List[Geometry], List[Geometry]]:
    """Partition contours into inner (hole) and outer groups.

    Args:
        contours: List of contour geometries.

    Returns:
        ``(inner, outer)`` tuple of geometry lists.

    Raises:
        RuntimeError: If the partitioning fails.
    """
    ...


def close_all_contours(geometry: Geometry) -> Geometry:
    """Ensure all sub-paths in the geometry are closed.

    Args:
        geometry: Source geometry.

    Returns:
        A new :class:`Geometry` with all contours closed.

    Raises:
        RuntimeError: If the closing operation fails.
    """
    ...


def normalize_winding_orders(
    contours: List[Geometry],
) -> List[Geometry]:
    """Normalize winding orders so outer contours are CCW and inner are CW.

    Args:
        contours: List of contour geometries.

    Returns:
        List of :class:`Geometry` objects with normalized winding.

    Raises:
        RuntimeError: If the normalization fails.
    """
    ...


def filter_to_external_contours(
    contours: List[Geometry],
) -> List[Geometry]:
    """Keep only the outermost contours, discarding holes.

    Args:
        contours: List of contour geometries.

    Returns:
        Only the external contours.

    Raises:
        RuntimeError: If the filtering fails.
    """
    ...


def remove_inner_edges(geometry: Geometry) -> Geometry:
    """Remove edges shared between adjacent sub-paths.

    Args:
        geometry: Source geometry.

    Returns:
        A new :class:`Geometry` with inner edges removed.

    Raises:
        RuntimeError: If the operation fails.
    """
    ...


def get_valid_contours_data(
    contour_geometries: List[Geometry],
) -> List[dict]:
    """Extract validated contour data for each geometry.

    Args:
        contour_geometries: List of contour geometries.

    Returns:
        A list of dicts, each with keys ``"geo"`` (Geometry),
        ``"vertices"`` (list of ``(x, y)``), ``"is_closed"`` (bool),
        and ``"original_index"`` (int).

    Raises:
        RuntimeError: If extraction fails.
    """
    ...


def close_geometry_gaps(
    geometry: Geometry,
    tolerance: float,
) -> Geometry:
    """Close small gaps between adjacent path segments.

    Args:
        geometry: Source geometry.
        tolerance: Maximum gap distance to close.

    Returns:
        A new :class:`Geometry` with gaps closed.

    Raises:
        RuntimeError: If the operation fails.
    """
    ...


def check_self_intersection(
    data: Optional[_ArrayLike],
    fail_on_t_junction: bool,
) -> bool:
    """Check whether a raw command array self-intersects.

    Args:
        data: Command rows, or ``None`` (returns ``False``).
        fail_on_t_junction: Treat T-junctions as intersections.

    Returns:
        ``True`` if a self-intersection is found.
    """
    ...


def check_intersection(
    data1: Optional[List[List[float]]],
    data2: Optional[List[List[float]]],
    fail_on_t_junction: bool,
) -> bool:
    """Check whether two raw command arrays intersect.

    Args:
        data1: First command rows, or ``None`` (returns ``False``).
        data2: Second command rows, or ``None`` (returns ``False``).
        fail_on_t_junction: Treat T-junctions as intersections.

    Returns:
        ``True`` if the two paths intersect.
    """
    ...


def check_self_intersection_from_array(
    data: _ArrayLike,
    fail_on_t_junction: bool,
) -> bool:
    """Like :func:`check_self_intersection` but requires non-optional *data*.

    Args:
        data: Command rows (each 8 floats).
        fail_on_t_junction: Treat T-junctions as intersections.

    Returns:
        ``True`` if a self-intersection is found.
    """
    ...


def check_intersection_from_array(
    data1: List[List[float]],
    data2: List[List[float]],
    fail_on_t_junction: bool,
) -> bool:
    """Like :func:`check_intersection` but requires non-optional data.

    Args:
        data1: First command rows (each 8 floats).
        data2: Second command rows (each 8 floats).
        fail_on_t_junction: Treat T-junctions as intersections.

    Returns:
        ``True`` if the two paths intersect.
    """
    ...


def remove_duplicate_segments(
    data: Optional[_ArrayLike],
    tolerance: float = ...,
) -> Optional[NDArray[np.float64]]:
    """Remove duplicate segments from raw command data.

    Args:
        data: Command rows, or ``None`` for ``None`` result.
        tolerance: Maximum distance to consider segments identical.
            Defaults to ``1e-6``.

    Returns:
        An ``(N, 8)`` NumPy array with duplicates removed, or ``None``.
    """
    ...


def flatten_to_points(
    data: Optional[_ArrayLike],
    tolerance: float,
) -> List[List[Point3D]]:
    """Flatten curves to point sequences per sub-path.

    Args:
        data: Command rows, or ``None`` for an empty list.
        tolerance: Maximum linearisation error.

    Returns:
        A list of sub-paths, each a list of ``(x, y, z)`` points.
    """
    ...


def linearize_geometry(
    data: Optional[_ArrayLike],
    tolerance: float,
) -> NDArray[np.float64]:
    """Convert all curves in the data to line segments.

    Args:
        data: Command rows, or ``None`` for an empty array.
        tolerance: Maximum linearisation error.

    Returns:
        An ``(N, 8)`` NumPy array of linearised command rows.
    """
    ...


def create_line_cmd(end_point: Union[Point, Point3D]) -> List[float]:
    """Build a single line-to command row.

    Args:
        end_point: Target ``(x, y, z)``.

    Returns:
        An 8-element float list.
    """
    ...


def create_arc_cmd(
    end: Point3D,
    center: Point,
    start: Point3D,
) -> List[float]:
    """Build a single arc-to command row.

    Args:
        end: Arc end point ``(x, y, z)``.
        center: Arc center ``(x, y)``.
        start: Arc start point ``(x, y, z)``.

    Returns:
        An 8-element float list.
    """
    ...


def convert_arc_to_beziers_from_array(
    start: Point3D,
    end: Point3D,
    center_offset: Point,
    clockwise: bool,
) -> List[List[float]]:
    """Convert an arc to one or more cubic Bezier command rows.

    Args:
        start: Arc start ``(x, y, z)``.
        end: Arc end ``(x, y, z)``.
        center_offset: Offset from start to arc center ``(i, j)``.
        clockwise: ``True`` for clockwise.

    Returns:
        A list of 8-element float command rows.
    """
    ...


def fit_curves(
    data: Optional[_ArrayLike],
    tolerance: float,
    preserve_beziers: bool,
    preserve_arcs: bool,
) -> NDArray[np.float64]:
    """Fit curves (Beziers and/or arcs) to linear command data.

    Args:
        data: Command rows, or ``None`` for an empty array.
        tolerance: Maximum fitting deviation.
        preserve_beziers: Keep existing Bezier segments.
        preserve_arcs: Keep existing arc segments.

    Returns:
        An ``(N, 8)`` NumPy array of fitted commands.
    """
    ...


def _are_points_equal(
    p1: Point3D,
    p2: Point3D,
    tolerance: float,
) -> bool:
    """Check if two 3D points are within *tolerance* of each other.

    Args:
        p1: First point ``(x, y, z)``.
        p2: Second point ``(x, y, z)``.
        tolerance: Maximum Euclidean distance.

    Returns:
        ``True`` if the points are equal within tolerance.
    """
    ...


def _get_segment_key(
    data: _ArrayLike,
    index: int,
    _tolerance: float,
) -> Optional[Tuple[str, ...]]:
    """Extract a hashable segment key for deduplication.

    Args:
        data: Command rows (each 8 floats).
        index: Row index to extract the key from.
        _tolerance: Tolerance (currently unused).

    Returns:
        A tuple like ``("LINE", (x, y, z))``,
        ``("ARC", end, center, clockwise)``, or
        ``("BEZIER", end, c1, c2)``, or ``None``.
    """
    ...


def _are_segments_equal(
    key1: object,
    key2: object,
    tolerance: float,
) -> bool:
    """Compare two segment keys for equality within tolerance.

    Args:
        key1: First segment key (as returned by :func:`_get_segment_key`).
        key2: Second segment key.
        tolerance: Maximum coordinate difference.

    Returns:
        ``True`` if the segments are equal.
    """
    ...


def _partial_segment_from_row(
    row: _ArrayLike,
    start_point: Point3D,
    t: float,
) -> Optional[List[float]]:
    """Compute a partial segment from *start_point* to parameter *t*.

    Args:
        row: A single command row (8 floats).
        start_point: Segment start ``(x, y, z)``.
        t: Parameter in ``[0, 1]`` — how far along the segment.

    Returns:
        An 8-element command row for the partial segment, or ``None``.
    """
    ...


def _segment_length_from_row(
    row: _ArrayLike,
    start_point: Point3D,
) -> float:
    """Compute the length of a single segment.

    Args:
        row: A single command row (8 floats).
        start_point: Segment start ``(x, y, z)``.

    Returns:
        Euclidean length of the segment.
    """
    ...


def apply_affine_transform_to_array(
    data: _ArrayLike,
    matrix: Union[List[List[float]], NDArray[np.float64]],
) -> NDArray[np.float64]:
    """Apply a 4×4 affine transform to a raw command array.

    Args:
        data: Command rows (each 8 floats).
        matrix: A ``4×4`` list-of-lists of floats.

    Returns:
        An ``(N, 8)`` NumPy array of transformed commands.
    """
    ...


def map_geometry_to_frame(
    geometry: Geometry,
    origin: Point,
    p_width: Tuple[float, float],
    p_height: Tuple[float, float],
    anchor_y: Optional[float] = ...,
    stable_src_height: Optional[float] = ...,
    anchor_x: Optional[float] = ...,
    stable_src_width: Optional[float] = ...,
) -> Geometry:
    """Map a geometry into a rectangular frame with anchoring support.

    Args:
        geometry: Source geometry.
        origin: Frame origin ``(x, y)``.
        p_width: ``(source_width, target_width)`` pair.
        p_height: ``(source_height, target_height)`` pair.
        anchor_y: Optional y anchor position.
        stable_src_height: Source height to keep stable during mapping.
        anchor_x: Optional x anchor position.
        stable_src_width: Source width to keep stable during mapping.

    Returns:
        A new :class:`Geometry` transformed to the frame.
    """
    ...


def get_angle_at_vertex(
    p0: Point,
    p1: Point,
    p2: Point,
) -> float:
    """Compute the angle at the middle vertex *p1* formed by the
    triangle ``p0 – p1 – p2``.

    Args:
        p0: First point ``(x, y)``.
        p1: Vertex point ``(x, y)``.
        p2: Third point ``(x, y)``.

    Returns:
        Angle in radians at *p1*.
    """
    ...


def remove_duplicates(points: List[Point]) -> List[Point]:
    """Remove consecutive duplicate points.

    Args:
        points: A list of ``(x, y)`` points.

    Returns:
        Deduplicated list.
    """
    ...


def is_clockwise(
    points: List[Tuple[float, ...]],
) -> bool:
    """Check whether a polygon winds clockwise.

    Args:
        points: Polygon vertices (2D or 3D; z is ignored).

    Returns:
        ``True`` if the polygon is clockwise.

    Raises:
        ValueError: If a point is not a 2- or 3-tuple of floats.
    """
    ...


def is_closed(
    commands: _ArrayLike,
    tolerance: float = ...,
) -> bool:
    """Check whether a raw command array forms a closed path.

    Args:
        commands: Command rows (each 8 floats).
        tolerance: Maximum gap between start and end.  Defaults to
            ``1e-6``.

    Returns:
        ``True`` if the path is closed.
    """
    ...


def get_outward_normal_at_from_array(
    data: _ArrayLike,
    row_index: int,
    t: float,
) -> Optional[Point]:
    """Get outward normal at parameter *t* on a specific command row.

    Args:
        data: Command rows (each 8 floats).
        row_index: Index of the command row.
        t: Parameter in ``[0, 1]``.

    Returns:
        ``(nx, ny)`` outward normal, or ``None`` if the data is empty.
    """
    ...


def get_point_tangent_at_py(
    data: List[List[float]],
    row_index: int,
    t: float,
) -> Optional[Tuple[Point, Point]]:
    """Get position and tangent at parameter *t* on a command row.

    Args:
        data: Command rows (each 8 floats).
        row_index: Index of the command row.
        t: Parameter in ``[0, 1]``.

    Returns:
        ``((px, py), (tx, ty))`` or ``None`` on failure.
    """
    ...
