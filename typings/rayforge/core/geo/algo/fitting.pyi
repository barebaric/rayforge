"""Curve fitting — circle fitting, Bezier conversion, and primitive
extraction from point sequences."""

from typing import List, Optional, Tuple

from typing import List, Optional, Tuple, TypeAlias, Union

from rayforge.core.geo import Point, Point3D

_Point2DOr3D: TypeAlias = Union[Tuple[float, float], Tuple[float, float, float]]


def are_points_collinear(
    points: List[Point3D],
    tolerance: float = ...,
) -> bool:
    """Check whether all points lie on (approximately) the same line.

    Args:
        points: Points to test, each ``(x, y, z)``.
        tolerance: Maximum deviation from collinearity.
            Defaults to ``1e-6``.

    Returns:
        ``True`` if all points are collinear within *tolerance*.
    """
    ...


def fit_circle_to_3_points(
    p1: _Point2DOr3D,
    p2: _Point2DOr3D,
    p3: _Point2DOr3D,
) -> Optional[Tuple[Point, float]]:
    """Fit a circle through exactly three 3D points.

    Only the ``x`` and ``y`` coordinates are used.

    Args:
        p1: First point ``(x, y, z)``.
        p2: Second point ``(x, y, z)``.
        p3: Third point ``(x, y, z)``.

    Returns:
        ``((cx, cy), radius)`` or ``None`` if the points are collinear.
    """
    ...


def fit_circle_to_points(
    points: List[Point3D],
) -> Optional[Tuple[Point, float, float]]:
    """Fit a circle to a set of points using least-squares.

    Args:
        points: Points to fit, each ``(x, y, z)``.

    Returns:
        ``((cx, cy), radius, error)`` or ``None`` if fitting fails.
    """
    ...


def project_circle_center_to_bisector(
    p1: _Point2DOr3D,
    p2: _Point2DOr3D,
    center: Point,
) -> Point:
    """Project a circle center onto the perpendicular bisector of two
    points.

    Args:
        p1: First point ``(x, y, z)``.
        p2: Second point ``(x, y, z)``.
        center: Candidate circle center ``(x, y)``.

    Returns:
        Adjusted center ``(x, y)`` on the bisector.
    """
    ...


def flatten_to_points(
    data: List[List[float]],
    tolerance: float,
) -> List[List[Point3D]]:
    """Convert a raw command array into point sequences per sub-path.

    Curves are linearised within *tolerance*.

    Args:
        data: Command rows (each 8 floats).
        tolerance: Maximum linearisation error.

    Returns:
        A list of sub-paths, each a list of ``(x, y, z)`` points.
    """
    ...


def linearize_geometry(
    data: List[List[float]],
    tolerance: float,
) -> List[List[float]]:
    """Convert all curves in a command array to line segments.

    Args:
        data: Command rows (each 8 floats).
        tolerance: Maximum linearisation error.

    Returns:
        Linearised command rows (each 8 floats).
    """
    ...


def create_line_cmd(end_point: _Point2DOr3D) -> List[float]:
    """Build a single line-to command row.

    Args:
        end_point: Target ``(x, y, z)``.

    Returns:
        An 8-element float list.
    """
    ...


def create_arc_cmd(
    end: _Point2DOr3D,
    center: Point,
    start: _Point2DOr3D,
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
    start: _Point2DOr3D,
    end: _Point2DOr3D,
    center_offset: Point,
    clockwise: bool,
) -> List[List[float]]:
    """Convert an arc to cubic Bezier command rows.

    Args:
        start: Arc start ``(x, y, z)``.
        end: Arc end ``(x, y, z)``.
        center_offset: ``(i, j)`` offset from start to arc center.
        clockwise: ``True`` for clockwise.

    Returns:
        List of 8-element float command rows.
    """
    ...


def fit_points_recursive(
    points: List[Point3D],
    tolerance: float,
    start_idx: int,
    end_idx: int,
) -> List[List[float]]:
    """Recursively fit curves to a subsequence of points.

    Args:
        points: Full point sequence.
        tolerance: Maximum fitting deviation.
        start_idx: Start index (inclusive).
        end_idx: End index (inclusive).

    Returns:
        Command rows representing the fitted curves.
    """
    ...


def fit_points_with_primitives(
    points: List[Point3D],
    tolerance: float,
) -> List[List[float]]:
    """Fit lines, arcs, and Bezier curves to a point sequence.

    Args:
        points: Ordered 3D points.
        tolerance: Maximum deviation from the original data.

    Returns:
        Command rows (each 8 floats).
    """
    ...


def get_polyline_line_deviation(
    points: List[Point3D],
    start: int,
    end: int,
) -> Tuple[float, int]:
    """Compute the maximum deviation of a point subsequence from the
    straight line connecting the endpoints.

    Args:
        points: Full point sequence.
        start: Start index (inclusive).
        end: End index (inclusive).

    Returns:
        ``(max_deviation, max_index)`` — the largest perpendicular
        distance and the index where it occurs.
    """
    ...


def get_polyline_arc_deviation(
    points: List[Point3D],
    center: Point,
    radius: float,
) -> float:
    """Compute the RMS deviation of points from a circular arc.

    Args:
        points: Points to measure.
        center: Arc center ``(x, y)``.
        radius: Arc radius.

    Returns:
        RMS deviation from the ideal arc.
    """
    ...
