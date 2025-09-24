import numpy as np
import cv2
from typing import Tuple, Optional, List
from ..core.geo import Geometry


def polygon_to_geometry(
    points: np.ndarray,
    scale_x: float,
    scale_y: float,
    height_px: int,
    border_size: int,
) -> Optional[Geometry]:
    """
    Helper to convert a simple polygon (list of vertices) into a Geometry
    object.
    """
    if points is None or len(points) < 3:
        return None

    squeezed_points = points.squeeze()

    def _transform_point(p) -> Tuple[float, float]:
        px, py = np.asarray(p).squeeze()
        ops_px = px - border_size
        ops_py = height_px - (py - border_size)
        return ops_px / scale_x, ops_py / scale_y

    geo = Geometry()
    start_pt = _transform_point(squeezed_points[0])
    geo.move_to(start_pt[0], start_pt[1])

    for point in squeezed_points[1:]:
        pt = _transform_point(point)
        geo.line_to(pt[0], pt[1])

    geo.close_path()
    return geo


def _curves_to_geometry(
    segments: List[Tuple[np.ndarray, np.ndarray, np.ndarray]],
    scale_x: float,
    scale_y: float,
    height_px: int,
    border_size: int,
    samples_per_curve: int = 20,
) -> Optional[Geometry]:
    """
    Helper to convert a list of quadratic Bézier segments into a Geometry
    object.
    """
    if not segments:
        return None

    def _transform_point(p) -> Tuple[float, float]:
        px, py = np.asarray(p).squeeze()
        ops_px = px - border_size
        ops_py = height_px - (py - border_size)
        return ops_px / scale_x, ops_py / scale_y

    geo = Geometry()
    # Move to the start of the very first segment
    start_pt = _transform_point(segments[0][0])
    geo.move_to(start_pt[0], start_pt[1])

    # Generate points for each curve
    for p0, p1, p2 in segments:
        # Use np.linspace for t to generate points along the curve
        t = np.linspace(0, 1, samples_per_curve)
        # Quadratic Bézier formula: (1-t)^2*P0 + 2(1-t)t*P1 + t^2*P2
        curve_points = (
            np.outer((1 - t) ** 2, p0)
            + np.outer(2 * (1 - t) * t, p1)
            + np.outer(t**2, p2)
        )
        # Append line_to commands for each point on the curve (except
        # the first)
        for point in curve_points[1:]:
            pt = _transform_point(point)
            geo.line_to(pt[0], pt[1])

    geo.close_path()
    return geo


def get_enclosing_hull(
    boolean_image: np.ndarray,
    scale_x: float,
    scale_y: float,
    height_px: int,
    border_size: int,
) -> Optional[Geometry]:
    """
    Calculates a single convex hull that encompasses all content in the image.
    """
    img_uint8 = boolean_image.astype(np.uint8) * 255
    contours, _ = cv2.findContours(
        img_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    if not contours:
        return None

    all_points = np.vstack(contours)
    if len(all_points) < 3:
        return None

    hull_points = cv2.convexHull(all_points)
    return polygon_to_geometry(
        hull_points, scale_x, scale_y, height_px, border_size
    )


def get_hulls_from_image(
    boolean_image: np.ndarray,
    scale_x: float,
    scale_y: float,
    height_px: int,
    border_size: int,
) -> List[Geometry]:
    """
    Finds all distinct contours in a boolean image, calculates the convex
    hull for each, and returns them as a list of Geometry objects.

    Args:
        boolean_image: The clean boolean image containing only major shapes.
        scale_x: Pixels per millimeter (X).
        scale_y: Pixels per millimeter (Y).
        height_px: Original height of the source surface in pixels.
        border_size: The pixel border size added during pre-processing.

    Returns:
        A list of Geometry objects, each representing a convex hull.
    """
    geometries = []
    img_uint8 = boolean_image.astype(np.uint8) * 255
    contours, _ = cv2.findContours(
        img_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    for contour in contours:
        # A hull requires at least 3 points, which is checked inside
        # polygon_to_geometry.
        hull_points = cv2.convexHull(contour)
        geo = polygon_to_geometry(
            hull_points, scale_x, scale_y, height_px, border_size
        )
        if geo:
            geometries.append(geo)

    return geometries


def get_concave_hull(
    boolean_image: np.ndarray,
    scale_x: float,
    scale_y: float,
    height_px: int,
    border_size: int,
    gravity: float = 0.1,
) -> Optional[Geometry]:
    """
    Calculates a smooth, constrained concave hull that "shrink-wraps" the
    content geometrically, mimicking a physical rubber band using Bézier
    curves.
    """
    effective_gravity = np.clip(gravity, 0.0, 1.0)
    if effective_gravity < 1e-6:
        return get_enclosing_hull(
            boolean_image, scale_x, scale_y, height_px, border_size
        )

    img_uint8 = boolean_image.astype(np.uint8) * 255
    input_contours, _ = cv2.findContours(
        img_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    if not input_contours:
        return None

    all_points = np.vstack(input_contours)
    if len(all_points) < 3:
        return get_enclosing_hull(
            boolean_image, scale_x, scale_y, height_px, border_size
        )

    # 1. Establish the "posts" - the vertices of the enclosing convex hull.
    hull_vertices = cv2.convexHull(all_points).squeeze(axis=1)

    # 2. Establish all "attractors" - every point on the input shapes.
    all_contour_points = np.vstack(input_contours).squeeze(axis=1)

    # 3. For each segment of the hull, calculate its corresponding Bézier
    # curve.
    bezier_segments = []
    num_hull_pts = len(hull_vertices)
    for i in range(num_hull_pts):
        p0 = hull_vertices[i]
        p2 = hull_vertices[(i + 1) % num_hull_pts]
        midpoint = (p0 + p2) / 2.0

        # Find the closest attractor point to the midpoint of the segment.
        distances = np.linalg.norm(all_contour_points - midpoint, axis=1)
        closest_attractor = all_contour_points[np.argmin(distances)]

        # 1. Find the target point where the curve should sag to. This point
        # lies on the line between the midpoint and the attractor.
        target_sag_point = (
            midpoint * (1 - effective_gravity)
            + closest_attractor * effective_gravity
        )

        # 2. To make a quadratic Bézier curve pass through target_sag_point at
        # its apex (t=0.5), the control point p1 must be "exaggerated" by a
        # factor of 2 along the vector from the midpoint.
        control_point = midpoint + 2 * (target_sag_point - midpoint)

        bezier_segments.append((p0, control_point, p2))

    # 4. Convert the list of smooth Bézier curves into a final Geometry object.
    return _curves_to_geometry(
        bezier_segments, scale_x, scale_y, height_px, border_size
    )
