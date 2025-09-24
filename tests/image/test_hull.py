import numpy as np
import cv2
from rayforge.image.hull import get_enclosing_hull, get_concave_hull
from rayforge.image.tracing import Geometry


def test_get_enclosing_hull():
    """
    Tests the logic of creating a single enclosing convex hull from a boolean
    image with multiple distinct components.
    """
    # Create a boolean image with two distinct, non-touching squares
    boolean_image = np.full((100, 100), False, dtype=bool)
    boolean_image[10:30, 10:30] = True
    boolean_image[70:90, 70:90] = True

    # Define test parameters
    scale_x, scale_y = 10.0, 10.0
    height_px = 100
    border_size = 2  # As defined in tracing.py

    geometries = get_enclosing_hull(
        boolean_image, scale_x, scale_y, height_px, border_size
    )

    # We should get exactly ONE geometry object back
    assert len(geometries) == 1
    geo = geometries[0]
    assert isinstance(geo, Geometry)

    # The convex hull of two diagonally offset squares is a hexagon
    # (6 vertices).
    # A 6-vertex shape results in 7 commands:
    # 1 MoveTo, 5 LineTo's, and 1 final LineTo from close_path().
    assert len(geo.commands) == 7


def test_get_enclosing_hull_no_content():
    """
    Tests that get_enclosing_hull returns an empty list for an image with
    no content.
    """
    boolean_image = np.full((50, 50), False, dtype=bool)
    geometries = get_enclosing_hull(
        boolean_image, scale_x=1.0, scale_y=1.0, height_px=50, border_size=2
    )
    assert geometries == []


def test_get_concave_hull_with_gravity():
    """
    Tests that the concave hull with gravity has a smaller area than the
    standard convex hull, proving it has been pulled inwards.
    """
    # Create a C-shape where a convex hull would cover the empty middle
    img = np.zeros((100, 100), dtype=np.uint8)
    img[10:90, 10:25] = 255  # Left bar
    img[10:25, 25:90] = 255  # Top bar
    img[75:90, 25:90] = 255  # Bottom bar

    # Get the points from the image contours for area calculation
    contours, _ = cv2.findContours(
        img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    all_points = np.vstack(contours)

    # 1. Calculate the area of the standard convex hull
    convex_hull_points = cv2.convexHull(all_points)
    convex_area = cv2.contourArea(convex_hull_points)

    # 2. Generate the concave hull points
    gravity = 0.2
    center = np.mean(all_points.astype(np.float64), axis=0)
    concave_points = []
    for point in convex_hull_points.squeeze(axis=1):
        vector = center - point
        new_point = point + vector * gravity
        concave_points.append(new_point)
    concave_hull_points = np.array(concave_points, dtype=np.int32)
    concave_area = cv2.contourArea(concave_hull_points)

    # Assert that the "gravity" has successfully reduced the hull's area
    assert concave_area < convex_area
    assert concave_area > 0


def test_get_concave_hull_zero_gravity():
    """
    Tests that a concave hull with zero gravity is identical to a convex hull.
    """
    boolean_image = np.zeros((100, 100), dtype=bool)
    boolean_image[20:80, 20:80] = True  # A simple square

    # We can compare the resulting Geometry objects
    convex_geo = get_enclosing_hull(boolean_image, 1.0, 1.0, 100, 0)[0]
    concave_geo = get_concave_hull(
        boolean_image, 1.0, 1.0, 100, 0, gravity=0.0
    )[0]

    assert len(convex_geo.commands) == len(concave_geo.commands)
    # Check if the command points are identical
    for cmd1, cmd2 in zip(convex_geo.commands, concave_geo.commands):
        assert cmd1.end == cmd2.end
