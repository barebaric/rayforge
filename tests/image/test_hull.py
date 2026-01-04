import numpy as np
import cv2
from rayforge.core.geo import Geometry
from rayforge.image.hull import (
    get_enclosing_hull,
    get_concave_hull,
    get_hulls_from_image,
)


def draw_rounded_rectangle(img, pt1, pt2, corner_radius, color, thickness):
    """Helper to draw the test shape."""
    x1, y1 = pt1
    x2, y2 = pt2
    r = corner_radius
    cv2.circle(img, (x1 + r, y1 + r), r, color, thickness)
    cv2.circle(img, (x2 - r, y1 + r), r, color, thickness)
    cv2.circle(img, (x1 + r, y2 - r), r, color, thickness)
    cv2.circle(img, (x2 - r, y2 - r), r, color, thickness)
    cv2.rectangle(img, (x1 + r, y1), (x2 - r, y2), color, thickness)
    cv2.rectangle(img, (x1, y1 + r), (x2, y2 - r), color, thickness)


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

    geo = get_enclosing_hull(
        boolean_image, scale_x, scale_y, height_px, border_size
    )

    # We should get exactly ONE geometry object back
    assert geo is not None
    assert isinstance(geo, Geometry)

    # The convex hull of two diagonally offset squares is a hexagon
    # (6 vertices).
    # A 6-vertex shape results in 7 commands:
    # 1 MoveTo, 5 LineTo's, and 1 final LineTo from close_path().
    assert len(geo) == 7


def test_get_enclosing_hull_no_content():
    """
    Tests that get_enclosing_hull returns None for an image with
    no content.
    """
    boolean_image = np.full((50, 50), False, dtype=bool)
    geo = get_enclosing_hull(boolean_image, 1.0, 1.0, 50, 2)
    assert geo is None


def test_get_hulls_from_image_multiple_components():
    """
    Tests that get_hulls_from_image correctly returns a separate convex hull
    for each distinct component in the image.
    """
    # Create a boolean image with two distinct, non-touching squares
    boolean_image = np.full((100, 100), False, dtype=bool)
    boolean_image[10:30, 10:30] = True
    boolean_image[70:90, 70:90] = True

    # Define test parameters
    scale_x, scale_y = 10.0, 10.0
    height_px = 100
    border_size = 2

    geometries = get_hulls_from_image(
        boolean_image, scale_x, scale_y, height_px, border_size
    )

    # We should get a list of TWO geometry objects back
    assert isinstance(geometries, list)
    assert len(geometries) == 2
    assert all(isinstance(g, Geometry) for g in geometries)

    # The convex hull of a square is a square (4 vertices).
    # A 4-vertex shape results in 5 commands:
    # 1 MoveTo, 3 LineTo's, and 1 final LineTo from close_path().
    assert len(geometries[0]) == 5
    assert len(geometries[1]) == 5


def test_get_hulls_from_image_no_content():
    """
    Tests that get_hulls_from_image returns an empty list for an image
    with no content.
    """
    boolean_image = np.full((50, 50), False, dtype=bool)
    geometries = get_hulls_from_image(boolean_image, 1.0, 1.0, 50, 2)
    assert isinstance(geometries, list)
    assert geometries == []


def test_concave_hull_creates_valid_indentation():
    height, width = 200, 200
    uint8_image = np.zeros((height, width), dtype=np.uint8)
    w_bottom, w_top, rect_h, gap, radius = 30, 150, 40, 20, 10
    top_y1 = (height - (2 * rect_h + gap)) // 2
    top_x1 = (width - w_top) // 2
    draw_rounded_rectangle(
        uint8_image,
        (top_x1, top_y1),
        (top_x1 + w_top, top_y1 + rect_h),
        radius,
        255,
        -1,
    )
    bottom_y1 = top_y1 + rect_h + gap
    bottom_x1 = (width - w_bottom) // 2
    draw_rounded_rectangle(
        uint8_image,
        (bottom_x1, bottom_y1),
        (bottom_x1 + w_bottom, bottom_y1 + rect_h),
        radius,
        255,
        -1,
    )

    boolean_image = uint8_image.astype(bool)
    convex_geo = get_concave_hull(
        boolean_image, 1.0, 1.0, height, 0, gravity=0.0
    )
    # A high gravity is needed for the algorithm to pick up the relatively
    # small indentation in the test shape.
    concave_geo = get_concave_hull(
        boolean_image, 1.0, 1.0, height, 0, gravity=0.9
    )

    assert convex_geo is not None
    assert concave_geo is not None, (
        "Concave hull with gravity must not be None"
    )

    # 1. Check basic properties: more vertices and smaller area than convex.
    assert len(concave_geo) > len(convex_geo), (
        "Concave hull should have more vertices than the convex hull."
    )
    assert convex_geo.area() > 0
    assert concave_geo.area() < convex_geo.area()

    # 2. Check for self-intersection using the new Geometry method.
    assert not concave_geo.has_self_intersections()

    # TODO: Improve algo to meet the following expectations
    """
    # 3. Check that shrinking the hull did not create new intersections
    convex_geo.grow(1)  # grow to avoid touching due to floating point errors
    assert not concave_geo.intersects_with(convex_geo), (
       f"Intersects with convex hull: {concave_geo.dump()}"
    )

    # 4. Check that the hull encloses all points of the original shape.
    assert concave_geo.encloses(convex_geo), f"Not enclosed: {concave_geo}"
    """


def test_get_concave_hull_zero_gravity():
    """
    Tests that a concave hull with zero gravity is identical to a convex hull.
    """
    boolean_image = np.zeros((100, 100), dtype=bool)
    boolean_image[20:80, 20:80] = True  # A simple square

    # We can compare the resulting Geometry objects
    convex_geo = get_enclosing_hull(boolean_image, 1.0, 1.0, 100, 0)
    concave_geo = get_concave_hull(
        boolean_image, 1.0, 1.0, 100, 0, gravity=0.0
    )

    assert convex_geo is not None
    assert concave_geo is not None
    assert len(convex_geo) == len(concave_geo)

    # Check if the command points are identical
    np.testing.assert_array_equal(convex_geo.data, concave_geo.data)
