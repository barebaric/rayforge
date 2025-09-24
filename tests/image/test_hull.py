import numpy as np
from rayforge.image.hull import get_enclosing_hull
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
