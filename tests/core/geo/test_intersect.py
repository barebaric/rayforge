import pytest
import numpy as np
from rayforge.core.geo import Geometry


@pytest.fixture
def square_geometry() -> Geometry:
    """A simple 10x10 square, which does not self-intersect."""
    return Geometry.from_points([(0, 0), (10, 0), (10, 10), (0, 10)])


@pytest.fixture
def figure_eight_geometry() -> Geometry:
    """A figure-eight shape, which does self-intersect."""
    return Geometry.from_points(
        [(0, 0), (10, 10), (0, 10), (10, 0)], close=True
    )


@pytest.fixture
def t_junction_geometry() -> Geometry:
    """A shape with a T-junction that is not a true crossing."""
    geo = Geometry()
    geo.move_to(-10, 0)
    geo.line_to(10, 0)  # Segment 1: from (-10,0) to (10,0)
    geo.line_to(10, 10)  # Segment 2
    geo.line_to(0, 10)  # Segment 3
    geo.line_to(
        0, 0
    )  # Segment 4: from (0,10) to (0,0). Its endpoint lies on Segment 1.
    return geo


def test_no_self_intersection_square(square_geometry):
    """A simple square should not have self-intersections."""
    assert not square_geometry.has_self_intersections()


def test_self_intersection_figure_eight(figure_eight_geometry):
    """A figure-eight shape should be detected as self-intersecting."""
    assert figure_eight_geometry.has_self_intersections()


def test_no_self_intersection_touching_endpoint():
    """
    A V-shape where segments touch at a vertex is not a self-intersection.
    """
    geo = Geometry()
    geo.move_to(0, 10)
    geo.line_to(5, 0)
    geo.line_to(10, 10)
    assert not geo.has_self_intersections()


def test_self_intersection_t_junction_configurable(t_junction_geometry):
    """
    A T-junction should not be a self-intersection by default, but can be
    configured to fail.
    """
    # By default, T-junctions are allowed (not considered intersections)
    assert not t_junction_geometry.has_self_intersections(
        fail_on_t_junction=False
    )
    assert not t_junction_geometry.has_self_intersections()  # Test default

    # With the flag, it should be detected as an intersection
    assert t_junction_geometry.has_self_intersections(fail_on_t_junction=True)


def test_self_intersection_with_arc():
    """
    Test a path where a line segment intersects its preceding arc segment at a
    point other than their shared vertex.
    """
    geo = Geometry()
    # A CCW semicircle from (10,0) to (0,0) with center (5,0)
    geo.move_to(10, 0)
    geo.arc_to(0, 0, i=-5, j=0, clockwise=False)

    # A line from the end of the arc (0,0) to the top of the arc (5,5).
    # This line segment intersects the arc at (5,5), which is not the
    # shared start/end vertex (0,0). This should be detected.
    geo.line_to(5, 5)
    assert geo.has_self_intersections()


def test_no_self_intersection_multiple_subpaths():
    """
    Test two separate, non-intersecting squares in the same geometry object.
    """
    geo = Geometry.from_points([(0, 0), (5, 0), (5, 5), (0, 5)])
    # Add a second, separate square
    second_square = Geometry.from_points(
        [(10, 10), (15, 10), (15, 15), (10, 15)]
    )
    geo.extend(second_square)
    assert not geo.has_self_intersections()


def test_self_intersection_in_one_of_multiple_subpaths(
    square_geometry, figure_eight_geometry
):
    """
    Test a geometry containing one simple shape and one self-intersecting
    shape.
    """
    # Offset the figure eight to ensure it doesn't intersect the square
    translation_matrix = np.array(
        [
            [1, 0, 0, 20],
            [0, 1, 0, 20],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ],
        dtype=float,
    )
    figure_eight_geometry.transform(translation_matrix)

    # Combine them
    combined_geo = square_geometry
    combined_geo.extend(figure_eight_geometry)

    assert combined_geo.has_self_intersections()


# --- Tests for intersects_with() ---


def test_no_intersection_separate_shapes():
    """Two geometries that are far apart should not intersect."""
    geo1 = Geometry.from_points([(0, 0), (1, 0), (1, 1), (0, 1)])
    geo2 = Geometry.from_points([(10, 10), (11, 10), (11, 11), (10, 11)])
    assert not geo1.intersects_with(geo2)
    assert not geo2.intersects_with(geo1)


def test_intersection_crossing_shapes():
    """Two geometries that cross each other should intersect."""
    geo1 = Geometry.from_points([(0, 5), (10, 5)], close=False)
    geo2 = Geometry.from_points([(5, 0), (5, 10)], close=False)
    assert geo1.intersects_with(geo2)
    assert geo2.intersects_with(geo1)


def test_intersection_touching_shapes():
    """
    Two squares that touch along an edge should be considered intersecting.
    """
    geo1 = Geometry.from_points([(0, 0), (10, 0), (10, 10), (0, 10)])
    geo2 = Geometry.from_points([(10, 0), (20, 0), (20, 10), (10, 10)])
    assert geo1.intersects_with(geo2)
    assert geo2.intersects_with(geo1)


def test_no_intersection_one_inside_another():
    """
    A shape fully contained within another does not intersect its boundary.
    """
    geo_outer = Geometry.from_points([(0, 0), (20, 0), (20, 20), (0, 20)])
    geo_inner = Geometry.from_points([(5, 5), (15, 5), (15, 15), (5, 15)])
    assert not geo_outer.intersects_with(geo_inner)
    assert not geo_inner.intersects_with(geo_outer)


def test_intersection_with_arc():
    """Test intersection between a square and an arc that passes through it."""
    square = Geometry.from_points([(0, 0), (10, 0), (10, 10), (0, 10)])
    arc = Geometry()
    arc.move_to(-5, 5)
    # This arc starts at (-5,5), ends at (15,5) and bows up,
    # passing through the square.
    arc.arc_to(15, 5, i=10, j=10, clockwise=False)
    assert square.intersects_with(arc)
    assert arc.intersects_with(square)


def test_no_intersection_bounding_box_overlap():
    """
    Tests two L-shapes in a yin-yang configuration whose bounding boxes
    overlap but whose paths do not intersect.
    """
    geo1 = Geometry.from_points(
        [(0, 0), (10, 0), (10, 1), (1, 1), (1, 10), (0, 10)]
    )
    geo2 = Geometry.from_points([(2, 2), (9, 2), (9, 9), (2, 9)], close=False)
    assert not geo1.intersects_with(geo2)
    assert not geo2.intersects_with(geo1)
