from rayforge.core.geo import Geometry
from rayforge.core.geo.split import split_into_contours, split_into_components


def test_split_into_contours_empty():
    """Tests splitting an empty Geometry object."""
    geo = Geometry()
    contours = split_into_contours(geo)
    assert len(contours) == 0


def test_split_into_contours_single():
    """Tests splitting a Geometry with a single continuous path."""
    geo = Geometry()
    geo.move_to(0, 0)
    geo.line_to(10, 0)
    geo.line_to(10, 10)
    contours = split_into_contours(geo)
    assert len(contours) == 1
    assert len(contours[0].commands) == 3
    assert contours[0].commands[0].end == (0, 0, 0)


def test_split_into_contours_multiple_disjoint():
    """Tests splitting a Geometry with multiple separate paths."""
    geo = Geometry()
    # Contour 1
    geo.move_to(0, 0)
    geo.line_to(1, 1)
    # Contour 2
    geo.move_to(10, 10)
    geo.line_to(11, 11)
    geo.line_to(12, 12)
    contours = split_into_contours(geo)
    assert len(contours) == 2
    assert len(contours[0].commands) == 2
    assert len(contours[1].commands) == 3
    assert contours[0].commands[0].end == (0, 0, 0)
    assert contours[1].commands[0].end == (10, 10, 0)


def test_split_into_contours_no_initial_move_to():
    """Tests splitting a Geometry that starts with a drawing command."""
    geo = Geometry()
    geo.line_to(5, 5)  # Implicit start
    geo.line_to(10, 0)
    contours = split_into_contours(geo)
    assert len(contours) == 1
    assert len(contours[0].commands) == 2


def test_split_into_components_empty_geometry():
    geo = Geometry()
    components = split_into_components(geo)
    assert len(components) == 0


def test_split_into_components_single_contour():
    geo = Geometry()
    geo.move_to(0, 0)
    geo.line_to(10, 0)
    geo.line_to(10, 10)
    components = split_into_components(geo)
    assert len(components) == 1
    assert len(components[0].commands) == 3


def test_split_into_components_two_separate_shapes():
    geo = Geometry()
    # Shape 1
    geo.move_to(0, 0)
    geo.line_to(10, 0)
    geo.line_to(10, 10)
    geo.line_to(0, 10)
    geo.close_path()
    # Shape 2
    geo.move_to(20, 20)
    geo.line_to(30, 20)
    geo.line_to(30, 30)
    geo.line_to(20, 30)
    geo.close_path()

    components = split_into_components(geo)
    assert len(components) == 2
    assert len(components[0].commands) == 5
    assert len(components[1].commands) == 5


def test_split_into_components_containment_letter_o():
    geo = Geometry()
    # Outer circle (r=10, center=0,0)
    geo.move_to(10, 0)
    geo.arc_to(-10, 0, i=-10, j=0, clockwise=False)
    geo.arc_to(10, 0, i=10, j=0, clockwise=False)
    # Inner circle (r=5, center=0,0)
    geo.move_to(5, 0)
    geo.arc_to(-5, 0, i=-5, j=0, clockwise=False)
    geo.arc_to(5, 0, i=5, j=0, clockwise=False)

    components = split_into_components(geo)
    assert len(components) == 1
    assert len(components[0].commands) == 6  # 2 moves, 4 arcs
