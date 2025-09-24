import pytest
from rayforge.core.geo import Geometry
from rayforge.core.geo.analysis import get_subpath_area
from rayforge.core.geo.contours import (
    split_into_contours,
    filter_to_external_contours,
    reverse_contour,
    normalize_winding_orders,
)


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


def test_reverse_contour_simple_polygon():
    """Tests reversing a simple square."""
    ccw_square = Geometry.from_points([(0, 0), (10, 0), (10, 10), (0, 10)])
    original_area = get_subpath_area(ccw_square.commands, 0)
    assert original_area > 0  # Is CCW

    reversed_square = reverse_contour(ccw_square)
    reversed_area = get_subpath_area(reversed_square.commands, 0)
    assert reversed_area < 0  # Is now CW
    assert reversed_area == pytest.approx(-original_area)


def test_reverse_contour_with_arc():
    """Tests reversing a path that includes an arc command."""
    # Semicircle from (10,0) to (-10,0) with center (0,0)
    semi = Geometry()
    semi.move_to(10, 0)
    semi.arc_to(-10, 0, i=-10, j=0, clockwise=False)  # CCW
    semi.line_to(10, 0)
    assert get_subpath_area(semi.commands, 0) > 0

    reversed_semi = reverse_contour(semi)
    assert get_subpath_area(reversed_semi.commands, 0) < 0


def test_normalize_winding_donut_all_ccw():
    """Tests a donut where both contours are incorrectly CCW."""
    outer = Geometry.from_points([(0, 0), (20, 0), (20, 20), (0, 20)])
    hole = Geometry.from_points([(5, 5), (15, 5), (15, 15), (5, 15)])
    assert get_subpath_area(outer.commands, 0) > 0
    assert get_subpath_area(hole.commands, 0) > 0

    normalized = normalize_winding_orders([outer, hole])
    # Outer (nesting 0) should remain CCW
    assert get_subpath_area(normalized[0].commands, 0) > 0
    # Inner (nesting 1) should be flipped to CW
    assert get_subpath_area(normalized[1].commands, 0) < 0


def test_normalize_winding_with_incorrect_container():
    """
    This test would have failed the old implementation.
    The container ('outer') is wound CW, which is incorrect.
    """
    # Create a CW (incorrect) outer shape
    outer_cw = reverse_contour(
        Geometry.from_points([(0, 0), (20, 0), (20, 20), (0, 20)])
    )
    # Create a hole (can be any direction, let's use CCW)
    hole_ccw = Geometry.from_points([(5, 5), (15, 5), (15, 15), (5, 15)])

    assert get_subpath_area(outer_cw.commands, 0) < 0  # Verify it's CW

    # The buggy `normalize_winding_orders` would fail here.
    # `outer_cw.encloses(hole_ccw)` would return False because outer_cw is CW.
    # Therefore, it would think the hole isn't nested and would not flip it.
    normalized = normalize_winding_orders([outer_cw, hole_ccw])

    # This assertion would fail: the hole would not have been flipped to CW.
    assert get_subpath_area(normalized[1].commands, 0) < 0


def test_filter_external_empty_list():
    """Tests filtering an empty list of contours."""
    assert filter_to_external_contours([]) == []


def test_filter_external_single_contour():
    """Tests a single contour, which should always be external."""
    contour = Geometry.from_points([(0, 0), (10, 0), (10, 10), (0, 10)])
    result = filter_to_external_contours([contour])
    assert len(result) == 1
    assert result[0] is contour


def test_filter_external_shape_with_hole():
    """Tests a donut shape; only the outer contour should be returned."""
    outer = Geometry.from_points([(0, 0), (20, 0), (20, 20), (0, 20)])
    hole = Geometry.from_points([(5, 5), (15, 5), (15, 15), (5, 15)])
    contours = [outer, hole]
    result = filter_to_external_contours(contours)
    assert len(result) == 1
    assert get_subpath_area(result[0].commands, 0) > 0


def test_filter_external_bullseye_nesting():
    """Tests three nested contours. Outer and inner-most should be returned."""
    c1 = Geometry.from_points([(0, 0), (30, 0), (30, 30), (0, 30)])  # Outer
    c2 = Geometry.from_points([(5, 5), (25, 5), (25, 25), (5, 25)])  # Middle
    c3 = Geometry.from_points([(10, 10), (20, 10), (20, 20), (10, 20)])
    contours = [c1, c2, c3]
    result = filter_to_external_contours(contours)
    assert len(result) == 2
    areas = [get_subpath_area(r.commands, 0) for r in result]
    # Both resulting areas should be positive (CCW) after normalization.
    assert all(a > 0 for a in areas)


def test_filter_external_robust_to_winding_order():
    """
    Tests that the filter works correctly even if the input winding order
    is wrong (e.g., a hole is wound CCW).
    """
    # Donut shape, but the "hole" is wound CCW, which is incorrect.
    outer = Geometry.from_points([(0, 0), (20, 0), (20, 20), (0, 20)])
    incorrect_hole = Geometry.from_points([(5, 5), (15, 5), (15, 15), (5, 15)])
    assert get_subpath_area(outer.commands, 0) > 0
    assert get_subpath_area(incorrect_hole.commands, 0) > 0

    # A correct filter should normalize the hole to CW and then discard it.
    result = filter_to_external_contours([outer, incorrect_hole])
    assert len(result) == 1
    # Check that the returned geometry is equivalent to the original outer one.
    assert result[0].area() == pytest.approx(outer.area())


def test_filter_external_two_separate_shapes():
    """Tests two separate, non-overlapping shapes. Both should be returned."""
    s1 = Geometry.from_points([(0, 0), (5, 0), (5, 5), (0, 5)])
    s2 = Geometry.from_points([(10, 10), (15, 10), (15, 15), (10, 15)])
    contours = [s1, s2]
    result = filter_to_external_contours(contours)
    assert len(result) == 2
    assert s1 in result
    assert s2 in result


def test_filter_external_shape_inside_another_hole():
    """Tests a shape that is inside the hole of another shape."""
    # This is topologically identical to the bullseye test.
    c1_outer_boundary = Geometry.from_points(
        [(0, 0), (30, 0), (30, 30), (0, 30)]
    )
    c2_hole_boundary = Geometry.from_points(
        [(5, 5), (25, 5), (25, 25), (5, 25)]
    )
    c3_island = Geometry.from_points([(10, 10), (20, 10), (20, 20), (10, 20)])
    contours = [c1_outer_boundary, c2_hole_boundary, c3_island]
    result = filter_to_external_contours(contours)

    assert len(result) == 2
    assert c1_outer_boundary in result
    assert c3_island in result
    assert c2_hole_boundary not in result
