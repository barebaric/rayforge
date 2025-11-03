import pytest
from rayforge.core.geo import Geometry, MoveToCommand, LineToCommand
from rayforge.core.geo.analysis import get_subpath_area
from rayforge.core.geo.contours import (
    split_into_contours,
    filter_to_external_contours,
    reverse_contour,
    normalize_winding_orders,
    close_geometry_gaps,
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


def test_close_geometry_gaps_functional():
    """Tests the core logic of the close_geometry_gaps function."""
    # 1. Test intra-contour gap closing (almost closed path)
    geo_intra = Geometry()
    geo_intra.move_to(0, 0)
    geo_intra.line_to(10, 0)
    geo_intra.line_to(10, 10)
    geo_intra.line_to(0.000001, 10)  # Ends near (0, 10)
    geo_intra.line_to(0.000002, 0.000003)  # Ends near start point (0, 0)
    original_intra_commands = geo_intra.copy().commands

    result_intra = close_geometry_gaps(geo_intra, tolerance=1e-5)
    assert result_intra is not geo_intra
    assert result_intra.commands is not geo_intra.commands
    assert (
        geo_intra.commands[-1].end == original_intra_commands[-1].end
    )  # Original is unchanged
    assert result_intra.commands[0].end == (0, 0, 0)
    # The final point should be snapped to the start point
    assert result_intra.commands[-1].end == (0, 0, 0)

    # 2. Test inter-contour gap closing (stitching paths)
    geo_inter = Geometry()
    geo_inter.move_to(0, 0)
    geo_inter.line_to(10, 10)
    geo_inter.move_to(10.000001, 10.000002)  # A small jump
    geo_inter.line_to(20, 20)
    original_inter_commands = geo_inter.copy().commands

    result_inter = close_geometry_gaps(geo_inter, tolerance=1e-5)
    assert result_inter is not geo_inter
    assert result_inter.commands is not geo_inter.commands
    # Assert original is unchanged by comparing its command to the saved copy
    assert isinstance(geo_inter.commands[2], MoveToCommand)
    assert geo_inter.commands[2].end == original_inter_commands[2].end

    # The MoveTo should be replaced with a LineTo in the result
    assert isinstance(result_inter.commands[2], LineToCommand)
    # The new LineTo should connect to the exact previous end point
    assert result_inter.commands[2].end == (10, 10, 0)


def test_remove_inner_edges():
    """
    Tests the remove_inner_edges function and the Geometry.remove_inner_edges
    method.
    """
    # Test Case 1: Empty Geometry
    geo_empty = Geometry()
    result_empty = geo_empty.remove_inner_edges()
    assert result_empty.is_empty()
    assert result_empty is not geo_empty, "Should return a new object"

    # Test Case 2: Geometry with only an open path
    geo_open = Geometry()
    geo_open.move_to(50, 50)
    geo_open.line_to(60, 60)
    result_open = geo_open.remove_inner_edges()
    contours_open = result_open.split_into_contours()
    assert len(contours_open) == 1
    assert not contours_open[0].is_closed()

    # Test Case 3: Geometry with only a single closed path
    geo_closed = Geometry.from_points([(0, 0), (1, 0), (1, 1), (0, 1)])
    result_closed = geo_closed.remove_inner_edges()
    assert result_closed.area() == pytest.approx(1.0)
    assert len(result_closed.split_into_contours()) == 1

    # Test Case 4: Donut shape (one outer, one inner closed path)
    geo_donut = Geometry.from_points([(0, 0), (10, 0), (10, 10), (0, 10)])
    hole = Geometry.from_points([(2, 2), (2, 8), (8, 8), (8, 2)])
    geo_donut.commands.extend(hole.commands)
    assert geo_donut.area() == pytest.approx(100 - 36)  # Area = 64

    result_donut = geo_donut.remove_inner_edges()
    # The result should only contain the outer shape's area
    assert result_donut.area() == pytest.approx(100)
    assert len(result_donut.split_into_contours()) == 1

    # Test Case 5: Mix of open and closed paths
    geo_mix = geo_donut.copy()  # Start with the donut
    # Add an open line segment outside the donut
    geo_mix.move_to(20, 20)
    geo_mix.line_to(30, 30)
    # Add another open line segment inside the donut's hole
    geo_mix.move_to(4, 4)
    geo_mix.line_to(6, 6)

    result_mix = geo_mix.remove_inner_edges()

    # The area should still be just the outer square's area
    assert result_mix.area() == pytest.approx(100)

    # Check the contours: should be 1 closed path and 2 open paths
    contours_mix = result_mix.split_into_contours()
    assert len(contours_mix) == 3

    closed_count = sum(1 for c in contours_mix if c.is_closed())
    open_count = sum(1 for c in contours_mix if not c.is_closed())

    assert closed_count == 1
    assert open_count == 2

    # Test Case 6: Bullseye shape (3 nested closed paths)
    c1 = Geometry.from_points([(0, 0), (30, 0), (30, 30), (0, 30)])  # Outer
    c2_ccw = Geometry.from_points(
        [(5, 5), (25, 5), (25, 25), (5, 25)]
    )  # Middle hole
    # Reverse the middle contour to make it a proper hole (CW)
    c2_hole = reverse_contour(c2_ccw)
    c3 = Geometry.from_points(
        [(10, 10), (20, 10), (20, 20), (10, 20)]
    )  # Inner
    geo_bullseye = Geometry()
    geo_bullseye.commands.extend(c1.commands)
    geo_bullseye.commands.extend(c2_hole.commands)
    geo_bullseye.commands.extend(c3.commands)
    # Total area = (30*30) - (20*20) + (10*10) = 900 - 400 + 100 = 600
    assert geo_bullseye.area() == pytest.approx(600)

    result_bullseye = geo_bullseye.remove_inner_edges()
    # The result should contain the outer and inner-most solids.
    # The area method sums the individual areas of the contours.
    # Expected area = area(c1) + area(c3) = 900 + 100 = 1000
    assert result_bullseye.area() == pytest.approx(1000)
    contours_bullseye = result_bullseye.split_into_contours()
    assert len(contours_bullseye) == 2
