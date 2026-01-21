import pytest
from rayforge.core.geo import Geometry
from rayforge.core.geo.analysis import get_subpath_area_from_array
from rayforge.core.geo.contours import (
    filter_to_external_contours,
    reverse_contour,
    normalize_winding_orders,
    split_inner_and_outer_contours,
    get_valid_contours_data,
)


def test_reverse_contour_simple_polygon():
    """Tests reversing a simple square."""
    ccw_square = Geometry.from_points([(0, 0), (10, 0), (10, 10), (0, 10)])
    assert ccw_square.data is not None
    original_area = get_subpath_area_from_array(ccw_square.data, 0)
    assert original_area > 0  # Is CCW

    reversed_square = reverse_contour(ccw_square)
    assert reversed_square.data is not None
    reversed_area = get_subpath_area_from_array(reversed_square.data, 0)
    assert reversed_area < 0  # Is now CW
    assert reversed_area == pytest.approx(-original_area)


def test_reverse_contour_with_arc():
    """Tests reversing a path that includes an arc command."""
    # Semicircle from (10,0) to (-10,0) with center (0,0)
    semi = Geometry()
    semi.move_to(10, 0)
    semi.arc_to(-10, 0, i=-10, j=0, clockwise=False)  # CCW
    semi.line_to(10, 0)
    assert semi.data is not None
    assert get_subpath_area_from_array(semi.data, 0) > 0

    reversed_semi = reverse_contour(semi)
    assert reversed_semi.data is not None
    assert get_subpath_area_from_array(reversed_semi.data, 0) < 0


def test_split_inner_and_outer_contours_empty_and_single():
    """Tests splitting with empty or single-item lists."""
    # Empty list
    internal, external = split_inner_and_outer_contours([])
    assert internal == []
    assert external == []

    # Single item (is always external)
    c1 = Geometry.from_points([(0, 0), (1, 0), (0, 1)])
    internal, external = split_inner_and_outer_contours([c1])
    assert internal == []
    assert external == [c1]


def test_split_inner_and_outer_contours_simple_donut():
    """Tests splitting a simple solid-and-hole shape."""
    outer = Geometry.from_points([(0, 0), (10, 0), (10, 10), (0, 10)])
    hole = Geometry.from_points([(2, 2), (8, 2), (8, 8), (2, 8)])

    # Test with standard order
    contours = [outer, hole]
    internal, external = split_inner_and_outer_contours(contours)
    assert internal == [hole]
    assert external == [outer]

    # Test with reversed input order
    contours_rev = [hole, outer]
    internal_rev, external_rev = split_inner_and_outer_contours(contours_rev)
    assert internal_rev == [hole]
    assert external_rev == [outer]


def test_split_inner_and_outer_contours_bullseye():
    """
    Tests splitting a multi-level nesting. The key is that the middle
    contour is a hole, while the inner and outer are solids.
    """
    c1_outer = Geometry.from_points([(0, 0), (30, 0), (30, 30), (0, 30)])
    c2_hole = Geometry.from_points([(5, 5), (25, 5), (25, 25), (5, 25)])
    c3_inner = Geometry.from_points([(10, 10), (20, 10), (20, 20), (10, 20)])

    # Solids: c1_outer, c3_inner. Hole: c2_hole.
    contours = [c1_outer, c2_hole, c3_inner]

    internal, external = split_inner_and_outer_contours(contours)

    assert len(internal) == 1
    assert internal[0] is c2_hole
    assert len(external) == 2
    assert set(external) == {c1_outer, c3_inner}


def test_split_inner_and_outer_contours_two_letter_b_shapes():
    """
    Tests that splitting correctly performs a global partition of solids and
    holes.
    """
    # Component 1: A "B" shape
    b1_outer = Geometry.from_points([(0, 0), (10, 0), (10, 20), (0, 20)])
    b1_hole_top = Geometry.from_points([(2, 12), (8, 12), (8, 18), (2, 18)])
    b1_hole_bottom = Geometry.from_points([(2, 2), (8, 2), (8, 8), (2, 8)])

    # Component 2: Another "B" shape, shifted
    b2_outer = Geometry.from_points([(100, 0), (110, 0), (110, 20), (100, 20)])
    b2_hole_top = Geometry.from_points(
        [(102, 12), (108, 12), (108, 18), (102, 18)]
    )
    b2_hole_bottom = Geometry.from_points(
        [(102, 2), (108, 2), (108, 8), (102, 2)]
    )

    all_solids = {b1_outer, b2_outer}
    all_holes = {b1_hole_top, b1_hole_bottom, b2_hole_top, b2_hole_bottom}

    # Unordered list containing all 6 contours
    contours = list(all_solids | all_holes)

    internal, external = split_inner_and_outer_contours(contours)
    assert len(internal) == 4
    assert len(external) == 2
    assert set(internal) == all_holes
    assert set(external) == all_solids


def test_normalize_winding_donut_all_ccw():
    """Tests a donut where both contours are incorrectly CCW."""
    outer = Geometry.from_points([(0, 0), (20, 0), (20, 20), (0, 20)])
    hole = Geometry.from_points([(5, 5), (15, 5), (15, 15), (5, 15)])
    assert outer.data is not None
    assert hole.data is not None
    assert get_subpath_area_from_array(outer.data, 0) > 0
    assert get_subpath_area_from_array(hole.data, 0) > 0

    normalized = normalize_winding_orders([outer, hole])
    # Outer (nesting 0) should remain CCW
    assert normalized[0].data is not None
    assert get_subpath_area_from_array(normalized[0].data, 0) > 0
    # Inner (nesting 1) should be flipped to CW
    assert normalized[1].data is not None
    assert get_subpath_area_from_array(normalized[1].data, 0) < 0


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

    assert outer_cw.data is not None
    assert get_subpath_area_from_array(outer_cw.data, 0) < 0  # Verify it's CW

    # The buggy `normalize_winding_orders` would fail here.
    # `outer_cw.encloses(hole_ccw)` would return False because outer_cw is CW.
    # Therefore, it would think the hole isn't nested and would not flip it.
    normalized = normalize_winding_orders([outer_cw, hole_ccw])

    # This assertion would fail: the hole would not have been flipped to CW.
    assert normalized[1].data is not None
    assert get_subpath_area_from_array(normalized[1].data, 0) < 0


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
    assert result[0] is outer


def test_filter_external_bullseye_nesting():
    """Tests three nested contours. Outer and inner-most should be returned."""
    c1 = Geometry.from_points([(0, 0), (30, 0), (30, 30), (0, 30)])  # Outer
    c2 = Geometry.from_points([(5, 5), (25, 5), (25, 25), (5, 25)])  # Middle
    c3 = Geometry.from_points([(10, 10), (20, 10), (20, 20), (10, 20)])
    contours = [c1, c2, c3]
    result = filter_to_external_contours(contours)
    assert len(result) == 2
    assert c1 in result
    assert c3 in result
    assert c2 not in result


def test_filter_external_robust_to_winding_order():
    """
    Tests that the filter works correctly even if the input winding order
    is wrong (e.g., a hole is wound CCW).
    """
    # Donut shape, but the "hole" is wound CCW, which is incorrect.
    outer = Geometry.from_points([(0, 0), (20, 0), (20, 20), (0, 20)])
    incorrect_hole = Geometry.from_points([(5, 5), (15, 5), (15, 15), (5, 15)])
    assert outer.data is not None
    assert incorrect_hole.data is not None
    assert get_subpath_area_from_array(outer.data, 0) > 0
    assert get_subpath_area_from_array(incorrect_hole.data, 0) > 0

    # A correct filter should normalize the hole to CW and then discard it.
    result = filter_to_external_contours([outer, incorrect_hole])
    assert len(result) == 1
    assert result[0] is outer


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
    geo_donut.extend(hole)
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
    geo_bullseye.extend(c1)
    geo_bullseye.extend(c2_hole)
    geo_bullseye.extend(c3)

    # Total area = (30*30) - (20*20) + (10*10) = 900 - 400 + 100 = 600
    assert geo_bullseye.area() == pytest.approx(600)

    result_bullseye = geo_bullseye.remove_inner_edges()
    # The result should contain the outer and inner-most solids.
    # The area method sums the individual areas of the contours.
    # Expected area = area(c1) + area(c3) = 900 + 100 = 1000
    assert result_bullseye.area() == pytest.approx(1000)
    contours_bullseye = result_bullseye.split_into_contours()
    assert len(contours_bullseye) == 2


def test_get_valid_contours_data_empty_list():
    """Tests that an empty list returns an empty result."""
    result = get_valid_contours_data([])
    assert result == []


def test_get_valid_contours_data_filters_empty_geometry():
    """Tests that empty geometries are filtered out."""
    empty_geo = Geometry()
    result = get_valid_contours_data([empty_geo])
    assert result == []


def test_get_valid_contours_data_filters_open_contour():
    """Tests that open contours are filtered out."""
    open_contour = Geometry()
    open_contour.move_to(0, 0)
    open_contour.line_to(10, 0)
    open_contour.line_to(10, 10)

    result = get_valid_contours_data([open_contour])
    assert result == []


def test_get_valid_contours_data_filters_small_bbox():
    """Tests that contours with very small bbox area are filtered out."""
    tiny_contour = Geometry()
    tiny_contour.move_to(0, 0)
    tiny_contour.line_to(1e-10, 0)
    tiny_contour.line_to(1e-10, 1e-10)
    tiny_contour.line_to(0, 1e-10)

    result = get_valid_contours_data([tiny_contour])
    assert result == []


def test_get_valid_contours_data_filters_no_move_to():
    """Tests that contours not starting with MoveTo are filtered out."""
    geo = Geometry()
    geo.line_to(0, 0)
    geo.line_to(10, 0)
    geo.line_to(10, 10)
    geo.line_to(0, 10)

    result = get_valid_contours_data([geo])
    assert result == []


def test_get_valid_contours_data_valid_closed_contour():
    """Tests that a valid closed contour is included."""
    contour = Geometry.from_points([(0, 0), (10, 0), (10, 10), (0, 10)])

    result = get_valid_contours_data([contour])

    assert len(result) == 1
    assert result[0]["geo"] is contour
    assert result[0]["is_closed"] is True
    assert result[0]["original_index"] == 0
    assert len(result[0]["vertices"]) == 5


def test_get_valid_contours_data_multiple_valid_contours():
    """Tests that multiple valid contours are all included."""
    c1 = Geometry.from_points([(0, 0), (5, 0), (5, 5), (0, 5)])
    c2 = Geometry.from_points([(10, 10), (15, 10), (15, 15), (10, 15)])

    result = get_valid_contours_data([c1, c2])

    assert len(result) == 2
    assert result[0]["geo"] is c1
    assert result[1]["geo"] is c2
    assert result[0]["original_index"] == 0
    assert result[1]["original_index"] == 1


def test_get_valid_contours_data_mixed_valid_invalid():
    """Tests filtering with a mix of valid and invalid contours."""
    valid = Geometry.from_points([(0, 0), (10, 0), (10, 10), (0, 10)])
    empty = Geometry()
    open_contour = Geometry()
    open_contour.move_to(20, 20)
    open_contour.line_to(30, 20)
    open_contour.line_to(30, 30)

    result = get_valid_contours_data([empty, valid, open_contour])

    assert len(result) == 1
    assert result[0]["geo"] is valid
    assert result[0]["original_index"] == 1


def test_get_valid_contours_data_preserves_indices():
    """Tests that original_index is preserved correctly."""
    c1 = Geometry.from_points([(0, 0), (5, 0), (5, 5), (0, 5)])
    empty = Geometry()
    c2 = Geometry.from_points([(10, 10), (15, 10), (15, 15), (10, 15)])
    open_contour = Geometry()
    open_contour.move_to(20, 20)
    open_contour.line_to(30, 20)

    result = get_valid_contours_data([c1, empty, c2, open_contour])

    assert len(result) == 2
    assert result[0]["original_index"] == 0
    assert result[1]["original_index"] == 2


def test_get_valid_contours_data_vertices_extraction():
    """Tests that vertices are correctly extracted from contours."""
    contour = Geometry.from_points([(0, 0), (10, 0), (10, 10), (0, 10)])

    result = get_valid_contours_data([contour])

    assert len(result) == 1
    vertices = result[0]["vertices"]
    assert len(vertices) == 5
    assert vertices[0] == pytest.approx((0.0, 0.0))
    assert vertices[1] == pytest.approx((10.0, 0.0))
    assert vertices[2] == pytest.approx((10.0, 10.0))
    assert vertices[3] == pytest.approx((0.0, 10.0))
    assert vertices[4] == pytest.approx((0.0, 0.0))
