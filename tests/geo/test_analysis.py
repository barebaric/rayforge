import pytest
import math
from rayforge.core.geo import Geometry
from rayforge.core.geo.analysis import (
    get_path_winding_order,
    get_point_and_tangent_at,
    get_outward_normal_at,
    get_angle_at_vertex,
    remove_duplicates,
    is_clockwise,
    arc_direction_is_clockwise,
    get_subpath_area,
    encloses,
)


@pytest.fixture
def ccw_square_geometry():
    geo = Geometry()
    geo.move_to(0, 0)  # cmd 0
    geo.line_to(10, 0)  # cmd 1: bottom
    geo.line_to(10, 10)  # cmd 2: right
    geo.line_to(0, 10)  # cmd 3: top
    geo.close_path()  # cmd 4: left (back to 0,0)
    return geo


@pytest.fixture
def cw_square_geometry():
    geo = Geometry()
    geo.move_to(0, 0)  # cmd 0
    geo.line_to(0, 10)  # cmd 1: left
    geo.line_to(10, 10)  # cmd 2: top
    geo.line_to(10, 0)  # cmd 3: right
    geo.close_path()  # cmd 4: bottom (back to 0,0)
    return geo


def test_get_subpath_area(ccw_square_geometry, cw_square_geometry):
    # Test CCW (positive area)
    # 10x10 square area = 100.0
    area_ccw = get_subpath_area(ccw_square_geometry.commands, 0)
    assert area_ccw == pytest.approx(100.0)

    # Test CW (negative area)
    area_cw = get_subpath_area(cw_square_geometry.commands, 0)
    assert area_cw == pytest.approx(-100.0)

    # Test open path
    open_geo = Geometry.from_points([(0, 0), (10, 10)], close=False)
    area_open = get_subpath_area(open_geo.commands, 0)
    assert area_open == 0.0

    # Test degenerate path (single point)
    point_geo = Geometry.from_points([(5, 5)], close=False)
    area_point = get_subpath_area(point_geo.commands, 0)
    assert area_point == 0.0


def test_get_winding_order(ccw_square_geometry, cw_square_geometry):
    # Test CCW
    assert get_path_winding_order(ccw_square_geometry.commands, 1) == "ccw"
    assert get_path_winding_order(ccw_square_geometry.commands, 3) == "ccw"

    # Test CW
    assert get_path_winding_order(cw_square_geometry.commands, 1) == "cw"
    assert get_path_winding_order(cw_square_geometry.commands, 3) == "cw"

    # Test open path
    open_geo = Geometry()
    open_geo.move_to(0, 0)
    open_geo.line_to(10, 10)
    assert get_path_winding_order(open_geo.commands, 1) == "unknown"


def test_get_point_and_tangent_at():
    geo = Geometry()
    geo.move_to(0, 0)
    geo.line_to(10, 0)  # cmd 1

    # Test horizontal line
    result = get_point_and_tangent_at(geo.commands, 1, 0.5)
    assert result is not None
    pt, tan = result
    assert pt == pytest.approx((5, 0))
    assert tan == pytest.approx((1, 0))

    geo.line_to(10, 10)  # cmd 2
    # Test vertical line
    result = get_point_and_tangent_at(geo.commands, 2, 0.25)
    assert result is not None
    pt, tan = result
    assert pt == pytest.approx((10, 2.5))
    assert tan == pytest.approx((0, 1))

    # Test arc (CCW 90 degree from (10,10) to (0,10))
    # Start: (10,10). Center offset: (-10,0). Center: (0,10). Radius: 10.
    geo.arc_to(0, 10, i=-10, j=0, clockwise=False)  # cmd 3
    # Start of arc
    result = get_point_and_tangent_at(geo.commands, 3, 0.0)
    assert result is not None
    pt, tan = result
    assert pt == pytest.approx((10, 10))
    assert tan == pytest.approx((0, 1))  # Tangent is vertical up

    # Midpoint of arc
    result = get_point_and_tangent_at(geo.commands, 3, 0.5)
    assert result is not None
    pt, tan = result
    # This arc is a spiral from (10,10) to its center (0,10), because the
    # end radius is 0. The angle is constant (0 rads).
    # At t=0.5, the radius is half the starting radius (5).
    # Point is (center_x + r*cos(0), center_y + r*sin(0))
    #       -> (0+5, 10+0) -> (5,10)
    assert pt == pytest.approx((5, 10))
    # Tangent for a spiral towards the center should be perpendicular to the
    # radius vector from the center. Radius vec is (5,0), so tangent is (0,5).
    assert tan == pytest.approx((0, 1))


def test_get_outward_normal_at(ccw_square_geometry, cw_square_geometry):
    # Test CCW square
    # Bottom edge, tangent (1,0) -> outward normal (0,-1)
    normal = get_outward_normal_at(ccw_square_geometry.commands, 1, 0.5)
    assert normal is not None
    assert normal == pytest.approx((0, -1))
    # Right edge, tangent (0,1) -> outward normal (1,0)
    normal = get_outward_normal_at(ccw_square_geometry.commands, 2, 0.5)
    assert normal is not None
    assert normal == pytest.approx((1, 0))
    # Top edge, tangent (-1,0) -> outward normal (0,1)
    normal = get_outward_normal_at(ccw_square_geometry.commands, 3, 0.5)
    assert normal is not None
    assert normal == pytest.approx((0, 1))
    # Left edge, tangent (0,-1) -> outward normal (-1,0)
    normal = get_outward_normal_at(ccw_square_geometry.commands, 4, 0.5)
    assert normal is not None
    assert normal == pytest.approx((-1, 0))

    # Test CW square
    # Left edge, tangent (0,1) -> outward normal (-1,0)
    normal = get_outward_normal_at(cw_square_geometry.commands, 1, 0.5)
    assert normal is not None
    assert normal == pytest.approx((-1, 0))
    # Bottom edge, tangent (-1,0) -> outward normal (0,-1)
    normal = get_outward_normal_at(cw_square_geometry.commands, 4, 0.5)
    assert normal is not None
    assert normal == pytest.approx((0, -1))


def test_get_angle_at_vertex():
    # 90 degree corner
    p0, p1, p2 = (0.0, 10.0), (0.0, 0.0), (10.0, 0.0)
    assert get_angle_at_vertex(p0, p1, p2) == pytest.approx(math.pi / 2)

    # Straight line (180 degrees)
    p0, p1, p2 = (-10.0, 0.0), (0.0, 0.0), (10.0, 0.0)
    assert get_angle_at_vertex(p0, p1, p2) == pytest.approx(math.pi)

    # 45 degree corner
    p0, p1, p2 = (0.0, 10.0), (0.0, 0.0), (10.0, 10.0)
    assert get_angle_at_vertex(p0, p1, p2) == pytest.approx(math.pi / 4)

    # Coincident points
    p0, p1, p2 = (10.0, 10.0), (0.0, 0.0), (0.0, 0.0)
    assert get_angle_at_vertex(p0, p1, p2) == pytest.approx(math.pi)


def test_remove_duplicates():
    points = [(1.0, 1.0), (1.0, 1.0), (2.0, 2.0), (2.0, 2.0)]
    assert remove_duplicates(points) == [(1.0, 1.0), (2.0, 2.0)]


def test_is_clockwise():
    # Clockwise points (right half-circle)
    points = [(0.0, 0.0, 0.0), (1.0, 1.0, 0.0), (2.0, 0.0, 0.0)]
    assert is_clockwise(points) is True

    # Counter-clockwise points (left half-circle)
    points = [(0.0, 0.0, 0.0), (-1.0, 1.0, 0.0), (-2.0, 0.0, 0.0)]
    assert is_clockwise(points) is False


def test_arc_direction_is_clockwise_half_circle():
    """Test a semicircle moving clockwise."""
    center = (0.0, 0.0)
    points = [
        (1.0, 0.0, 0.0),
        (0.0, -1.0, 0.0),
        (-1.0, 0.0, 0.0),
    ]
    assert arc_direction_is_clockwise(points, center) is True


def test_arc_direction_is_counter_clockwise_half_circle():
    """Test a semicircle moving counter-clockwise."""
    center = (0.0, 0.0)
    points = [
        (1.0, 0.0, 0.0),
        (0.0, 1.0, 0.0),
        (-1.0, 0.0, 0.0),
    ]
    assert arc_direction_is_clockwise(points, center) is False


def test_arc_direction_is_clockwise_full_circle():
    """Test a full clockwise circle."""
    center = (0.0, 0.0)
    points = [
        (1.0, 0.0, 0.0),
        (0.0, -1.0, 0.0),
        (-1.0, 0.0, 0.0),
        (0.0, 1.0, 0.0),
        (1.0, 0.0, 0.0),
    ]
    assert arc_direction_is_clockwise(points, center) is True


def test_arc_direction_is_counter_clockwise_full_circle():
    """Test a full counter-clockwise circle."""
    center = (0.0, 0.0)
    points = [
        (1.0, 0.0, 0.0),
        (0.0, 1.0, 0.0),
        (-1.0, 0.0, 0.0),
        (0.0, -1.0, 0.0),
        (1.0, 0.0, 0.0),
    ]
    assert arc_direction_is_clockwise(points, center) is False


def test_arc_direction_is_minimal_clockwise_arc():
    """Test a minimal 3-point clockwise arc."""
    center = (0.0, 0.0)
    points = [
        (2.0, 0.0, 0.0),
        (1.0, -1.0, 0.0),
        (0.0, 0.0, 0.0),
    ]
    assert arc_direction_is_clockwise(points, center) is True


def test_arc_direction_is_minimal_counter_clockwise_arc():
    """Test a minimal 3-point counter-clockwise arc."""
    center = (0.0, 0.0)
    points = [
        (2.0, 0.0, 0.0),
        (1.0, 1.0, 0.0),
        (0.0, 0.0, 0.0),
    ]
    assert arc_direction_is_clockwise(points, center) is False


def test_arc_direction_is_crossing_angle_discontinuity_counter_clockwise():
    """Test an arc crossing the π/-π discontinuity (counter-clockwise)."""
    center = (0.0, 0.0)
    points = [
        (1.0, 0.1, 0.0),
        (0.0, 1.0, 0.0),
        (-1.0, 0.1, 0.0),
    ]
    assert arc_direction_is_clockwise(points, center) is False


def test_arc_direction_is_small_radius_arc():
    """Test a small-radius clockwise arc."""
    center = (1.0, 1.0)
    points = [
        (1.1, 1.0, 0.0),
        (1.0, 0.9, 0.0),
        (0.9, 1.0, 0.0),
    ]
    assert arc_direction_is_clockwise(points, center) is True


def test_encloses_simple():
    """Test a simple case of one square enclosing another."""
    outer = Geometry.from_points([(0, 0), (10, 0), (10, 10), (0, 10)])
    inner = Geometry.from_points([(2, 2), (8, 2), (8, 8), (2, 8)])
    assert outer.encloses(inner) is True
    assert encloses(outer, inner) is True  # Also test direct function call
    assert inner.encloses(outer) is False


def test_encloses_separate():
    """Test non-enclosing, separate shapes."""
    geo1 = Geometry.from_points([(0, 0), (5, 0), (5, 5), (0, 5)])
    geo2 = Geometry.from_points([(10, 10), (15, 10), (15, 15), (10, 15)])
    assert geo1.encloses(geo2) is False
    assert geo2.encloses(geo1) is False


def test_encloses_intersecting():
    """Test intersecting shapes do not enclose."""
    geo1 = Geometry.from_points([(0, 0), (10, 0), (10, 10), (0, 10)])
    geo2 = Geometry.from_points([(5, 5), (15, 5), (15, 15), (5, 15)])
    assert geo1.encloses(geo2) is False
    assert geo2.encloses(geo1) is False


def test_encloses_touching():
    """Test touching shapes do not enclose."""
    geo1 = Geometry.from_points([(0, 0), (10, 0), (10, 10), (0, 10)])
    geo2 = Geometry.from_points([(10, 0), (20, 0), (20, 10), (10, 10)])
    assert geo1.encloses(geo2) is False
    assert geo2.encloses(geo1) is False


def test_encloses_with_hole():
    """Test enclosure in a shape with a hole."""
    # Outer CCW rect
    outer = Geometry.from_points([(0, 0), (20, 0), (20, 20), (0, 20)])
    # Inner CW rect (the hole)
    hole = Geometry.from_points([(5, 5), (5, 15), (15, 15), (15, 5)])

    donut = outer.copy()
    donut.commands.extend(hole.commands)

    # Shape fully inside the donut's material
    content_inside = Geometry.from_points([(1, 1), (4, 1), (4, 4), (1, 4)])
    assert donut.encloses(content_inside) is True

    # Shape fully inside the donut's hole
    content_in_hole = Geometry.from_points(
        [(7, 7), (13, 7), (13, 13), (7, 13)]
    )
    assert donut.encloses(content_in_hole) is False


def test_encloses_bbox_contained_but_path_outside():
    """
    Test a C-shape where the bbox contains the other shape, but path does not.
    """
    c_shape = Geometry.from_points(
        [
            (0, 0),
            (10, 0),
            (10, 1),
            (1, 1),
            (1, 9),
            (10, 9),
            (10, 10),
            (0, 10),
        ],
        close=True,
    )
    other = Geometry.from_points([(2, 4), (5, 4), (5, 6), (2, 6)])
    assert c_shape.encloses(other) is False
