import pytest
from rayforge.core.geo import Geometry
from rayforge.core.geo.analysis import (
    get_path_winding_order,
    get_point_and_tangent_at,
    get_outward_normal_at,
)


@pytest.fixture
def ccw_square_geometry():
    geo = Geometry()
    geo.move_to(0, 0)  # cmd 0
    geo.line_to(10, 0)  # cmd 1: bottom
    geo.line_to(10, 10)  # cmd 2: right
    geo.line_to(0, 10)  # cmd 3: top
    geo.close_path()  # cmd 4: left
    return geo


@pytest.fixture
def cw_square_geometry():
    geo = Geometry()
    geo.move_to(0, 0)  # cmd 0
    geo.line_to(0, 10)  # cmd 1: left
    geo.line_to(10, 10)  # cmd 2: top
    geo.line_to(10, 0)  # cmd 3: right
    geo.close_path()  # cmd 4: bottom
    return geo


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
