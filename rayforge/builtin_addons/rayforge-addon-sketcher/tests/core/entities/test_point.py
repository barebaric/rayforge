import math

from sketcher.core.entities import Point
from sketcher.core.entities.bezier import Bezier
from sketcher.core.entities.point import WaypointType
from sketcher.core.registry import EntityRegistry


def test_point_instantiation():
    """Tests basic point creation and attribute access."""
    pt = Point(id=0, x=10.5, y=-20.0, fixed=True)
    assert pt.id == 0
    assert pt.x == 10.5
    assert pt.y == -20.0
    assert pt.fixed is True
    assert pt.pos() == (10.5, -20.0)
    assert pt.constrained is False


def test_point_serialization_round_trip():
    """Tests the to_dict and from_dict methods for a Point."""
    original_point = Point(id=1, x=1.2, y=3.4, fixed=True)
    data = original_point.to_dict()
    assert data == {"id": 1, "x": 1.2, "y": 3.4, "fixed": True}

    new_point = Point.from_dict(data)
    assert isinstance(new_point, Point)
    assert new_point.id == original_point.id
    assert new_point.x == original_point.x
    assert new_point.y == original_point.y
    assert new_point.fixed == original_point.fixed


def test_point_is_in_rect():
    """Tests the Point.is_in_rect method."""
    pt_inside = Point(0, 5, 5)
    pt_outside = Point(1, 15, 15)
    pt_on_edge = Point(2, 10, 5)
    rect = (0, 0, 10, 10)  # min_x, min_y, max_x, max_y

    assert pt_inside.is_in_rect(rect) is True
    assert pt_outside.is_in_rect(rect) is False
    assert pt_on_edge.is_in_rect(rect) is True  # Edges are inclusive


def test_point_get_connected_beziers():
    """Tests finding beziers connected to a point."""
    registry = EntityRegistry()
    p1 = registry.add_point(0, 0)
    p2 = registry.add_point(10, 0)
    p3 = registry.add_point(20, 0)

    b1_id = registry.add_bezier(p1, p2)
    b2_id = registry.add_bezier(p2, p3)

    pt1 = registry.get_point(p1)
    pt2 = registry.get_point(p2)
    pt3 = registry.get_point(p3)

    connected_to_p1 = pt1.get_connected_beziers(registry)
    assert len(connected_to_p1) == 1
    assert connected_to_p1[0].id == b1_id

    connected_to_p2 = pt2.get_connected_beziers(registry)
    assert len(connected_to_p2) == 2
    ids = {b.id for b in connected_to_p2}
    assert ids == {b1_id, b2_id}

    connected_to_p3 = pt3.get_connected_beziers(registry)
    assert len(connected_to_p3) == 1
    assert connected_to_p3[0].id == b2_id


def test_point_get_paired_beziers():
    """Tests getting paired beziers (first two connected)."""
    registry = EntityRegistry()
    p1 = registry.add_point(0, 0)
    p2 = registry.add_point(10, 0)
    p3 = registry.add_point(20, 0)

    b1_id = registry.add_bezier(p1, p2)
    b2_id = registry.add_bezier(p2, p3)

    pt2 = registry.get_point(p2)

    b1, b2 = pt2.get_paired_beziers(registry)
    assert b1 is not None
    assert b2 is not None
    assert b1.id == b1_id
    assert b2.id == b2_id


def test_point_paired_beziers_single():
    """Tests paired beziers when only one is connected."""
    registry = EntityRegistry()
    p1 = registry.add_point(0, 0)
    p2 = registry.add_point(10, 0)

    registry.add_bezier(p1, p2)

    pt1 = registry.get_point(p1)
    b1, b2 = pt1.get_paired_beziers(registry)
    assert b1 is not None
    assert b2 is None


def test_point_apply_constraint_symmetric():
    """Tests symmetric constraint: mirrored control points."""
    registry = EntityRegistry()
    p1 = registry.add_point(0, 0)
    p2 = registry.add_point(10, 10)
    p3 = registry.add_point(20, 0)

    b1_id = registry.add_bezier(p1, p2)
    b2_id = registry.add_bezier(p2, p3)

    b1 = registry.get_entity(b1_id)
    b2 = registry.get_entity(b2_id)
    assert isinstance(b1, Bezier)
    assert isinstance(b2, Bezier)

    pt2 = registry.get_point(p2)
    pt2.waypoint_type = WaypointType.SYMMETRIC

    b1.cp2 = (5.0, 3.0)

    pt2.apply_constraint(registry, b1, cp_index=2)

    assert b2.cp1 == (-5.0, -3.0)


def test_point_apply_constraint_smooth():
    """Tests smooth constraint: collinear with preserved length."""
    registry = EntityRegistry()
    p1 = registry.add_point(0, 0)
    p2 = registry.add_point(10, 10)
    p3 = registry.add_point(20, 0)

    b1_id = registry.add_bezier(p1, p2)
    b2_id = registry.add_bezier(p2, p3)

    b1 = registry.get_entity(b1_id)
    b2 = registry.get_entity(b2_id)
    assert isinstance(b1, Bezier)
    assert isinstance(b2, Bezier)

    pt2 = registry.get_point(p2)
    pt2.waypoint_type = WaypointType.SMOOTH

    b2.cp1 = (2.0, 4.0)
    b1.cp2 = (6.0, 8.0)

    pt2.apply_constraint(registry, b1, cp_index=2)

    assert b2.cp1 is not None

    expected_length = math.sqrt(2.0**2 + 4.0**2)
    actual_length = math.sqrt(b2.cp1[0] ** 2 + b2.cp1[1] ** 2)
    assert abs(actual_length - expected_length) < 0.001

    assert b2.cp1[0] < 0
    assert b2.cp1[1] < 0


def test_point_apply_constraint_sharp_noop():
    """Tests that SHARP waypoint type doesn't apply constraints."""
    registry = EntityRegistry()
    p1 = registry.add_point(0, 0)
    p2 = registry.add_point(10, 10)
    p3 = registry.add_point(20, 0)

    b1_id = registry.add_bezier(p1, p2)
    b2_id = registry.add_bezier(p2, p3)

    b1 = registry.get_entity(b1_id)
    b2 = registry.get_entity(b2_id)
    assert isinstance(b1, Bezier)
    assert isinstance(b2, Bezier)

    pt2 = registry.get_point(p2)
    pt2.waypoint_type = WaypointType.SHARP

    b1.cp2 = (5.0, 3.0)

    pt2.apply_constraint(registry, b1, cp_index=2)

    assert b2.cp1 is None
