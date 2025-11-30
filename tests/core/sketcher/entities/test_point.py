from rayforge.core.sketcher.entities import Point


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
