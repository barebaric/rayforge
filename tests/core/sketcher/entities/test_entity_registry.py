import pytest
from rayforge.core.sketcher.entities import (
    Point,
    Line,
    Arc,
    Circle,
)
from rayforge.core.sketcher.registry import EntityRegistry


@pytest.fixture
def registry():
    return EntityRegistry()


def test_add_point(registry):
    pid = registry.add_point(10.0, 20.0, fixed=True)
    assert pid == 0
    pt = registry.get_point(pid)

    # Check instance type and attributes
    assert isinstance(pt, Point)
    assert pt.x == 10.0
    assert pt.y == 20.0
    assert pt.fixed is True
    assert pt.pos() == (10.0, 20.0)

    # Check defaults
    assert pt.constrained is False


def test_add_line(registry):
    p1 = registry.add_point(0, 0)
    p2 = registry.add_point(10, 10)

    # Test standard line
    lid = registry.add_line(p1, p2)
    assert lid == 2  # 0, 1 used by points

    line = registry.get_entity(lid)
    assert isinstance(line, Line)
    assert line.p1_idx == p1
    assert line.p2_idx == p2
    assert line.type == "line"
    assert line.construction is False


def test_add_construction_line(registry):
    """Test that explicit construction flag is passed to the class."""
    p1 = registry.add_point(0, 0)
    p2 = registry.add_point(10, 0)
    lid = registry.add_line(p1, p2, construction=True)

    line = registry.get_entity(lid)
    assert line.construction is True


def test_add_arc(registry):
    start = registry.add_point(0, 0)
    end = registry.add_point(10, 0)
    center = registry.add_point(5, 0)

    aid = registry.add_arc(start, end, center, cw=True)
    arc = registry.get_entity(aid)

    assert isinstance(arc, Arc)
    assert arc.start_idx == start
    assert arc.center_idx == center
    assert arc.clockwise is True
    assert arc.type == "arc"
    assert arc.construction is False


def test_add_circle(registry):
    center = registry.add_point(0, 0)
    radius_pt = registry.add_point(10, 0)
    cid = registry.add_circle(center, radius_pt, construction=True)
    assert cid == 2

    circle = registry.get_entity(cid)
    assert isinstance(circle, Circle)
    assert circle.center_idx == center
    assert circle.radius_pt_idx == radius_pt
    assert circle.construction is True
    assert circle.type == "circle"


def test_registry_indices(registry):
    # Ensure ID counter increments across types
    id1 = registry.add_point(0, 0)
    id2 = registry.add_line(id1, id1)
    id3 = registry.add_point(1, 1)
    assert id1 == 0
    assert id2 == 1
    assert id3 == 2


def test_registry_lookup_failures(registry):
    """Test behavior when looking up invalid IDs."""
    with pytest.raises(IndexError):
        registry.get_point(999)

    # get_entity returns None, doesn't raise
    assert registry.get_entity(999) is None


def test_entity_registry_serialization_round_trip():
    """Tests to_dict and from_dict for the entire EntityRegistry."""
    reg = EntityRegistry()
    p1 = reg.add_point(0, 0, fixed=True)
    p2 = reg.add_point(10, 0)
    p3 = reg.add_point(10, 10)
    p4 = reg.add_point(0, 10)
    _ = reg.add_line(p1, p2)
    l2 = reg.add_line(p2, p3, construction=True)
    arc = reg.add_arc(p3, p4, p1, cw=True)
    circ = reg.add_circle(p1, p2)

    data = reg.to_dict()

    # Basic structure check
    assert "points" in data
    assert "entities" in data
    assert "id_counter" in data
    assert len(data["points"]) == 4
    assert len(data["entities"]) == 4  # Line, Line, Arc, Circle
    assert data["id_counter"] == 8  # 4 points + 4 entities

    new_reg = EntityRegistry.from_dict(data)

    # Check integrity
    assert new_reg._id_counter == 8
    assert len(new_reg.points) == 4
    assert len(new_reg.entities) == 4

    # Check point details
    new_p1 = new_reg.get_point(p1)
    assert new_p1.x == 0
    assert new_p1.fixed is True

    # Check entity details and map
    new_l2 = new_reg.get_entity(l2)
    assert isinstance(new_l2, Line)
    assert new_l2.p1_idx == p2
    assert new_l2.construction is True

    new_arc = new_reg.get_entity(arc)
    assert isinstance(new_arc, Arc)
    assert new_arc.center_idx == p1
    assert new_arc.clockwise is True

    new_circ = new_reg.get_entity(circ)
    assert isinstance(new_circ, Circle)
    assert new_circ.center_idx == p1
    assert new_circ.radius_pt_idx == p2


def test_registry_is_point_used(registry):
    """Test checking if a point is referenced by any entity."""
    p1 = registry.add_point(0, 0)
    p2 = registry.add_point(10, 0)
    p3 = registry.add_point(10, 10)
    p4 = registry.add_point(0, 10)
    p_unused = registry.add_point(100, 100)

    # Initially, no points are used
    assert registry.is_point_used(p1) is False
    assert registry.is_point_used(p_unused) is False

    # Add a line and check
    registry.add_line(p1, p2)
    assert registry.is_point_used(p1) is True
    assert registry.is_point_used(p2) is True
    assert registry.is_point_used(p3) is False
    assert registry.is_point_used(p_unused) is False

    # Add an arc and check
    registry.add_arc(p2, p3, p4)
    assert registry.is_point_used(p2) is True
    assert registry.is_point_used(p3) is True
    assert registry.is_point_used(p4) is True

    # Add a circle and check
    registry.add_circle(p1, p4)
    assert registry.is_point_used(p1) is True
    assert registry.is_point_used(p4) is True
    assert registry.is_point_used(p_unused) is False
