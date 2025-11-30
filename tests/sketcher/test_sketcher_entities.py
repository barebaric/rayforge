import pytest
import math
from rayforge.core.sketcher.entities import (
    EntityRegistry,
    Point,
    Line,
    Arc,
    Circle,
)


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


def test_line_serialization_round_trip():
    """Tests the to_dict and from_dict methods for a single Line."""
    original_line = Line(id=10, p1_idx=1, p2_idx=2, construction=True)

    data = original_line.to_dict()
    assert data == {
        "id": 10,
        "type": "line",
        "construction": True,
        "p1_idx": 1,
        "p2_idx": 2,
    }

    new_line = Line.from_dict(data)
    assert isinstance(new_line, Line)
    assert new_line.id == original_line.id
    assert new_line.p1_idx == original_line.p1_idx
    assert new_line.p2_idx == original_line.p2_idx
    assert new_line.construction == original_line.construction


def test_arc_serialization_round_trip():
    """Tests the to_dict and from_dict methods for a single Arc."""
    original_arc = Arc(
        id=20,
        start_idx=3,
        end_idx=4,
        center_idx=5,
        clockwise=True,
        construction=True,
    )

    data = original_arc.to_dict()
    assert data == {
        "id": 20,
        "type": "arc",
        "construction": True,
        "start_idx": 3,
        "end_idx": 4,
        "center_idx": 5,
        "clockwise": True,
    }

    new_arc = Arc.from_dict(data)
    assert isinstance(new_arc, Arc)
    assert new_arc.id == original_arc.id
    assert new_arc.start_idx == original_arc.start_idx
    assert new_arc.end_idx == original_arc.end_idx
    assert new_arc.center_idx == original_arc.center_idx
    assert new_arc.clockwise == original_arc.clockwise
    assert new_arc.construction == original_arc.construction


def test_circle_serialization_round_trip():
    """Tests the to_dict and from_dict methods for a single Circle."""
    original_circle = Circle(
        id=30, center_idx=6, radius_pt_idx=7, construction=True
    )

    data = original_circle.to_dict()
    assert data == {
        "id": 30,
        "type": "circle",
        "construction": True,
        "center_idx": 6,
        "radius_pt_idx": 7,
    }

    new_circle = Circle.from_dict(data)
    assert isinstance(new_circle, Circle)
    assert new_circle.id == original_circle.id
    assert new_circle.center_idx == original_circle.center_idx
    assert new_circle.radius_pt_idx == original_circle.radius_pt_idx
    assert new_circle.construction == original_circle.construction


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


def test_arc_get_midpoint(registry):
    """Test calculation of the arc's midpoint."""
    center = registry.add_point(0, 0)
    start = registry.add_point(10, 0)
    end = registry.add_point(-10, 0)

    # Counter-clockwise arc (upper semi-circle)
    aid_ccw = registry.add_arc(start, end, center, cw=False)
    arc_ccw = registry.get_entity(aid_ccw)
    mid_ccw = arc_ccw.get_midpoint(registry)
    assert mid_ccw[0] == pytest.approx(0)
    assert mid_ccw[1] == pytest.approx(10)

    # Clockwise arc (lower semi-circle)
    aid_cw = registry.add_arc(start, end, center, cw=True)
    arc_cw = registry.get_entity(aid_cw)
    mid_cw = arc_cw.get_midpoint(registry)
    assert mid_cw[0] == pytest.approx(0)
    assert mid_cw[1] == pytest.approx(-10)


def test_arc_is_angle_within_sweep(registry):
    """Test checking if an angle is within the arc's sweep."""
    center = registry.add_point(0, 0)
    start = registry.add_point(10, 0)  # 0 degrees
    end = registry.add_point(0, 10)  # 90 degrees (pi/2)

    # Counter-clockwise from 0 to 90 degrees
    arc_ccw = registry.get_entity(
        registry.add_arc(start, end, center, cw=False)
    )

    assert (
        arc_ccw.is_angle_within_sweep(math.pi / 4, registry) is True
    )  # 45 deg
    assert arc_ccw.is_angle_within_sweep(math.pi, registry) is False  # 180 deg
    assert (
        arc_ccw.is_angle_within_sweep(0, registry) is True
    )  # on start boundary
    assert (
        arc_ccw.is_angle_within_sweep(math.pi / 2, registry) is True
    )  # on end boundary

    # Clockwise from 0 to 90 (sweep is the long way around)
    arc_cw = registry.get_entity(registry.add_arc(start, end, center, cw=True))

    assert (
        arc_cw.is_angle_within_sweep(math.pi / 4, registry) is False
    )  # 45 deg
    assert arc_cw.is_angle_within_sweep(math.pi, registry) is True  # 180 deg
    assert (
        arc_cw.is_angle_within_sweep(-math.pi / 2, registry) is True
    )  # -90 deg


def test_circle_get_midpoint(registry):
    """Test getting a point on the circle's circumference."""
    center = registry.add_point(5, 5)
    radius_pt_idx = registry.add_point(15, 5)

    cid = registry.add_circle(center, radius_pt_idx)
    circle = registry.get_entity(cid)

    midpoint = circle.get_midpoint(registry)
    radius_pt = registry.get_point(radius_pt_idx)

    assert midpoint is not None
    assert midpoint == radius_pt.pos()
    assert midpoint == (15.0, 5.0)


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


def test_line_update_constrained_status(registry):
    """Test Line.update_constrained_status logic."""
    p1 = registry.add_point(0, 0)
    p2 = registry.add_point(10, 10)
    lid = registry.add_line(p1, p2)
    line = registry.get_entity(lid)

    pt1 = registry.get_point(p1)
    pt2 = registry.get_point(p2)

    # Initially unconstrained
    pt1.constrained = False
    pt2.constrained = False
    line.update_constrained_status(registry, [])
    assert line.constrained is False

    # One point constrained
    pt1.constrained = True
    line.update_constrained_status(registry, [])
    assert line.constrained is False

    # Both points constrained
    pt2.constrained = True
    line.update_constrained_status(registry, [])
    assert line.constrained is True


def test_arc_update_constrained_status(registry):
    """Test Arc.update_constrained_status logic."""
    s = registry.add_point(10, 0)
    e = registry.add_point(0, 10)
    c = registry.add_point(0, 0)
    aid = registry.add_arc(s, e, c)
    arc = registry.get_entity(aid)

    pt_s = registry.get_point(s)
    pt_e = registry.get_point(e)
    pt_c = registry.get_point(c)

    # Initial state
    pt_s.constrained = False
    pt_e.constrained = False
    pt_c.constrained = False
    arc.update_constrained_status(registry, [])
    assert arc.constrained is False

    # Fully constrained points
    pt_s.constrained = True
    pt_e.constrained = True
    pt_c.constrained = True
    arc.update_constrained_status(registry, [])
    assert arc.constrained is True


def test_circle_update_constrained_status(registry):
    """
    Test Circle.update_constrained_status logic.
    Circle requires center point constrained AND radius defined.
    """
    c = registry.add_point(0, 0)
    r = registry.add_point(10, 0)
    cid = registry.add_circle(c, r)
    circle = registry.get_entity(cid)

    pt_c = registry.get_point(c)
    pt_r = registry.get_point(r)

    # Case 1: Nothing constrained
    pt_c.constrained = False
    pt_r.constrained = False
    circle.update_constrained_status(registry, [])
    assert circle.constrained is False

    # Case 2: Only Center constrained (Radius undefined)
    pt_c.constrained = True
    circle.update_constrained_status(registry, [])
    assert circle.constrained is False

    # Case 3: Center + Radius Point constrained (Fully defined)
    pt_r.constrained = True
    circle.update_constrained_status(registry, [])
    assert circle.constrained is True

    # Case 4: Center constrained
