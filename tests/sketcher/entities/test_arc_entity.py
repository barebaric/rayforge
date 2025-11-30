import pytest
import math
from rayforge.core.sketcher.entities import EntityRegistry, Arc


@pytest.fixture
def registry():
    return EntityRegistry()


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


def test_arc_get_point_ids(registry):
    """Tests that an arc correctly reports its defining point IDs."""
    p1 = registry.add_point(0, 0)
    p2 = registry.add_point(10, 0)
    p3 = registry.add_point(5, 5)
    arc = registry.get_entity(registry.add_arc(p1, p2, p3))
    assert set(arc.get_point_ids()) == {p1, p2, p3}


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


@pytest.fixture
def selection_setup(registry):
    """Fixture for setting up arc entities for selection tests."""
    rect = (20, 20, 80, 80)

    # Arc fully inside (semi-circle with radius 10)
    s_arc_in = registry.add_point(40, 50)
    e_arc_in = registry.add_point(60, 50)
    c_arc_in = registry.add_point(50, 50)
    arc_in = registry.get_entity(
        registry.add_arc(s_arc_in, e_arc_in, c_arc_in)
    )

    # Arc intersecting
    s_arc_cross = registry.add_point(70, 50)
    e_arc_cross = registry.add_point(90, 50)
    c_arc_cross = registry.add_point(80, 50)
    arc_cross = registry.get_entity(
        registry.add_arc(s_arc_cross, e_arc_cross, c_arc_cross)
    )

    # Arc outside
    s_arc_out = registry.add_point(0, 0)
    e_arc_out = registry.add_point(10, 0)
    c_arc_out = registry.add_point(5, 0)
    arc_out = registry.get_entity(
        registry.add_arc(s_arc_out, e_arc_out, c_arc_out)
    )

    return (
        registry,
        rect,
        {
            "arc_in": arc_in,
            "arc_cross": arc_cross,
            "arc_out": arc_out,
        },
    )


def test_arc_is_contained_by(selection_setup):
    """Test the is_contained_by method for Arc entities."""
    registry, rect, entities = selection_setup
    assert entities["arc_in"].is_contained_by(rect, registry) is True
    assert entities["arc_cross"].is_contained_by(rect, registry) is False
    assert entities["arc_out"].is_contained_by(rect, registry) is False


def test_arc_intersects_rect(selection_setup):
    """Test the intersects_rect method for Arc entities."""
    registry, rect, entities = selection_setup
    assert entities["arc_in"].intersects_rect(rect, registry) is True
    assert entities["arc_cross"].intersects_rect(rect, registry) is True
    assert entities["arc_out"].intersects_rect(rect, registry) is False
