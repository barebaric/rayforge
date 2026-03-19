import pytest
from rayforge.core.sketcher.entities import Bezier
from rayforge.core.sketcher.registry import EntityRegistry
from rayforge.core.geo.geometry import Geometry


@pytest.fixture
def registry():
    return EntityRegistry()


def test_bezier_serialization_round_trip():
    """Tests the to_dict and from_dict methods for a single Bezier."""
    original_bezier = Bezier(
        id=10,
        start_idx=1,
        cp1_idx=2,
        cp2_idx=3,
        end_idx=4,
        construction=True,
    )

    data = original_bezier.to_dict()
    assert data == {
        "id": 10,
        "type": "bezier",
        "construction": True,
        "start_idx": 1,
        "cp1_idx": 2,
        "cp2_idx": 3,
        "end_idx": 4,
    }

    new_bezier = Bezier.from_dict(data)
    assert isinstance(new_bezier, Bezier)
    assert new_bezier.id == original_bezier.id
    assert new_bezier.start_idx == original_bezier.start_idx
    assert new_bezier.cp1_idx == original_bezier.cp1_idx
    assert new_bezier.cp2_idx == original_bezier.cp2_idx
    assert new_bezier.end_idx == original_bezier.end_idx
    assert new_bezier.construction == original_bezier.construction


def test_bezier_get_point_ids(registry):
    """Tests that a bezier correctly reports its defining point IDs."""
    p1 = registry.add_point(0, 0)
    p2 = registry.add_point(5, 10)
    p3 = registry.add_point(15, 10)
    p4 = registry.add_point(20, 0)
    bezier = registry.get_entity(registry.add_bezier(p1, p2, p3, p4))
    assert set(bezier.get_point_ids()) == {p1, p2, p3, p4}


def test_bezier_get_junction_point_ids(registry):
    """Tests that a bezier correctly reports its junction point IDs."""
    p1 = registry.add_point(0, 0)
    p2 = registry.add_point(5, 10)
    p3 = registry.add_point(15, 10)
    p4 = registry.add_point(20, 0)
    bezier = registry.get_entity(registry.add_bezier(p1, p2, p3, p4))
    assert set(bezier.get_junction_point_ids()) == {p1, p2, p3, p4}


def test_bezier_hit_test(registry):
    """Tests Bezier.hit_test method."""
    start = registry.add_point(0, 0)
    cp1 = registry.add_point(0, 20)
    cp2 = registry.add_point(20, 20)
    end = registry.add_point(20, 0)
    bezier = registry.get_entity(registry.add_bezier(start, cp1, cp2, end))
    threshold = 5.0

    # Point on the bezier (start and end points)
    assert bezier.hit_test(0, 0, threshold, registry) is True
    assert bezier.hit_test(20, 0, threshold, registry) is True

    # Point near the bezier curve (midpoint at t=0.5 is approximately (5, 15))
    assert bezier.hit_test(5, 15, threshold, registry) is True

    # Point far from the bezier
    assert bezier.hit_test(10, -10, threshold, registry) is False
    assert bezier.hit_test(50, 10, threshold, registry) is False


def test_bezier_update_constrained_status(registry):
    """Test Bezier.update_constrained_status logic."""
    p1 = registry.add_point(0, 0)
    p2 = registry.add_point(5, 10)
    p3 = registry.add_point(15, 10)
    p4 = registry.add_point(20, 0)
    bid = registry.add_bezier(p1, p2, p3, p4)
    bezier = registry.get_entity(bid)

    pt1 = registry.get_point(p1)
    pt2 = registry.get_point(p2)
    pt3 = registry.get_point(p3)
    pt4 = registry.get_point(p4)

    # Initially unconstrained
    pt1.constrained = False
    pt2.constrained = False
    pt3.constrained = False
    pt4.constrained = False
    bezier.update_constrained_status(registry, [])
    assert bezier.constrained is False

    # All points constrained
    pt1.constrained = True
    pt2.constrained = True
    pt3.constrained = True
    pt4.constrained = True
    bezier.update_constrained_status(registry, [])
    assert bezier.constrained is True


def test_bezier_to_geometry(registry):
    """Test Bezier.to_geometry method."""
    start = registry.add_point(0, 0)
    cp1 = registry.add_point(5, 10)
    cp2 = registry.add_point(15, 10)
    end = registry.add_point(20, 0)
    bezier = registry.get_entity(registry.add_bezier(start, cp1, cp2, end))
    geo = bezier.to_geometry(registry)
    assert isinstance(geo, Geometry)
    assert len(geo) == 2
    assert geo.data is not None
    assert geo.data[0][0] == 1.0  # Move command
    assert geo.data[1][0] == 4.0  # Bezier command


def test_bezier_get_set_state(registry):
    """Test state capture and restoration for Undo/Redo."""
    p1 = registry.add_point(0, 0)
    p2 = registry.add_point(5, 10)
    p3 = registry.add_point(15, 10)
    p4 = registry.add_point(20, 0)
    bid = registry.add_bezier(p1, p2, p3, p4)
    bezier = registry.get_entity(bid)

    # Verify initial state
    state = bezier.get_state()
    assert state == {"construction": False}

    # Modify state
    bezier.construction = True

    # Restore state
    bezier.set_state(state)
    assert bezier.construction is False
