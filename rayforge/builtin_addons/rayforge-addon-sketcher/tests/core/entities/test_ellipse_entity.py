import pytest
from sketcher.core.entities import Ellipse
from sketcher.core.registry import EntityRegistry
from rayforge.core.geo.geometry import Geometry


@pytest.fixture
def registry():
    return EntityRegistry()


def test_ellipse_serialization_round_trip():
    """Tests the to_dict and from_dict methods for a single Ellipse."""
    original = Ellipse(
        id=30,
        center_idx=6,
        radius_x_pt_idx=7,
        radius_y_pt_idx=8,
        construction=True,
        helper_line_ids=[100, 101],
    )

    data = original.to_dict()
    assert data == {
        "id": 30,
        "type": "ellipse",
        "construction": True,
        "center_idx": 6,
        "radius_x_pt_idx": 7,
        "radius_y_pt_idx": 8,
        "helper_line_ids": [100, 101],
    }

    new_ellipse = Ellipse.from_dict(data)
    assert isinstance(new_ellipse, Ellipse)
    assert new_ellipse.id == original.id
    assert new_ellipse.center_idx == original.center_idx
    assert new_ellipse.radius_x_pt_idx == original.radius_x_pt_idx
    assert new_ellipse.radius_y_pt_idx == original.radius_y_pt_idx
    assert new_ellipse.construction == original.construction
    assert new_ellipse.helper_line_ids == original.helper_line_ids


def test_ellipse_serialization_without_helper_lines():
    """Tests serialization when helper_line_ids is None."""
    original = Ellipse(
        id=31,
        center_idx=1,
        radius_x_pt_idx=2,
        radius_y_pt_idx=3,
        helper_line_ids=None,
    )
    data = original.to_dict()
    assert data["helper_line_ids"] == []

    restored = Ellipse.from_dict(data)
    assert restored.helper_line_ids == []


def test_ellipse_get_point_ids(registry):
    """Tests that an ellipse correctly reports its defining point IDs."""
    center = registry.add_point(0, 0)
    rx = registry.add_point(10, 0)
    ry = registry.add_point(0, 5)
    ellipse = registry.get_entity(registry.add_ellipse(center, rx, ry))
    assert set(ellipse.get_point_ids()) == {center, rx, ry}


def test_ellipse_get_endpoint_ids(registry):
    """Tests that an ellipse has no endpoints (it's a closed loop)."""
    center = registry.add_point(0, 0)
    rx = registry.add_point(10, 0)
    ry = registry.add_point(0, 5)
    ellipse = registry.get_entity(registry.add_ellipse(center, rx, ry))
    assert ellipse.get_endpoint_ids() == []


def test_ellipse_get_junction_point_ids(registry):
    """Tests that an ellipse correctly reports its junction point IDs."""
    center = registry.add_point(0, 0)
    rx = registry.add_point(10, 0)
    ry = registry.add_point(0, 5)
    ellipse = registry.get_entity(registry.add_ellipse(center, rx, ry))
    assert set(ellipse.get_junction_point_ids()) == {center, rx, ry}


def test_ellipse_hit_test_on_edge(registry):
    """Tests Ellipse.hit_test on ellipse edge."""
    center = registry.add_point(0, 0)
    rx = registry.add_point(10, 0)
    ry = registry.add_point(0, 5)
    ellipse = registry.get_entity(registry.add_ellipse(center, rx, ry))
    threshold = 1.0

    assert ellipse.hit_test(10, 0, threshold, registry) is True
    assert ellipse.hit_test(-10, 0, threshold, registry) is True
    assert ellipse.hit_test(0, 5, threshold, registry) is True
    assert ellipse.hit_test(0, -5, threshold, registry) is True


def test_ellipse_hit_test_off_edge(registry):
    """Tests Ellipse.hit_test for points off the ellipse."""
    center = registry.add_point(0, 0)
    rx = registry.add_point(10, 0)
    ry = registry.add_point(0, 5)
    ellipse = registry.get_entity(registry.add_ellipse(center, rx, ry))
    threshold = 1.0

    assert ellipse.hit_test(5, 0, threshold, registry) is False
    assert ellipse.hit_test(0, 2, threshold, registry) is False
    assert ellipse.hit_test(15, 0, threshold, registry) is False


def test_ellipse_hit_test_rotated(registry):
    """Tests hit_test for a rotated ellipse."""
    center = registry.add_point(0, 0)
    rx = registry.add_point(5, 5)
    ry = registry.add_point(-5, 5)
    ellipse = registry.get_entity(registry.add_ellipse(center, rx, ry))
    threshold = 1.0

    import math

    dist = math.hypot(5, 5)
    assert ellipse.hit_test(dist, 0, threshold, registry) is True


def test_ellipse_hit_test_missing_points(registry):
    """Tests hit_test raises IndexError when points are missing."""
    ellipse = Ellipse(
        id=999, center_idx=999, radius_x_pt_idx=998, radius_y_pt_idx=997
    )
    with pytest.raises(IndexError):
        ellipse.hit_test(0, 0, 1.0, registry)


def test_ellipse_hit_test_zero_radii(registry):
    """Tests hit_test returns False when radii are zero."""
    center = registry.add_point(0, 0)
    same = registry.add_point(0, 0)
    ellipse = registry.get_entity(registry.add_ellipse(center, same, same))
    assert ellipse.hit_test(0, 0, 1.0, registry) is False


class MockRadiusConstraint:
    """Mock constraint class for testing ellipse status updates."""

    def __init__(self, ellipse_id_to_constrain):
        self._ellipse_id = ellipse_id_to_constrain

    def constrains_radius(self, registry, ellipse_id):
        return ellipse_id == self._ellipse_id


def test_ellipse_update_constrained_status(registry):
    """Test Ellipse.update_constrained_status logic."""
    center = registry.add_point(0, 0)
    rx = registry.add_point(10, 0)
    ry = registry.add_point(0, 5)
    eid = registry.add_ellipse(center, rx, ry)
    ellipse = registry.get_entity(eid)

    pt_c = registry.get_point(center)
    pt_rx = registry.get_point(rx)
    pt_ry = registry.get_point(ry)

    pt_c.constrained = False
    pt_rx.constrained = False
    pt_ry.constrained = False
    ellipse.update_constrained_status(registry, [])
    assert ellipse.constrained is False

    pt_c.constrained = True
    ellipse.update_constrained_status(registry, [])
    assert ellipse.constrained is False

    pt_rx.constrained = True
    pt_ry.constrained = True
    ellipse.update_constrained_status(registry, [])
    assert ellipse.constrained is True

    pt_rx.constrained = False
    pt_ry.constrained = False
    radius_constraint = MockRadiusConstraint(eid)
    ellipse.update_constrained_status(registry, [radius_constraint])
    assert ellipse.constrained is True

    pt_c.constrained = False
    ellipse.update_constrained_status(registry, [radius_constraint])
    assert ellipse.constrained is False


def test_ellipse_get_midpoint(registry):
    """Test getting the radius-x point position as midpoint."""
    center = registry.add_point(5, 5)
    rx = registry.add_point(15, 5)
    ry = registry.add_point(5, 10)

    eid = registry.add_ellipse(center, rx, ry)
    ellipse = registry.get_entity(eid)

    midpoint = ellipse.get_midpoint(registry)
    assert midpoint is not None
    assert midpoint == (15.0, 5.0)


def test_ellipse_get_ignorable_unconstrained_points(registry):
    """Tests ignorable points when ellipse is constrained."""
    center = registry.add_point(0, 0)
    rx = registry.add_point(10, 0)
    ry = registry.add_point(0, 5)
    eid = registry.add_ellipse(center, rx, ry)
    ellipse = registry.get_entity(eid)

    ellipse.constrained = False
    assert ellipse.get_ignorable_unconstrained_points() == []

    ellipse.constrained = True
    assert set(ellipse.get_ignorable_unconstrained_points()) == {rx, ry}


def test_ellipse_get_rigidly_connected_points(registry):
    """Test get_rigidly_connected_points returns all ellipse points."""
    center = registry.add_point(0, 0)
    rx = registry.add_point(10, 0)
    ry = registry.add_point(0, 5)
    eid = registry.add_ellipse(center, rx, ry)
    ellipse = registry.get_entity(eid)

    result = ellipse.get_rigidly_connected_points(center)
    assert set(result) == {center, rx, ry}

    result = ellipse.get_rigidly_connected_points(rx)
    assert result == []


@pytest.fixture
def selection_setup(registry):
    """Fixture for setting up ellipse entities for selection tests."""
    rect = (20, 20, 80, 80)

    c_in = registry.add_point(50, 50)
    rx_in = registry.add_point(60, 50)
    ry_in = registry.add_point(50, 60)
    ellipse_in = registry.get_entity(registry.add_ellipse(c_in, rx_in, ry_in))

    c_cross = registry.add_point(85, 50)
    rx_cross = registry.add_point(95, 50)
    ry_cross = registry.add_point(85, 60)
    ellipse_cross = registry.get_entity(
        registry.add_ellipse(c_cross, rx_cross, ry_cross)
    )

    c_out = registry.add_point(0, 0)
    rx_out = registry.add_point(5, 0)
    ry_out = registry.add_point(0, 3)
    ellipse_out = registry.get_entity(
        registry.add_ellipse(c_out, rx_out, ry_out)
    )

    return (
        registry,
        rect,
        {
            "ellipse_in": ellipse_in,
            "ellipse_cross": ellipse_cross,
            "ellipse_out": ellipse_out,
        },
    )


def test_ellipse_is_contained_by(selection_setup):
    """Test the is_contained_by method for Ellipse entities."""
    registry, rect, entities = selection_setup
    assert entities["ellipse_in"].is_contained_by(rect, registry) is True
    assert entities["ellipse_cross"].is_contained_by(rect, registry) is False
    assert entities["ellipse_out"].is_contained_by(rect, registry) is False


def test_ellipse_intersects_rect(selection_setup):
    """Test of intersects_rect method for Ellipse entities."""
    registry, rect, entities = selection_setup
    assert entities["ellipse_in"].intersects_rect(rect, registry) is True
    assert entities["ellipse_cross"].intersects_rect(rect, registry) is True
    assert entities["ellipse_out"].intersects_rect(rect, registry) is False


def test_ellipse_to_geometry(registry):
    """Test Ellipse.to_geometry method."""
    center = registry.add_point(0, 0)
    rx = registry.add_point(10, 0)
    ry = registry.add_point(0, 5)
    ellipse = registry.get_entity(registry.add_ellipse(center, rx, ry))
    geo = ellipse.to_geometry(registry)
    assert isinstance(geo, Geometry)
    assert len(geo) > 0
    assert geo.data is not None


def test_ellipse_to_geometry_zero_radii(registry):
    """Test to_geometry returns empty geometry for zero radii."""
    center = registry.add_point(0, 0)
    same = registry.add_point(0, 0)
    ellipse = registry.get_entity(registry.add_ellipse(center, same, same))
    geo = ellipse.to_geometry(registry)
    assert len(geo) == 0


def test_ellipse_create_fill_geometry(registry):
    """Test Ellipse.create_fill_geometry method."""
    center = registry.add_point(0, 0)
    rx = registry.add_point(10, 0)
    ry = registry.add_point(0, 5)
    ellipse = registry.get_entity(registry.add_ellipse(center, rx, ry))
    geo = ellipse.create_fill_geometry(registry)
    assert isinstance(geo, Geometry)
    assert len(geo) > 0


def test_ellipse_get_set_state(registry):
    """Test state capture and restoration for Undo/Redo."""
    center = registry.add_point(0, 0)
    rx = registry.add_point(10, 0)
    ry = registry.add_point(0, 5)
    eid = registry.add_ellipse(center, rx, ry)
    ellipse = registry.get_entity(eid)

    state = ellipse.get_state()
    assert state == {"construction": False}

    ellipse.construction = True

    ellipse.set_state(state)
    assert ellipse.construction is False


def test_ellipse_repr():
    """Test Ellipse string representation."""
    ellipse = Ellipse(
        id=1,
        center_idx=2,
        radius_x_pt_idx=3,
        radius_y_pt_idx=4,
        construction=False,
    )
    r = repr(ellipse)
    assert "Ellipse" in r
    assert "id=1" in r
    assert "center=2" in r


def test_ellipse_get_radii(registry):
    """Test _get_radii returns correct values."""
    center = registry.add_point(0, 0)
    rx = registry.add_point(10, 0)
    ry = registry.add_point(0, 5)
    eid = registry.add_ellipse(center, rx, ry)
    ellipse = registry.get_entity(eid)

    rx_val, ry_val = ellipse._get_radii(registry)
    assert rx_val == 10.0
    assert ry_val == 5.0


def test_ellipse_get_rotation(registry):
    """Test _get_rotation returns correct angle."""
    center = registry.add_point(0, 0)
    rx = registry.add_point(10, 10)
    ry = registry.add_point(-10, 10)
    eid = registry.add_ellipse(center, rx, ry)
    ellipse = registry.get_entity(eid)

    import math

    rotation = ellipse._get_rotation(registry)
    expected = math.atan2(10, 10)
    assert abs(rotation - expected) < 1e-9
