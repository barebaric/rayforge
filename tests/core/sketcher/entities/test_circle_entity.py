import pytest
from rayforge.core.sketcher.entities import Circle
from rayforge.core.sketcher.registry import EntityRegistry


@pytest.fixture
def registry():
    return EntityRegistry()


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


def test_circle_get_point_ids(registry):
    """Tests that a circle correctly reports its defining point IDs."""
    p1 = registry.add_point(0, 0)
    p2 = registry.add_point(10, 0)
    circle = registry.get_entity(registry.add_circle(p1, p2))
    assert set(circle.get_point_ids()) == {p1, p2}


class MockRadiusConstraint:
    """Mock constraint class for testing circle status updates."""

    def __init__(self, circle_id_to_constrain):
        self._circle_id = circle_id_to_constrain

    def constrains_radius(self, registry, circle_id):
        return circle_id == self._circle_id


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

    # Case 3: Center + Radius Point constrained (Fully defined by points)
    pt_r.constrained = True
    circle.update_constrained_status(registry, [])
    assert circle.constrained is True

    # Case 4: Center constrained, but a Radius constraint exists
    pt_c.constrained = True
    pt_r.constrained = False  # Radius point is not constrained
    radius_constraint = MockRadiusConstraint(cid)
    circle.update_constrained_status(registry, [radius_constraint])
    assert circle.constrained is True

    # Case 5: Only a radius constraint exists, center is not constrained
    pt_c.constrained = False
    pt_r.constrained = False
    circle.update_constrained_status(registry, [radius_constraint])
    assert circle.constrained is False


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


def test_circle_get_ignorable_unconstrained_points(registry):
    """
    Tests that a Circle correctly identifies its radius point as ignorable
    only when the circle itself is constrained.
    """
    center = registry.add_point(0, 0)
    radius_pt = registry.add_point(10, 0)
    cid = registry.add_circle(center, radius_pt)
    circle = registry.get_entity(cid)

    # When circle is not constrained, list should be empty
    circle.constrained = False
    assert circle.get_ignorable_unconstrained_points() == []

    # When circle is constrained, radius point can be ignored
    circle.constrained = True
    assert circle.get_ignorable_unconstrained_points() == [radius_pt]


@pytest.fixture
def selection_setup(registry):
    """Fixture for setting up circle entities for selection tests."""
    rect = (20, 20, 80, 80)

    # Circle fully inside
    c_in = registry.add_point(50, 50)
    r_in = registry.add_point(60, 50)  # radius 10
    circle_in = registry.get_entity(registry.add_circle(c_in, r_in))

    # Circle intersecting
    c_cross = registry.add_point(85, 50)
    r_cross = registry.add_point(95, 50)  # radius 10
    circle_cross = registry.get_entity(registry.add_circle(c_cross, r_cross))

    # Circle outside
    c_out = registry.add_point(0, 0)
    r_out = registry.add_point(5, 0)  # radius 5
    circle_out = registry.get_entity(registry.add_circle(c_out, r_out))

    return (
        registry,
        rect,
        {
            "circle_in": circle_in,
            "circle_cross": circle_cross,
            "circle_out": circle_out,
        },
    )


def test_circle_is_contained_by(selection_setup):
    """Test the is_contained_by method for Circle entities."""
    registry, rect, entities = selection_setup
    assert entities["circle_in"].is_contained_by(rect, registry) is True
    assert entities["circle_cross"].is_contained_by(rect, registry) is False
    assert entities["circle_out"].is_contained_by(rect, registry) is False


def test_circle_intersects_rect(selection_setup):
    """Test the intersects_rect method for Circle entities."""
    registry, rect, entities = selection_setup
    assert entities["circle_in"].intersects_rect(rect, registry) is True
    assert entities["circle_cross"].intersects_rect(rect, registry) is True
    assert entities["circle_out"].intersects_rect(rect, registry) is False
