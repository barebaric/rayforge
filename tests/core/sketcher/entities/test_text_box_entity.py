import pytest
from rayforge.core.sketcher.entities import TextBoxEntity
from rayforge.core.sketcher.registry import EntityRegistry
from rayforge.core.geo.geometry import Geometry


@pytest.fixture
def registry():
    return EntityRegistry()


def test_text_box_serialization_round_trip():
    """Tests to_dict and from_dict methods for a single TextBox."""
    original_box = TextBoxEntity(
        id=10,
        origin_id=1,
        width_id=2,
        height_id=3,
        content="Hello World",
        font_params={
            "family": "sans-serif",
            "size": 10.0,
            "bold": False,
            "italic": False,
        },
    )

    data = original_box.to_dict()
    assert data["id"] == 10
    assert data["type"] == "text_box"
    assert data["origin_id"] == 1
    assert data["width_id"] == 2
    assert data["height_id"] == 3
    assert data["content"] == "Hello World"
    assert data["font_params"] == {
        "family": "sans-serif",
        "size": 10.0,
        "bold": False,
        "italic": False,
    }

    new_box = TextBoxEntity.from_dict(data)
    assert isinstance(new_box, TextBoxEntity)
    assert new_box.id == original_box.id
    assert new_box.origin_id == original_box.origin_id
    assert new_box.width_id == original_box.width_id
    assert new_box.height_id == original_box.height_id
    assert new_box.content == original_box.content
    assert new_box.font_params == original_box.font_params


def test_text_box_get_point_ids(registry):
    """Tests that a text box correctly reports its defining point IDs."""
    p1 = registry.add_point(0, 0)
    p2 = registry.add_point(10, 0)
    p3 = registry.add_point(0, 10)
    box = registry.get_entity(registry.add_text_box(p1, p2, p3, "Test", {}))
    assert set(box.get_point_ids()) == {p1, p2, p3}


def test_text_box_update_constrained_status(registry):
    """Test TextBoxEntity.update_constrained_status logic."""
    p1 = registry.add_point(0, 0)
    p2 = registry.add_point(10, 0)
    p3 = registry.add_point(0, 10)
    box = registry.get_entity(registry.add_text_box(p1, p2, p3, "Test", {}))

    pt1 = registry.get_point(p1)
    pt2 = registry.get_point(p2)
    pt3 = registry.get_point(p3)

    # Initially unconstrained
    pt1.constrained = False
    pt2.constrained = False
    pt3.constrained = False
    box.update_constrained_status(registry, [])
    assert box.constrained is False

    # One point constrained
    pt1.constrained = True
    box.update_constrained_status(registry, [])
    assert box.constrained is False

    # Two points constrained
    pt2.constrained = True
    box.update_constrained_status(registry, [])
    assert box.constrained is False

    # All three points constrained
    pt3.constrained = True
    box.update_constrained_status(registry, [])
    assert box.constrained is True


@pytest.fixture
def selection_setup(registry):
    """Fixture for setting up text box entities for selection tests."""
    rect = (20, 20, 80, 80)

    # Text box fully inside
    p_in1 = registry.add_point(30, 30)
    p_in2 = registry.add_point(70, 30)
    p_in3 = registry.add_point(30, 70)
    box_in = registry.get_entity(
        registry.add_text_box(p_in1, p_in2, p_in3, "Inside", {})
    )

    # Text box intersecting
    p_cross1 = registry.add_point(10, 50)
    p_cross2 = registry.add_point(90, 50)
    p_cross3 = registry.add_point(10, 90)
    box_cross = registry.get_entity(
        registry.add_text_box(p_cross1, p_cross2, p_cross3, "Crossing", {})
    )

    # Text box outside
    p_out1 = registry.add_point(0, 0)
    p_out2 = registry.add_point(10, 0)
    p_out3 = registry.add_point(0, 10)
    box_out = registry.get_entity(
        registry.add_text_box(p_out1, p_out2, p_out3, "Outside", {})
    )

    return (
        registry,
        rect,
        {
            "box_in": box_in,
            "box_cross": box_cross,
            "box_out": box_out,
        },
    )


def test_text_box_is_contained_by(selection_setup):
    """Test is_contained_by method for TextBox entities."""
    registry, rect, entities = selection_setup
    assert entities["box_in"].is_contained_by(rect, registry) is True
    assert entities["box_cross"].is_contained_by(rect, registry) is False
    assert entities["box_out"].is_contained_by(rect, registry) is False


def test_text_box_intersects_rect(selection_setup):
    """Test of intersects_rect method for TextBox entities."""
    registry, rect, entities = selection_setup
    assert entities["box_in"].intersects_rect(rect, registry) is True
    assert entities["box_cross"].intersects_rect(rect, registry) is True
    assert entities["box_out"].intersects_rect(rect, registry) is False


def test_text_box_to_geometry(registry):
    """Test TextBoxEntity.to_geometry method."""
    p1 = registry.add_point(0, 0)
    p2 = registry.add_point(10, 0)
    p3 = registry.add_point(0, 10)
    box = registry.get_entity(
        registry.add_text_box(p1, p2, p3, "Test", {})
    )
    geo = box.to_geometry(registry)
    assert isinstance(geo, Geometry)
    assert len(geo) > 0
