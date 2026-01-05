import pytest
from rayforge.core.sketcher.entities import Line
from rayforge.core.sketcher.registry import EntityRegistry


@pytest.fixture
def registry():
    return EntityRegistry()


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


def test_line_get_point_ids(registry):
    """Tests that a line correctly reports its defining point IDs."""
    p1 = registry.add_point(0, 0)
    p2 = registry.add_point(10, 0)
    line = registry.get_entity(registry.add_line(p1, p2))
    assert set(line.get_point_ids()) == {p1, p2}


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


@pytest.fixture
def selection_setup(registry):
    """Fixture for setting up line entities for selection tests."""
    rect = (20, 20, 80, 80)

    # Line fully inside
    p_in1 = registry.add_point(30, 30)
    p_in2 = registry.add_point(70, 70)
    line_in = registry.get_entity(registry.add_line(p_in1, p_in2))

    # Line intersecting
    p_cross1 = registry.add_point(10, 50)
    p_cross2 = registry.add_point(90, 50)
    line_cross = registry.get_entity(registry.add_line(p_cross1, p_cross2))

    # Line outside
    p_out1 = registry.add_point(0, 0)
    p_out2 = registry.add_point(10, 10)
    line_out = registry.get_entity(registry.add_line(p_out1, p_out2))

    return (
        registry,
        rect,
        {
            "line_in": line_in,
            "line_cross": line_cross,
            "line_out": line_out,
        },
    )


def test_line_is_contained_by(selection_setup):
    """Test the is_contained_by method for Line entities."""
    registry, rect, entities = selection_setup
    assert entities["line_in"].is_contained_by(rect, registry) is True
    assert entities["line_cross"].is_contained_by(rect, registry) is False
    assert entities["line_out"].is_contained_by(rect, registry) is False


def test_line_intersects_rect(selection_setup):
    """Test the intersects_rect method for Line entities."""
    registry, rect, entities = selection_setup
    assert entities["line_in"].intersects_rect(rect, registry) is True
    assert entities["line_cross"].intersects_rect(rect, registry) is True
    assert entities["line_out"].intersects_rect(rect, registry) is False
