import pytest
from rayforge.core.sketcher.entities import (
    EntityRegistry,
    Point,
    Line,
    Arc,
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

    registry.add_arc(start, end, center, cw=True)

    arc = registry.entities[0]
    assert isinstance(arc, Arc)
    assert arc.start_idx == start
    assert arc.center_idx == center
    assert arc.clockwise is True
    assert arc.type == "arc"
    assert arc.construction is False


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
