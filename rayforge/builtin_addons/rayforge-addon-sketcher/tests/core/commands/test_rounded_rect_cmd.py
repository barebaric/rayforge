from sketcher.core import Sketch
from sketcher.core.commands import (
    RoundedRectCommand,
    RoundedRectPreviewState,
)
from sketcher.core.constraints import TangentConstraint
from sketcher.core.entities import Line, Arc


def test_rounded_rect_calculate_geometry():
    """Test the static geometry calculation for a rounded rectangle."""
    result = RoundedRectCommand.calculate_geometry(0, 0, 100, 50, 10.0)
    assert result is not None
    points = result["points"]
    assert len(points) == 12  # 8 tangent + 4 center
    assert len(result["entities"]) == 8  # 4 lines + 4 arcs
    assert len(result["constraints"]) == 17

    assert sum(isinstance(e, Line) for e in result["entities"]) == 4
    assert sum(isinstance(e, Arc) for e in result["entities"]) == 4
    assert (
        sum(isinstance(c, TangentConstraint) for c in result["constraints"])
        == 8
    )


def test_rounded_rect_calculate_geometry_degenerate():
    """Test static calculation returns None for a zero-size rectangle."""
    assert RoundedRectCommand.calculate_geometry(0, 0, 0, 50, 10.0) is None


def test_rounded_rect_command_execute():
    """Test command execution creates the correct number of items."""
    sketch = Sketch()
    start_pid = sketch.add_point(0, 0)
    cmd = RoundedRectCommand(sketch, start_pid, (100, 50), 10.0)
    cmd.execute()

    # Points: 1 origin + 1 existing start + (8 tangent + 4 center) added = 14
    # The virtual corner points are not added to the sketch.
    assert len(sketch.registry.points) == 14
    assert len(sketch.registry.entities) == 8
    assert len(sketch.constraints) == 17


def test_rounded_rect_start_preview_no_snap():
    """Test start_preview creates initial preview state with temp point."""
    sketch = Sketch()
    state = RoundedRectCommand.start_preview(
        sketch.registry, 10, 20, snapped_pid=None, radius=10.0
    )

    assert isinstance(state, RoundedRectPreviewState)
    assert state.start_temp is True
    assert state.p_end_id is not None
    assert state.preview_ids is not None
    assert len(state.preview_ids) > 0
    assert state.radius == 10.0

    start_p = sketch.registry.get_point(state.start_id)
    assert start_p.x == 10
    assert start_p.y == 20


def test_rounded_rect_start_preview_with_snap():
    """Test start_preview uses existing point when snapped."""
    sketch = Sketch()
    existing_pid = sketch.add_point(50, 60)
    state = RoundedRectCommand.start_preview(
        sketch.registry, 10, 20, snapped_pid=existing_pid, radius=15.0
    )

    assert isinstance(state, RoundedRectPreviewState)
    assert state.start_temp is False
    assert state.start_id == existing_pid
    assert state.radius == 15.0


def test_rounded_rect_update_preview():
    """Test update_preview moves end point and refreshes geometry."""
    sketch = Sketch()
    state = RoundedRectCommand.start_preview(
        sketch.registry, 0, 0, snapped_pid=None, radius=10.0
    )

    RoundedRectCommand.update_preview(sketch.registry, state, 100, 50)

    end_p = sketch.registry.get_point(state.p_end_id)
    assert end_p.x == 100
    assert end_p.y == 50


def test_rounded_rect_cleanup_preview():
    """Test cleanup_preview removes all preview geometry except start point."""
    sketch = Sketch()
    initial_point_count = len(sketch.registry.points)
    initial_entity_count = len(sketch.registry.entities)

    state = RoundedRectCommand.start_preview(
        sketch.registry, 0, 0, snapped_pid=None, radius=10.0
    )

    assert len(sketch.registry.points) > initial_point_count
    assert len(sketch.registry.entities) > initial_entity_count

    RoundedRectCommand.cleanup_preview(sketch.registry, state)

    # cleanup_preview removes preview entities and points (p_end_id,
    # preview_ids) but leaves the start_id point - it's the tool's
    # responsibility to remove it if start_temp is True
    assert len(sketch.registry.entities) == initial_entity_count
    # Only start point remains from preview
    remaining_preview_ids = {
        state.start_id,
    }
    for p in sketch.registry.points:
        if p.id != 0:  # origin point
            assert p.id in remaining_preview_ids


def test_rounded_rect_cleanup_preview_with_snapped_start():
    """Test cleanup when start point was snapped (not temp)."""
    sketch = Sketch()
    existing_pid = sketch.add_point(50, 50)
    initial_point_count = len(sketch.registry.points)

    state = RoundedRectCommand.start_preview(
        sketch.registry, 0, 0, snapped_pid=existing_pid, radius=10.0
    )

    RoundedRectCommand.cleanup_preview(sketch.registry, state)

    # Existing point should still be there
    assert len(sketch.registry.points) == initial_point_count


def test_rounded_rect_preview_lifecycle():
    """Test full preview lifecycle: start -> update -> cleanup."""
    sketch = Sketch()

    state = RoundedRectCommand.start_preview(
        sketch.registry, 0, 0, snapped_pid=None, radius=10.0
    )
    assert state.start_temp is True

    RoundedRectCommand.update_preview(sketch.registry, state, 80, 60)

    end_p = sketch.registry.get_point(state.p_end_id)
    assert end_p.x == 80
    assert end_p.y == 60

    RoundedRectCommand.cleanup_preview(sketch.registry, state)

    assert len(sketch.registry.entities) == 0


def test_rounded_rect_create_preview_new():
    """Test create_preview creates new preview geometry."""
    sketch = Sketch()
    start_pid = sketch.add_point(0, 0)
    end_pid = sketch.add_point(100, 50)

    preview_ids = RoundedRectCommand.create_preview(
        sketch.registry, start_pid, end_pid, 10.0
    )

    assert preview_ids is not None
    assert "t1" in preview_ids
    assert "t2" in preview_ids
    assert "line1" in preview_ids
    assert "arc1" in preview_ids


def test_rounded_rect_create_preview_update():
    """Test create_preview updates existing geometry."""
    sketch = Sketch()
    start_pid = sketch.add_point(0, 0)
    end_pid = sketch.add_point(100, 50)

    preview_ids = RoundedRectCommand.create_preview(
        sketch.registry, start_pid, end_pid, 10.0
    )

    sketch.registry.get_point(end_pid).x = 200
    sketch.registry.get_point(end_pid).y = 100

    result = RoundedRectCommand.create_preview(
        sketch.registry, start_pid, end_pid, 10.0, preview_ids=preview_ids
    )

    assert result == preview_ids


def test_rounded_rect_preview_with_small_radius():
    """Test preview handles radius larger than half dimensions."""
    sketch = Sketch()
    state = RoundedRectCommand.start_preview(
        sketch.registry, 0, 0, snapped_pid=None, radius=50.0
    )

    RoundedRectCommand.update_preview(sketch.registry, state, 30, 20)

    end_p = sketch.registry.get_point(state.p_end_id)
    assert end_p.x == 30
    assert end_p.y == 20


def test_rounded_rect_preview_get_dimensions_returns_all():
    """Test that dimensions show width, height, and radius."""
    sketch = Sketch()
    state = RoundedRectCommand.start_preview(
        sketch.registry, 0, 0, snapped_pid=None, radius=10.0
    )
    RoundedRectCommand.update_preview(sketch.registry, state, 100, 50)

    dims = state.get_dimensions(sketch.registry)

    assert len(dims) == 3
    labels = [d.label for d in dims]
    assert "100.00" in labels
    assert "50.00" in labels
    assert "R10.00" in labels


def test_rounded_rect_preview_get_dimensions_zero_radius():
    """Test that zero radius doesn't add radius dimension."""
    sketch = Sketch()
    state = RoundedRectCommand.start_preview(
        sketch.registry, 0, 0, snapped_pid=None, radius=0.0
    )
    RoundedRectCommand.update_preview(sketch.registry, state, 100, 50)

    dims = state.get_dimensions(sketch.registry)

    assert len(dims) == 2
    labels = [d.label for d in dims]
    assert "100.00" in labels
    assert "50.00" in labels


def test_rounded_rect_preview_get_dimensions_width_position_on_top_edge():
    """Test width label is positioned on top edge."""
    sketch = Sketch()
    state = RoundedRectCommand.start_preview(
        sketch.registry, 0, 0, snapped_pid=None, radius=10.0
    )
    RoundedRectCommand.update_preview(sketch.registry, state, 100, 50)

    dims = state.get_dimensions(sketch.registry)

    width_dim = next(d for d in dims if d.label == "100.00")
    assert width_dim.position[0] == 50.0
    assert width_dim.position[1] == 0


def test_rounded_rect_preview_get_dimensions_height_position_on_right_edge():
    """Test height label is positioned on right edge."""
    sketch = Sketch()
    state = RoundedRectCommand.start_preview(
        sketch.registry, 0, 0, snapped_pid=None, radius=10.0
    )
    RoundedRectCommand.update_preview(sketch.registry, state, 100, 50)

    dims = state.get_dimensions(sketch.registry)

    height_dim = next(d for d in dims if d.label == "50.00")
    assert height_dim.position[0] == 100
    assert height_dim.position[1] == 25.0


def test_rounded_rect_preview_get_dimensions_radius_label_format():
    """Test that radius label uses R prefix."""
    sketch = Sketch()
    state = RoundedRectCommand.start_preview(
        sketch.registry, 0, 0, snapped_pid=None, radius=15.5
    )
    RoundedRectCommand.update_preview(sketch.registry, state, 100, 50)

    dims = state.get_dimensions(sketch.registry)

    radius_dim = next(d for d in dims if d.label.startswith("R"))
    assert radius_dim.label == "R15.50"


def test_rounded_rect_preview_get_dimensions_no_leader_end():
    """Test that rounded rect dimensions have no leader_end."""
    sketch = Sketch()
    state = RoundedRectCommand.start_preview(
        sketch.registry, 0, 0, snapped_pid=None, radius=10.0
    )
    RoundedRectCommand.update_preview(sketch.registry, state, 100, 50)

    dims = state.get_dimensions(sketch.registry)

    for dim in dims:
        assert dim.leader_end is None


def test_rounded_rect_preview_get_dimensions_missing_point():
    """Test that missing points return empty list."""
    sketch = Sketch()
    state = RoundedRectCommand.start_preview(
        sketch.registry, 0, 0, snapped_pid=None, radius=10.0
    )
    sketch.registry.points.clear()

    dims = state.get_dimensions(sketch.registry)

    assert dims == []
