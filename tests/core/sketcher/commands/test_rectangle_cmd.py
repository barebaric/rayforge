from rayforge.core.sketcher import Sketch
from rayforge.core.sketcher.commands import (
    RectangleCommand,
    RectanglePreviewState,
)
from rayforge.core.sketcher.constraints import (
    HorizontalConstraint,
    VerticalConstraint,
)
from rayforge.core.sketcher.entities import Point


def test_rectangle_calculate_geometry_no_snap():
    """Test static calculation when no points are snapped."""
    result = RectangleCommand.calculate_geometry(0, 0, 100, 50, 0, None)
    assert result is not None
    points = result["points"]
    assert len(points) == 4
    assert len(result["entities"]) == 4
    assert len(result["constraints"]) == 4

    # p1 is the start point, represented by its ID
    assert points["p1_id"] == 0

    # p2, p3, p4 are new Point objects
    assert points["p2"].x == 100 and points["p2"].y == 0
    assert points["p3"].x == 100 and points["p3"].y == 50
    assert points["p4"].x == 0 and points["p4"].y == 50


def test_rectangle_calculate_geometry_with_snap():
    """Test static calculation when the end point is snapped."""
    result = RectangleCommand.calculate_geometry(0, 0, 100, 50, 0, 7)
    assert result is not None
    points = result["points"]
    # Check that the snapped point ID is preserved
    assert points["p3"].id == 7


def test_rectangle_calculate_geometry_degenerate():
    """Test static calculation returns None for a zero-size rectangle."""
    assert RectangleCommand.calculate_geometry(0, 0, 0, 50, 0, None) is None
    assert RectangleCommand.calculate_geometry(0, 0, 100, 0, 0, None) is None


def test_rectangle_command_execute_no_snap():
    """Test command execution with no point snapping."""
    sketch = Sketch()
    start_pid = sketch.add_point(0, 0)
    cmd = RectangleCommand(sketch, start_pid, (100, 50))
    cmd.execute()

    # 1 origin + 1 start_point + 2 new corners + 1 new end corner
    #   = 5 points total
    assert len(sketch.registry.points) == 5
    assert len(sketch.registry.entities) == 4
    assert len(sketch.constraints) == 4
    assert (
        sum(isinstance(c, HorizontalConstraint) for c in sketch.constraints)
        == 2
    )
    assert (
        sum(isinstance(c, VerticalConstraint) for c in sketch.constraints) == 2
    )


def test_rectangle_command_execute_with_snap():
    """Test command execution when snapping to an existing end point."""
    sketch = Sketch()
    start_pid = sketch.add_point(0, 0)
    end_pid = sketch.add_point(100, 50)
    cmd = RectangleCommand(sketch, start_pid, (100, 50), end_pid=end_pid)
    cmd.execute()

    # 1 origin + 1 start + 1 end + 2 new corners = 5 points total
    assert len(sketch.registry.points) == 5


def test_rectangle_command_execute_temp_start():
    """Test command execution when the start point was temporary."""
    sketch = Sketch()
    # Manually add a "temp" point to the registry
    start_pid = 100
    sketch.registry.points.append(Point(start_pid, 0, 0))
    assert len(sketch.registry.points) == 2  # Origin + temp start

    cmd = RectangleCommand(sketch, start_pid, (100, 50), is_start_temp=True)
    cmd.execute()

    # 1 origin + 3 new points (p2, p3, p4) + 1 re-added temp start point = 5
    assert len(sketch.registry.points) == 5
    # Verify the start point ID was reassigned by AddItemsCommand
    assert cmd.add_cmd is not None
    # Find the re-added start point in the command's point list
    re_added_start_point = next(
        p for p in cmd.add_cmd.points if p.x == 0 and p.y == 0
    )
    assert re_added_start_point.id != 100

    cmd.undo()
    # Temp points should NOT be restored on undo - only origin remains
    assert len(sketch.registry.points) == 1
    assert sketch.registry.points[0].id == 0  # Only origin


def test_rectangle_command_undo_no_dangling_points():
    """Test undo removes all added points."""
    sketch = Sketch()
    start_pid = sketch.add_point(0, 0)

    initial_point_count = len(sketch.registry.points)
    initial_entity_count = len(sketch.registry.entities)

    cmd = RectangleCommand(sketch, start_pid, (100, 50))
    cmd.execute()

    # Added 3 new points (p2, p3, p4)
    assert len(sketch.registry.points) == initial_point_count + 3
    assert len(sketch.registry.entities) > initial_entity_count

    cmd.undo()

    # All should be back to initial state
    assert len(sketch.registry.points) == initial_point_count
    assert len(sketch.registry.entities) == initial_entity_count


def test_rectangle_command_undo_with_temp_start():
    """Test undo with temporary start point."""
    sketch = Sketch()

    start_pid = sketch.add_point(0, 0)

    cmd = RectangleCommand(sketch, start_pid, (100, 50), is_start_temp=True)
    cmd.execute()

    assert len(sketch.registry.entities) == 4

    cmd.undo()

    # After undo: all entities removed, temp start should NOT be restored
    assert len(sketch.registry.entities) == 0
    # Temp start point should NOT be restored - only origin remains
    assert len(sketch.registry.points) == 1
    assert sketch.registry.points[0].id == 0  # Only origin


def test_rectangle_start_preview_no_snap():
    """Test start_preview creates initial preview state with temp point."""
    sketch = Sketch()
    state = RectangleCommand.start_preview(
        sketch.registry, 10, 20, snapped_pid=None
    )

    assert isinstance(state, RectanglePreviewState)
    assert state.start_temp is True
    assert state.p_end_id is not None
    assert state.preview_ids is not None
    assert len(state.preview_ids) > 0

    start_p = sketch.registry.get_point(state.start_id)
    assert start_p.x == 10
    assert start_p.y == 20


def test_rectangle_start_preview_with_snap():
    """Test start_preview uses existing point when snapped."""
    sketch = Sketch()
    existing_pid = sketch.add_point(50, 60)
    state = RectangleCommand.start_preview(
        sketch.registry, 10, 20, snapped_pid=existing_pid
    )

    assert isinstance(state, RectanglePreviewState)
    assert state.start_temp is False
    assert state.start_id == existing_pid


def test_rectangle_update_preview():
    """Test update_preview moves end point and refreshes geometry."""
    sketch = Sketch()
    state = RectangleCommand.start_preview(
        sketch.registry, 0, 0, snapped_pid=None
    )

    RectangleCommand.update_preview(sketch.registry, state, 100, 50)

    end_p = sketch.registry.get_point(state.p_end_id)
    assert end_p.x == 100
    assert end_p.y == 50

    p2 = sketch.registry.get_point(state.preview_ids["p2"])
    assert p2.x == 100
    assert p2.y == 0


def test_rectangle_cleanup_preview():
    """Test cleanup_preview removes all preview geometry except start."""
    sketch = Sketch()
    initial_point_count = len(sketch.registry.points)
    initial_entity_count = len(sketch.registry.entities)

    state = RectangleCommand.start_preview(
        sketch.registry, 0, 0, snapped_pid=None
    )

    assert len(sketch.registry.points) > initial_point_count
    assert len(sketch.registry.entities) > initial_entity_count

    RectangleCommand.cleanup_preview(sketch.registry, state)

    # cleanup_preview removes preview entities and points
    # (p_end_id, preview_ids) but leaves the start_id point -
    # it's the tool's responsibility to remove it if start_temp is True
    assert len(sketch.registry.entities) == initial_entity_count
    # Only start point remains from preview
    remaining_preview_ids = {
        state.start_id,
    }
    for p in sketch.registry.points:
        if p.id != 0:  # origin point
            assert p.id in remaining_preview_ids


def test_rectangle_cleanup_preview_with_snapped_start():
    """Test cleanup when start point was snapped (not temp)."""
    sketch = Sketch()
    existing_pid = sketch.add_point(50, 50)
    initial_point_count = len(sketch.registry.points)

    state = RectangleCommand.start_preview(
        sketch.registry, 0, 0, snapped_pid=existing_pid
    )

    RectangleCommand.cleanup_preview(sketch.registry, state)

    # Existing point should still be there
    assert len(sketch.registry.points) == initial_point_count


def test_rectangle_preview_lifecycle():
    """Test full preview lifecycle: start -> update -> cleanup."""
    sketch = Sketch()

    state = RectangleCommand.start_preview(
        sketch.registry, 0, 0, snapped_pid=None
    )
    assert state.start_temp is True

    RectangleCommand.update_preview(sketch.registry, state, 80, 60)

    end_p = sketch.registry.get_point(state.p_end_id)
    assert end_p.x == 80
    assert end_p.y == 60

    RectangleCommand.cleanup_preview(sketch.registry, state)

    assert len(sketch.registry.entities) == 0


def test_rectangle_create_preview_new():
    """Test create_preview creates new preview geometry."""
    sketch = Sketch()
    start_pid = sketch.add_point(0, 0)
    end_pid = sketch.add_point(100, 50)

    preview_ids = RectangleCommand.create_preview(
        sketch.registry, start_pid, end_pid
    )

    assert preview_ids is not None
    assert "p2" in preview_ids
    assert "p4" in preview_ids
    assert "line1" in preview_ids


def test_rectangle_create_preview_update():
    """Test create_preview updates existing geometry."""
    sketch = Sketch()
    start_pid = sketch.add_point(0, 0)
    end_pid = sketch.add_point(100, 50)

    preview_ids = RectangleCommand.create_preview(
        sketch.registry, start_pid, end_pid
    )
    assert preview_ids is not None

    sketch.registry.get_point(end_pid).x = 200
    sketch.registry.get_point(end_pid).y = 100

    result = RectangleCommand.create_preview(
        sketch.registry, start_pid, end_pid, preview_ids=preview_ids
    )

    assert result == preview_ids
    p2 = sketch.registry.get_point(preview_ids["p2"])
    assert p2.x == 200
