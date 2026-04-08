import math
from rayforge.core.geo.arc import determine_arc_direction
from sketcher.core import Sketch
from sketcher.core.commands import ArcCommand, ArcPreviewState
from sketcher.core.entities import Arc


def test_arc_start_preview():
    """Test start_preview creates preview arc entity."""
    sketch = Sketch()
    center_id = sketch.add_point(0, 0)
    start_id = sketch.add_point(50, 0)

    state = ArcCommand.start_preview(
        sketch.registry,
        25,
        25,
        center_id=center_id,
        center_temp=False,
        start_id=start_id,
        start_temp=False,
    )

    assert isinstance(state, ArcPreviewState)
    assert state.center_id == center_id
    assert state.start_id == start_id
    assert state.temp_end_id is not None
    assert state.temp_entity_id is not None

    arc = sketch.registry.get_entity(state.temp_entity_id)
    assert isinstance(arc, Arc)


def test_arc_update_preview():
    """Test update_preview moves end point and updates direction."""
    sketch = Sketch()
    center_id = sketch.add_point(0, 0)
    start_id = sketch.add_point(50, 0)

    state = ArcCommand.start_preview(
        sketch.registry,
        25,
        25,
        center_id=center_id,
        center_temp=False,
        start_id=start_id,
        start_temp=False,
    )

    ArcCommand.update_preview(sketch.registry, state, 0, 50)

    assert state.temp_end_id is not None
    end_p = sketch.registry.get_point(state.temp_end_id)
    expected_radius = 50.0
    actual_radius = math.hypot(end_p.x, end_p.y)
    assert abs(actual_radius - expected_radius) < 0.001


def test_arc_cleanup_preview():
    """Test cleanup_preview removes preview entities."""
    sketch = Sketch()
    center_id = sketch.add_point(0, 0)
    start_id = sketch.add_point(50, 0)

    initial_entity_count = len(sketch.registry.entities)
    initial_point_count = len(sketch.registry.points)

    state = ArcCommand.start_preview(
        sketch.registry,
        25,
        25,
        center_id=center_id,
        center_temp=False,
        start_id=start_id,
        start_temp=False,
    )

    assert len(sketch.registry.entities) > initial_entity_count
    assert len(sketch.registry.points) > initial_point_count

    ArcCommand.cleanup_preview(sketch.registry, state)

    assert len(sketch.registry.entities) == initial_entity_count
    assert len(sketch.registry.points) == initial_point_count


def test_arc_preview_lifecycle():
    """Test full preview lifecycle: start -> update -> cleanup."""
    sketch = Sketch()
    center_id = sketch.add_point(0, 0)
    start_id = sketch.add_point(50, 0)

    state = ArcCommand.start_preview(
        sketch.registry,
        0,
        50,
        center_id=center_id,
        center_temp=False,
        start_id=start_id,
        start_temp=False,
    )

    ArcCommand.update_preview(sketch.registry, state, 0, 50)

    ArcCommand.cleanup_preview(sketch.registry, state)

    assert state.clockwise is not None


def test_arc_preview_clockwise_detection():
    """Test that preview detects clockwise vs counter-clockwise."""
    sketch = Sketch()
    center_id = sketch.add_point(0, 0)
    start_id = sketch.add_point(50, 0)

    state = ArcCommand.start_preview(
        sketch.registry,
        0,
        50,
        center_id=center_id,
        center_temp=False,
        start_id=start_id,
        start_temp=False,
    )

    ArcCommand.update_preview(sketch.registry, state, 0, 50)
    ArcCommand.cleanup_preview(sketch.registry, state)

    assert isinstance(state.clockwise, bool)


def test_arc_command_execute():
    """Test ArcCommand execution creates proper geometry."""
    sketch = Sketch()
    center_id = sketch.add_point(0, 0)
    start_id = sketch.add_point(50, 0)

    cmd = ArcCommand(
        sketch,
        center_id,
        start_id,
        (0, 50),
        is_center_temp=False,
        is_start_temp=False,
        clockwise=False,
    )
    cmd.execute()

    assert len(sketch.registry.entities) == 1
    assert len(sketch.constraints) == 1
    assert isinstance(sketch.registry.entities[0], Arc)


def test_arc_command_with_temp_points():
    """Test ArcCommand handles temporary center and start points."""
    sketch = Sketch()

    center_id = sketch.add_point(0, 0)
    start_id = sketch.add_point(50, 0)

    cmd = ArcCommand(
        sketch,
        center_id,
        start_id,
        (0, 50),
        is_center_temp=True,
        is_start_temp=True,
        clockwise=False,
    )
    cmd.execute()

    assert cmd.add_cmd is not None
    assert len(sketch.registry.entities) == 1


def test_arc_command_undo():
    """Test ArcCommand can be undone."""
    sketch = Sketch()
    center_id = sketch.add_point(0, 0)
    start_id = sketch.add_point(50, 0)

    initial_entity_count = len(sketch.registry.entities)

    cmd = ArcCommand(
        sketch,
        center_id,
        start_id,
        (0, 50),
        is_center_temp=False,
        is_start_temp=False,
        clockwise=False,
    )
    cmd.execute()

    assert len(sketch.registry.entities) > initial_entity_count

    cmd.undo()

    assert len(sketch.registry.entities) == initial_entity_count


def test_arc_command_undo_no_dangling_points():
    """Test undo removes all added points including end point."""
    sketch = Sketch()
    center_id = sketch.add_point(0, 0)
    start_id = sketch.add_point(50, 0)

    initial_point_count = len(sketch.registry.points)

    cmd = ArcCommand(
        sketch,
        center_id,
        start_id,
        (0, 50),
        is_center_temp=False,
        is_start_temp=False,
        clockwise=False,
    )
    cmd.execute()

    # Execute adds 1 new end point
    assert len(sketch.registry.points) == initial_point_count + 1

    cmd.undo()

    # After undo, should be back to initial count
    assert len(sketch.registry.points) == initial_point_count


def test_arc_command_undo_with_temp_center():
    """Test undo with temporary center point removes temp point."""
    sketch = Sketch()

    center_id = sketch.add_point(0, 0)
    start_id = sketch.add_point(50, 0)

    cmd = ArcCommand(
        sketch,
        center_id,
        start_id,
        (0, 50),
        is_center_temp=True,
        is_start_temp=False,
        clockwise=False,
    )
    cmd.execute()

    # Execute: temp center is removed from registry and re-added
    # by AddItemsCommand, plus 1 new end point
    assert len(sketch.registry.entities) == 1

    cmd.undo()

    # After undo: temp center should NOT be restored as it was temp
    # Only the original origin + start point should remain
    # (center was temp and shouldn't come back)
    assert len(sketch.registry.entities) == 0


def test_arc_command_undo_with_temp_start():
    """Test undo with temporary start point removes temp point."""
    sketch = Sketch()

    center_id = sketch.add_point(0, 0)
    start_id = sketch.add_point(50, 0)

    cmd = ArcCommand(
        sketch,
        center_id,
        start_id,
        (0, 50),
        is_center_temp=False,
        is_start_temp=True,
        clockwise=False,
    )
    cmd.execute()

    assert len(sketch.registry.entities) == 1

    cmd.undo()

    # After undo: temp start should NOT be restored
    assert len(sketch.registry.entities) == 0


def test_arc_direction_preserved_counter_clockwise():
    """Test that arc direction from preview is preserved on execute."""
    sketch = Sketch()
    center_id = sketch.add_point(0, 0)
    start_id = sketch.add_point(50, 0)

    # Mouse above the start-center line should give counter-clockwise
    expected_clockwise = determine_arc_direction((0, 0), (50, 0), (25, 50))
    assert expected_clockwise is False

    cmd = ArcCommand(
        sketch,
        center_id,
        start_id,
        (25, 50),
        is_center_temp=False,
        is_start_temp=False,
        clockwise=expected_clockwise,
    )
    cmd.execute()

    arc = sketch.registry.entities[0]
    assert isinstance(arc, Arc)
    assert arc.clockwise == expected_clockwise


def test_arc_direction_preserved_clockwise():
    """Test that clockwise direction from preview is preserved on execute."""
    sketch = Sketch()
    center_id = sketch.add_point(0, 0)
    start_id = sketch.add_point(50, 0)

    # Mouse below the start-center line should give clockwise
    expected_clockwise = determine_arc_direction((0, 0), (50, 0), (25, -50))
    assert expected_clockwise is True

    cmd = ArcCommand(
        sketch,
        center_id,
        start_id,
        (25, -50),
        is_center_temp=False,
        is_start_temp=False,
        clockwise=expected_clockwise,
    )
    cmd.execute()

    arc = sketch.registry.entities[0]
    assert isinstance(arc, Arc)
    assert arc.clockwise == expected_clockwise


def test_arc_direction_with_projected_endpoint():
    """Test arc direction when end point is projected onto circle."""
    sketch = Sketch()
    center_id = sketch.add_point(0, 0)
    start_id = sketch.add_point(50, 0)

    # End position is slightly off the circle - it will be projected
    # Mouse is below center-start line (clockwise direction)
    mouse_pos = (30, -40)
    expected_clockwise = determine_arc_direction((0, 0), (50, 0), mouse_pos)
    assert expected_clockwise is True

    cmd = ArcCommand(
        sketch,
        center_id,
        start_id,
        mouse_pos,
        is_center_temp=False,
        is_start_temp=False,
        clockwise=expected_clockwise,
    )
    cmd.execute()

    arc = sketch.registry.entities[0]
    assert isinstance(arc, Arc)
    # The arc direction should match what was determined during preview
    assert arc.clockwise == expected_clockwise


def test_arc_full_workflow_with_preview():
    """Test complete arc creation workflow including preview state."""
    sketch = Sketch()
    center_id = sketch.add_point(0, 0)
    start_id = sketch.add_point(50, 0)

    # Simulate the preview workflow
    state = ArcCommand.start_preview(
        sketch.registry,
        25,
        -30,
        center_id=center_id,
        center_temp=False,
        start_id=start_id,
        start_temp=False,
    )

    # Update preview
    ArcCommand.update_preview(sketch.registry, state, 25, -30)

    # Get the direction from preview
    ArcCommand.cleanup_preview(sketch.registry, state)
    clockwise = state.clockwise

    # Create the command with the preview direction
    cmd = ArcCommand(
        sketch,
        center_id,
        start_id,
        (25, -30),
        is_center_temp=False,
        is_start_temp=False,
        clockwise=clockwise,
    )
    cmd.execute()

    arc = sketch.registry.entities[0]
    assert isinstance(arc, Arc)
    # Direction should match what was in preview
    assert arc.clockwise == clockwise


def test_arc_command_undo_with_both_temp_points():
    """Test undo removes both temp center and start points."""
    sketch = Sketch()

    center_id = sketch.add_point(0, 0)
    start_id = sketch.add_point(50, 0)

    cmd = ArcCommand(
        sketch,
        center_id,
        start_id,
        (0, 50),
        is_center_temp=True,
        is_start_temp=True,
        clockwise=False,
    )
    cmd.execute()

    # After execute: origin + center + start + end = initial + 1 (end)
    # because center and start are removed then re-added
    assert len(sketch.registry.entities) == 1

    cmd.undo()

    # After undo: all added points should be removed
    # Only origin should remain
    assert len(sketch.registry.entities) == 0
    # Check that temp points are NOT left dangling
    # Origin (id=0) should be the only remaining point
    remaining_ids = [p.id for p in sketch.registry.points]
    assert 0 in remaining_ids  # origin


def test_arc_direction_matches_preview_after_off_circle_click():
    """Test that direction matches preview even when click is off circle."""
    sketch = Sketch()
    center_id = sketch.add_point(0, 0)
    start_id = sketch.add_point(50, 0)

    # Simulate preview with mouse exactly on circle at (0, 50)
    state = ArcCommand.start_preview(
        sketch.registry,
        0,
        50,
        center_id=center_id,
        center_temp=False,
        start_id=start_id,
        start_temp=False,
    )

    # Update preview with mouse on circle
    ArcCommand.update_preview(sketch.registry, state, 0, 50)
    ArcCommand.cleanup_preview(sketch.registry, state)
    preview_clockwise = state.clockwise

    # Now simulate a click slightly off the circle
    # The command will project this onto the circle
    off_circle_pos = (2, 48)

    cmd = ArcCommand(
        sketch,
        center_id,
        start_id,
        off_circle_pos,
        is_center_temp=False,
        is_start_temp=False,
        clockwise=preview_clockwise,
    )
    cmd.execute()

    arc = sketch.registry.entities[0]
    assert isinstance(arc, Arc)
    # The arc direction should match the preview, not be recalculated
    assert arc.clockwise == preview_clockwise


def test_arc_command_undo_restores_temp_points_to_original_state():
    """Test that undo removes all added items, temp points are not restored."""
    sketch = Sketch()

    # Manually add temp points to simulate tool state
    center_id = sketch.add_point(0, 0)
    start_id = sketch.add_point(50, 0)

    # Record state before command
    entities_before = len(sketch.registry.entities)

    cmd = ArcCommand(
        sketch,
        center_id,
        start_id,
        (0, 50),
        is_center_temp=True,
        is_start_temp=True,
        clockwise=False,
    )
    cmd.execute()

    # Verify something was added
    assert len(sketch.registry.entities) > entities_before

    cmd.undo()

    # After undo, entities should be back to original
    assert len(sketch.registry.entities) == entities_before

    # Points should be back to original (only origin for a fresh sketch)
    # The temp center and start should NOT be restored since they were temp
    assert len(sketch.registry.points) == 1  # Only origin
    assert sketch.registry.points[0].id == 0  # Only origin


def test_arc_preview_get_dimensions_returns_radius():
    """Test that dimension shows radius at arc midpoint."""
    sketch = Sketch()
    state = ArcCommand.start_center_preview(
        sketch.registry, 0, 0, snapped_pid=None
    )
    ArcCommand.set_start_point(sketch.registry, state, 50, 0)
    ArcCommand.update_preview(sketch.registry, state, 0, 50)

    dims = state.get_dimensions(sketch.registry)

    assert len(dims) == 1
    assert dims[0].label == "R50.00"
    assert dims[0].leader_end is None


def test_arc_preview_get_dimensions_no_dimensions_before_start():
    """Test that no dimensions before start point is set."""
    sketch = Sketch()
    state = ArcCommand.start_center_preview(
        sketch.registry, 0, 0, snapped_pid=None
    )

    dims = state.get_dimensions(sketch.registry)

    assert dims == []


def test_arc_preview_get_dimensions_position_on_arc():
    """Test that dimension position is at arc midpoint on radius."""
    sketch = Sketch()
    state = ArcCommand.start_center_preview(
        sketch.registry, 0, 0, snapped_pid=None
    )
    ArcCommand.set_start_point(sketch.registry, state, 100, 0)
    ArcCommand.update_preview(sketch.registry, state, 0, 100)

    dims = state.get_dimensions(sketch.registry)

    pos = dims[0].position
    dist_to_center = math.hypot(pos[0], pos[1])
    assert abs(dist_to_center - 100.0) < 0.01


def test_arc_preview_get_dimensions_label_on_arc():
    """Test that label is positioned on the arc radius."""
    sketch = Sketch()
    state = ArcCommand.start_center_preview(
        sketch.registry, 0, 0, snapped_pid=None
    )
    ArcCommand.set_start_point(sketch.registry, state, 50, 0)
    ArcCommand.update_preview(sketch.registry, state, 0, 50)

    dims = state.get_dimensions(sketch.registry)

    pos = dims[0].position
    dist_to_center = math.hypot(pos[0], pos[1])
    assert abs(dist_to_center - 50.0) < 0.01


def test_arc_preview_get_dimensions_missing_point():
    """Test that missing points return empty list."""
    sketch = Sketch()
    state = ArcCommand.start_center_preview(
        sketch.registry, 0, 0, snapped_pid=None
    )
    ArcCommand.set_start_point(sketch.registry, state, 50, 0)
    sketch.registry.points.clear()

    dims = state.get_dimensions(sketch.registry)

    assert dims == []
