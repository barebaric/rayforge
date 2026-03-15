from rayforge.core.sketcher import Sketch
from rayforge.core.sketcher.commands import CircleCommand, CirclePreviewState
from rayforge.core.sketcher.entities import Circle, Point


def test_circle_command_execute_no_snap():
    """Test command execution with no point snapping."""
    sketch = Sketch()
    center_pid = sketch.add_point(0, 0)
    cmd = CircleCommand(sketch, center_pid, (100, 50))
    cmd.execute()

    assert len(sketch.registry.points) == 3
    assert len(sketch.registry.entities) == 1

    circle = sketch.registry.entities[0]
    assert isinstance(circle, Circle)
    assert circle.center_idx == center_pid


def test_circle_command_execute_with_snap():
    """Test command execution when snapping to an existing end point."""
    sketch = Sketch()
    center_pid = sketch.add_point(0, 0)
    radius_pid = sketch.add_point(100, 50)
    cmd = CircleCommand(sketch, center_pid, (100, 50), end_pid=radius_pid)
    cmd.execute()

    assert len(sketch.registry.points) == 3
    circle = sketch.registry.entities[0]
    assert isinstance(circle, Circle)
    assert circle.radius_pt_idx == radius_pid


def test_circle_command_execute_temp_center():
    """Test command execution when the center point was temporary."""
    sketch = Sketch()
    center_pid = 100
    sketch.registry.points.append(Point(center_pid, 0, 0))
    assert len(sketch.registry.points) == 2

    cmd = CircleCommand(sketch, center_pid, (100, 50), is_center_temp=True)
    cmd.execute()

    assert len(sketch.registry.points) == 3
    assert len(sketch.registry.entities) == 1
    assert cmd.add_cmd is not None

    re_added_center = next(
        p for p in cmd.add_cmd.points if p.x == 0 and p.y == 0
    )
    assert re_added_center.id != 100


def test_circle_command_execute_center_equals_radius():
    """Test command does nothing if center equals radius point."""
    sketch = Sketch()
    center_pid = sketch.add_point(0, 0)
    initial_point_count = len(sketch.registry.points)
    initial_entity_count = len(sketch.registry.entities)

    cmd = CircleCommand(sketch, center_pid, (0, 0), end_pid=center_pid)
    cmd.execute()

    assert len(sketch.registry.points) == initial_point_count
    assert len(sketch.registry.entities) == initial_entity_count


def test_circle_command_undo():
    """Test undo removes all added items."""
    sketch = Sketch()
    center_pid = sketch.add_point(0, 0)

    initial_point_count = len(sketch.registry.points)
    initial_entity_count = len(sketch.registry.entities)

    cmd = CircleCommand(sketch, center_pid, (100, 50))
    cmd.execute()

    assert len(sketch.registry.points) == initial_point_count + 1
    assert len(sketch.registry.entities) == initial_entity_count + 1

    cmd.undo()

    assert len(sketch.registry.points) == initial_point_count
    assert len(sketch.registry.entities) == initial_entity_count


def test_circle_command_undo_with_temp_center():
    """Test undo with temporary center point."""
    sketch = Sketch()
    center_pid = sketch.add_point(0, 0)

    cmd = CircleCommand(sketch, center_pid, (100, 50), is_center_temp=True)
    cmd.execute()

    assert len(sketch.registry.entities) == 1

    cmd.undo()

    assert len(sketch.registry.entities) == 0
    assert len(sketch.registry.points) == 1
    assert sketch.registry.points[0].id == 0


def test_circle_start_preview_no_snap():
    """Test start_preview creates initial preview state with temp point."""
    sketch = Sketch()
    state = CircleCommand.start_preview(
        sketch.registry, 10, 20, snapped_pid=None
    )

    assert isinstance(state, CirclePreviewState)
    assert state.center_temp is True
    assert state.radius_id is not None
    assert state.entity_id is not None

    center_p = sketch.registry.get_point(state.center_id)
    assert center_p.x == 10
    assert center_p.y == 20

    radius_p = sketch.registry.get_point(state.radius_id)
    assert radius_p.x == 10
    assert radius_p.y == 20


def test_circle_start_preview_with_snap():
    """Test start_preview uses existing point when snapped."""
    sketch = Sketch()
    existing_pid = sketch.add_point(50, 60)
    state = CircleCommand.start_preview(
        sketch.registry, 10, 20, snapped_pid=existing_pid
    )

    assert isinstance(state, CirclePreviewState)
    assert state.center_temp is False
    assert state.center_id == existing_pid


def test_circle_update_preview():
    """Test update_preview moves radius point."""
    sketch = Sketch()
    state = CircleCommand.start_preview(
        sketch.registry, 0, 0, snapped_pid=None
    )

    CircleCommand.update_preview(sketch.registry, state, 100, 50)

    radius_p = sketch.registry.get_point(state.radius_id)
    assert radius_p.x == 100
    assert radius_p.y == 50


def test_circle_cleanup_preview():
    """Test cleanup_preview removes preview entities but leaves center."""
    sketch = Sketch()
    initial_point_count = len(sketch.registry.points)
    initial_entity_count = len(sketch.registry.entities)

    state = CircleCommand.start_preview(
        sketch.registry, 0, 0, snapped_pid=None
    )

    assert len(sketch.registry.points) > initial_point_count
    assert len(sketch.registry.entities) > initial_entity_count

    CircleCommand.cleanup_preview(sketch.registry, state)

    assert len(sketch.registry.entities) == initial_entity_count
    remaining_preview_ids = {state.center_id}
    for p in sketch.registry.points:
        if p.id != 0:
            assert p.id in remaining_preview_ids


def test_circle_cleanup_preview_with_snapped_center():
    """Test cleanup when center point was snapped (not temp)."""
    sketch = Sketch()
    existing_pid = sketch.add_point(50, 50)
    initial_point_count = len(sketch.registry.points)

    state = CircleCommand.start_preview(
        sketch.registry, 0, 0, snapped_pid=existing_pid
    )

    CircleCommand.cleanup_preview(sketch.registry, state)

    assert len(sketch.registry.points) == initial_point_count


def test_circle_preview_lifecycle():
    """Test full preview lifecycle: start -> update -> cleanup."""
    sketch = Sketch()

    state = CircleCommand.start_preview(
        sketch.registry, 0, 0, snapped_pid=None
    )
    assert state.center_temp is True

    CircleCommand.update_preview(sketch.registry, state, 50, 50)
    radius_p = sketch.registry.get_point(state.radius_id)
    assert radius_p.x == 50
    assert radius_p.y == 50

    CircleCommand.cleanup_preview(sketch.registry, state)

    assert len(sketch.registry.entities) == 0
    assert state.center_id in [p.id for p in sketch.registry.points]


def test_circle_preview_then_execute():
    """Test executing a circle after preview."""
    sketch = Sketch()

    state = CircleCommand.start_preview(
        sketch.registry, 10, 20, snapped_pid=None
    )

    CircleCommand.update_preview(sketch.registry, state, 50, 30)

    center_id = state.center_id
    center_temp = state.center_temp

    CircleCommand.cleanup_preview(sketch.registry, state)

    cmd = CircleCommand(
        sketch, center_id, (50, 30), is_center_temp=center_temp
    )
    cmd.execute()

    assert len(sketch.registry.entities) == 1
    circle = sketch.registry.entities[0]
    assert isinstance(circle, Circle)
    assert cmd.add_cmd is not None
    center_point = next(
        (p for p in cmd.add_cmd.points if p.x == 10 and p.y == 20), None
    )
    assert center_point is not None
    assert circle.center_idx == center_point.id


def test_circle_undo_no_dangling_points():
    """Test undo removes all added points including temp radius."""
    sketch = Sketch()
    center_pid = sketch.add_point(0, 0)

    initial_point_count = len(sketch.registry.points)
    initial_entity_count = len(sketch.registry.entities)

    cmd = CircleCommand(sketch, center_pid, (100, 50))
    cmd.execute()

    assert len(sketch.registry.points) == initial_point_count + 1
    assert len(sketch.registry.entities) > initial_entity_count

    cmd.undo()

    assert len(sketch.registry.points) == initial_point_count
    assert len(sketch.registry.entities) == initial_entity_count
