import math
from sketcher.core import Sketch
from sketcher.core.commands import LineCommand, LinePreviewState
from sketcher.core.entities import Line, Point


def test_line_command_execute_no_snap():
    """Test command execution with no point snapping."""
    sketch = Sketch()
    start_pid = sketch.add_point(0, 0)
    cmd = LineCommand(sketch, start_pid, (100, 50))
    cmd.execute()

    assert len(sketch.registry.points) == 3
    assert len(sketch.registry.entities) == 1

    line = sketch.registry.entities[0]
    assert isinstance(line, Line)
    assert line.p1_idx == start_pid


def test_line_command_execute_with_snap():
    """Test command execution when snapping to an existing end point."""
    sketch = Sketch()
    start_pid = sketch.add_point(0, 0)
    end_pid = sketch.add_point(100, 50)
    cmd = LineCommand(sketch, start_pid, (100, 50), end_pid=end_pid)
    cmd.execute()

    assert len(sketch.registry.points) == 3
    line = sketch.registry.entities[0]
    assert isinstance(line, Line)
    assert line.p2_idx == end_pid


def test_line_command_execute_temp_start():
    """Test command execution when the start point was temporary."""
    sketch = Sketch()
    start_pid = 100
    sketch.registry.points.append(Point(start_pid, 0, 0))
    assert len(sketch.registry.points) == 2

    cmd = LineCommand(sketch, start_pid, (100, 50), is_start_temp=True)
    cmd.execute()

    assert len(sketch.registry.points) == 3
    assert len(sketch.registry.entities) == 1
    assert cmd.add_cmd is not None

    re_added_start = next(
        p for p in cmd.add_cmd.points if p.x == 0 and p.y == 0
    )
    assert re_added_start.id != 100


def test_line_command_execute_start_equals_end():
    """Test command does nothing if start equals end point."""
    sketch = Sketch()
    start_pid = sketch.add_point(0, 0)
    initial_point_count = len(sketch.registry.points)
    initial_entity_count = len(sketch.registry.entities)

    cmd = LineCommand(sketch, start_pid, (0, 0), end_pid=start_pid)
    cmd.execute()

    assert len(sketch.registry.points) == initial_point_count
    assert len(sketch.registry.entities) == initial_entity_count


def test_line_command_undo():
    """Test undo removes all added items."""
    sketch = Sketch()
    start_pid = sketch.add_point(0, 0)

    initial_point_count = len(sketch.registry.points)
    initial_entity_count = len(sketch.registry.entities)

    cmd = LineCommand(sketch, start_pid, (100, 50))
    cmd.execute()

    assert len(sketch.registry.points) == initial_point_count + 1
    assert len(sketch.registry.entities) == initial_entity_count + 1

    cmd.undo()

    assert len(sketch.registry.points) == initial_point_count
    assert len(sketch.registry.entities) == initial_entity_count


def test_line_command_undo_with_temp_start():
    """Test undo with temporary start point."""
    sketch = Sketch()
    start_pid = sketch.add_point(0, 0)

    cmd = LineCommand(sketch, start_pid, (100, 50), is_start_temp=True)
    cmd.execute()

    assert len(sketch.registry.entities) == 1

    cmd.undo()

    assert len(sketch.registry.entities) == 0
    assert len(sketch.registry.points) == 1
    assert sketch.registry.points[0].id == 0


def test_line_start_preview_no_snap():
    """Test start_preview creates initial preview state with temp point."""
    sketch = Sketch()
    state = LineCommand.start_preview(
        sketch.registry, 10, 20, snapped_pid=None
    )

    assert isinstance(state, LinePreviewState)
    assert state.start_temp is True
    assert state.end_id is not None
    assert state.entity_id is not None

    start_p = sketch.registry.get_point(state.start_id)
    assert start_p.x == 10
    assert start_p.y == 20

    end_p = sketch.registry.get_point(state.end_id)
    assert end_p.x == 10
    assert end_p.y == 20


def test_line_start_preview_with_snap():
    """Test start_preview uses existing point when snapped."""
    sketch = Sketch()
    existing_pid = sketch.add_point(50, 60)
    state = LineCommand.start_preview(
        sketch.registry, 10, 20, snapped_pid=existing_pid
    )

    assert isinstance(state, LinePreviewState)
    assert state.start_temp is False
    assert state.start_id == existing_pid


def test_line_update_preview():
    """Test update_preview moves end point."""
    sketch = Sketch()
    state = LineCommand.start_preview(sketch.registry, 0, 0, snapped_pid=None)

    LineCommand.update_preview(sketch.registry, state, 100, 50)

    end_p = sketch.registry.get_point(state.end_id)
    assert end_p.x == 100
    assert end_p.y == 50


def test_line_cleanup_preview():
    """Test cleanup_preview removes preview entities but leaves start."""
    sketch = Sketch()
    initial_point_count = len(sketch.registry.points)
    initial_entity_count = len(sketch.registry.entities)

    state = LineCommand.start_preview(sketch.registry, 0, 0, snapped_pid=None)

    assert len(sketch.registry.points) > initial_point_count
    assert len(sketch.registry.entities) > initial_entity_count

    LineCommand.cleanup_preview(sketch.registry, state)

    assert len(sketch.registry.entities) == initial_entity_count
    remaining_preview_ids = {state.start_id}
    for p in sketch.registry.points:
        if p.id != 0:
            assert p.id in remaining_preview_ids


def test_line_cleanup_preview_with_snapped_start():
    """Test cleanup when start point was snapped (not temp)."""
    sketch = Sketch()
    existing_pid = sketch.add_point(50, 50)
    initial_point_count = len(sketch.registry.points)

    state = LineCommand.start_preview(
        sketch.registry, 0, 0, snapped_pid=existing_pid
    )

    LineCommand.cleanup_preview(sketch.registry, state)

    assert len(sketch.registry.points) == initial_point_count


def test_line_preview_lifecycle():
    """Test full preview lifecycle: start -> update -> cleanup."""
    sketch = Sketch()

    state = LineCommand.start_preview(sketch.registry, 0, 0, snapped_pid=None)
    assert state.start_temp is True

    LineCommand.update_preview(sketch.registry, state, 50, 50)
    end_p = sketch.registry.get_point(state.end_id)
    assert end_p.x == 50
    assert end_p.y == 50

    LineCommand.cleanup_preview(sketch.registry, state)

    assert len(sketch.registry.entities) == 0
    assert state.start_id in [p.id for p in sketch.registry.points]


def test_line_preview_then_execute():
    """Test executing a line after preview."""
    sketch = Sketch()

    state = LineCommand.start_preview(
        sketch.registry, 10, 20, snapped_pid=None
    )

    LineCommand.update_preview(sketch.registry, state, 50, 30)

    start_id = state.start_id
    start_temp = state.start_temp

    LineCommand.cleanup_preview(sketch.registry, state)

    cmd = LineCommand(sketch, start_id, (50, 30), is_start_temp=start_temp)
    cmd.execute()

    assert len(sketch.registry.entities) == 1
    line = sketch.registry.entities[0]
    assert isinstance(line, Line)
    assert cmd.add_cmd is not None
    start_point = next(
        (p for p in cmd.add_cmd.points if p.x == 10 and p.y == 20), None
    )
    assert start_point is not None
    assert line.p1_idx == start_point.id


def test_line_undo_no_dangling_points():
    """Test undo removes all added points including temp end."""
    sketch = Sketch()
    start_pid = sketch.add_point(0, 0)

    initial_point_count = len(sketch.registry.points)
    initial_entity_count = len(sketch.registry.entities)

    cmd = LineCommand(sketch, start_pid, (100, 50))
    cmd.execute()

    assert len(sketch.registry.points) == initial_point_count + 1
    assert len(sketch.registry.entities) > initial_entity_count

    cmd.undo()

    assert len(sketch.registry.points) == initial_point_count
    assert len(sketch.registry.entities) == initial_entity_count


def test_line_preview_get_dimensions_returns_length_at_midpoint():
    """Test that dimension shows length at line midpoint."""
    sketch = Sketch()
    state = LineCommand.start_preview(sketch.registry, 0, 0, snapped_pid=None)
    LineCommand.update_preview(sketch.registry, state, 100, 0)

    dims = state.get_dimensions(sketch.registry)

    assert len(dims) == 1
    assert dims[0].label == "100.00"
    assert dims[0].position == (50.0, 0.0)
    assert dims[0].leader_end is None


def test_line_preview_get_dimensions_diagonal_line():
    """Test dimension for diagonal line."""
    sketch = Sketch()
    state = LineCommand.start_preview(sketch.registry, 0, 0, snapped_pid=None)
    LineCommand.update_preview(sketch.registry, state, 30, 40)

    dims = state.get_dimensions(sketch.registry)

    assert len(dims) == 1
    expected_length = math.hypot(30, 40)
    assert dims[0].label == f"{expected_length:.2f}"
    assert dims[0].position == (15.0, 20.0)


def test_line_preview_get_dimensions_very_small_length():
    """Test that very small lengths show as 0.00."""
    sketch = Sketch()
    state = LineCommand.start_preview(sketch.registry, 0, 0, snapped_pid=None)
    LineCommand.update_preview(sketch.registry, state, 0.005, 0)

    dims = state.get_dimensions(sketch.registry)

    assert len(dims) == 1
    assert dims[0].label == "0.00"


def test_line_preview_get_dimensions_missing_point():
    """Test that missing points return empty list."""
    sketch = Sketch()
    state = LineCommand.start_preview(sketch.registry, 0, 0, snapped_pid=None)
    sketch.registry.points.clear()

    dims = state.get_dimensions(sketch.registry)

    assert dims == []
