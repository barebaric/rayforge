from sketcher.core import Sketch
from sketcher.core.commands import EllipseCommand, EllipsePreviewState
from sketcher.core.constraints import EqualDistanceConstraint
from sketcher.core.entities import Ellipse, Point, Line


def test_ellipse_command_execute_no_snap():
    """Test command execution with no point snapping."""
    sketch = Sketch()
    start_pid = sketch.add_point(0, 0)
    cmd = EllipseCommand(sketch, start_pid, (100, 50))
    cmd.execute()

    assert len(sketch.registry.entities) == 5
    assert len(sketch.registry.points) == 5

    ellipse = next(
        e for e in sketch.registry.entities if isinstance(e, Ellipse)
    )
    assert ellipse is not None


def test_ellipse_command_execute_with_snap():
    """Test command execution when snapping to an existing end point."""
    sketch = Sketch()
    start_pid = sketch.add_point(0, 0)
    end_pid = sketch.add_point(100, 50)
    cmd = EllipseCommand(sketch, start_pid, (100, 50), end_pid=end_pid)
    cmd.execute()

    ellipse = next(
        e for e in sketch.registry.entities if isinstance(e, Ellipse)
    )
    assert ellipse is not None


def test_ellipse_command_execute_temp_start():
    """Test command execution when the start point was temporary."""
    sketch = Sketch()
    start_pid = 100
    sketch.registry.points.append(Point(start_pid, 0, 0))

    cmd = EllipseCommand(sketch, start_pid, (100, 50), is_start_temp=True)
    cmd.execute()

    assert len(sketch.registry.entities) == 5
    assert cmd.add_cmd is not None


def test_ellipse_command_execute_center_on_start():
    """Test command with center_on_start=True."""
    sketch = Sketch()
    start_pid = sketch.add_point(50, 50)
    cmd = EllipseCommand(sketch, start_pid, (100, 80), center_on_start=True)
    cmd.execute()

    ellipse = next(
        e for e in sketch.registry.entities if isinstance(e, Ellipse)
    )
    center = sketch.registry.get_point(ellipse.center_idx)
    assert center.x == 50.0
    assert center.y == 50.0


def test_ellipse_command_execute_constrain_circle():
    """Test command with constrain_circle=True."""
    sketch = Sketch()
    start_pid = sketch.add_point(0, 0)
    cmd = EllipseCommand(sketch, start_pid, (100, 50), constrain_circle=True)
    cmd.execute()

    ellipse = next(
        e for e in sketch.registry.entities if isinstance(e, Ellipse)
    )
    rx, ry = ellipse._get_radii(sketch.registry)
    assert rx == ry
    assert rx == 25.0


def test_ellipse_command_execute_center_on_start_constrain_circle():
    """Test command with both center_on_start and constrain_circle."""
    sketch = Sketch()
    start_pid = sketch.add_point(50, 50)
    cmd = EllipseCommand(
        sketch,
        start_pid,
        (100, 80),
        center_on_start=True,
        constrain_circle=True,
    )
    cmd.execute()

    ellipse = next(
        e for e in sketch.registry.entities if isinstance(e, Ellipse)
    )
    center = sketch.registry.get_point(ellipse.center_idx)
    assert center.x == 50.0
    assert center.y == 50.0
    rx, ry = ellipse._get_radii(sketch.registry)
    assert rx == ry


def test_ellipse_command_execute_zero_radii():
    """Test command does nothing if radii would be zero."""
    sketch = Sketch()
    start_pid = sketch.add_point(0, 0)
    initial_entity_count = len(sketch.registry.entities)

    cmd = EllipseCommand(sketch, start_pid, (0, 0))
    cmd.execute()

    assert len(sketch.registry.entities) == initial_entity_count


def test_ellipse_command_undo():
    """Test undo removes all added items."""
    sketch = Sketch()
    start_pid = sketch.add_point(0, 0)

    initial_entity_count = len(sketch.registry.entities)

    cmd = EllipseCommand(sketch, start_pid, (100, 50))
    cmd.execute()

    assert len(sketch.registry.entities) > initial_entity_count

    cmd.undo()

    assert len(sketch.registry.entities) == initial_entity_count


def test_ellipse_command_undo_with_temp_start():
    """Test undo with temporary start point."""
    sketch = Sketch()
    start_pid = sketch.add_point(0, 0)

    cmd = EllipseCommand(sketch, start_pid, (100, 50), is_start_temp=True)
    cmd.execute()

    assert len(sketch.registry.entities) == 5

    cmd.undo()

    assert len(sketch.registry.entities) == 0


def test_calculate_ellipse_params_default():
    """Test _calculate_ellipse_params with default mode."""
    cx, cy, rx, ry = EllipseCommand._calculate_ellipse_params(
        0, 0, 100, 50, center_on_start=False, constrain_circle=False
    )
    assert cx == 50.0
    assert cy == 25.0
    assert rx == 50.0
    assert ry == 25.0


def test_calculate_ellipse_params_center_on_start():
    """Test _calculate_ellipse_params with center_on_start=True."""
    cx, cy, rx, ry = EllipseCommand._calculate_ellipse_params(
        50, 50, 100, 80, center_on_start=True, constrain_circle=False
    )
    assert cx == 50.0
    assert cy == 50.0
    assert rx == 50.0
    assert ry == 30.0


def test_calculate_ellipse_params_constrain_circle():
    """Test _calculate_ellipse_params with constrain_circle=True."""
    cx, cy, rx, ry = EllipseCommand._calculate_ellipse_params(
        0, 0, 100, 50, center_on_start=False, constrain_circle=True
    )
    assert rx == ry
    assert rx == 25.0
    assert cx == 25.0
    assert cy == 25.0


def test_calculate_ellipse_params_constrain_circle_negative_direction():
    """Test constrain_circle with negative drag direction."""
    cx, cy, rx, ry = EllipseCommand._calculate_ellipse_params(
        100, 100, 0, 50, center_on_start=False, constrain_circle=True
    )
    assert rx == ry
    assert rx == 25.0


def test_calculate_ellipse_params_center_on_start_constrain_circle():
    """Test _calculate_ellipse_params with both modifiers."""
    cx, cy, rx, ry = EllipseCommand._calculate_ellipse_params(
        50, 50, 100, 30, center_on_start=True, constrain_circle=True
    )
    assert cx == 50.0
    assert cy == 50.0
    assert rx == ry
    assert rx == 20.0


def test_ellipse_start_preview_no_snap():
    """Test start_preview creates initial preview state with temp point."""
    sketch = Sketch()
    state = EllipseCommand.start_preview(
        sketch.registry, 10, 20, snapped_pid=None
    )

    assert isinstance(state, EllipsePreviewState)
    assert state.start_temp is True
    assert state.center_id is not None
    assert state.radius_x_id is not None
    assert state.radius_y_id is not None
    assert state.entity_id is not None

    center_p = sketch.registry.get_point(state.center_id)
    assert center_p.x == 10
    assert center_p.y == 20


def test_ellipse_start_preview_with_snap():
    """Test start_preview uses existing point when snapped."""
    sketch = Sketch()
    existing_pid = sketch.add_point(50, 60)
    state = EllipseCommand.start_preview(
        sketch.registry, 10, 20, snapped_pid=existing_pid
    )

    assert isinstance(state, EllipsePreviewState)
    assert state.start_temp is False
    assert state.start_id == existing_pid


def test_ellipse_update_preview():
    """Test update_preview moves ellipse points."""
    sketch = Sketch()
    state = EllipseCommand.start_preview(
        sketch.registry, 0, 0, snapped_pid=None
    )

    EllipseCommand.update_preview(
        sketch.registry,
        state,
        100,
        50,
        center_on_start=False,
        constrain_circle=False,
    )

    center_p = sketch.registry.get_point(state.center_id)
    radius_x_p = sketch.registry.get_point(state.radius_x_id)
    radius_y_p = sketch.registry.get_point(state.radius_y_id)

    assert center_p.x == 50.0
    assert center_p.y == 25.0
    assert radius_x_p.x == 100.0
    assert radius_y_p.y == 50.0


def test_ellipse_update_preview_center_on_start():
    """Test update_preview with center_on_start=True."""
    sketch = Sketch()
    state = EllipseCommand.start_preview(
        sketch.registry, 50, 50, snapped_pid=None
    )

    EllipseCommand.update_preview(
        sketch.registry,
        state,
        100,
        80,
        center_on_start=True,
        constrain_circle=False,
    )

    center_p = sketch.registry.get_point(state.center_id)
    assert center_p.x == 50.0
    assert center_p.y == 50.0


def test_ellipse_update_preview_constrain_circle():
    """Test update_preview with constrain_circle=True."""
    sketch = Sketch()
    state = EllipseCommand.start_preview(
        sketch.registry, 0, 0, snapped_pid=None
    )

    EllipseCommand.update_preview(
        sketch.registry,
        state,
        100,
        50,
        center_on_start=False,
        constrain_circle=True,
    )

    center_p = sketch.registry.get_point(state.center_id)
    radius_x_p = sketch.registry.get_point(state.radius_x_id)
    radius_y_p = sketch.registry.get_point(state.radius_y_id)

    rx = abs(radius_x_p.x - center_p.x)
    ry = abs(radius_y_p.y - center_p.y)
    assert rx == ry


def test_ellipse_cleanup_preview():
    """Test cleanup_preview removes preview entities."""
    sketch = Sketch()
    initial_entity_count = len(sketch.registry.entities)

    state = EllipseCommand.start_preview(
        sketch.registry, 0, 0, snapped_pid=None
    )

    assert len(sketch.registry.entities) > initial_entity_count

    EllipseCommand.cleanup_preview(sketch.registry, state)

    assert len(sketch.registry.entities) == initial_entity_count


def test_ellipse_cleanup_preview_with_snapped_start():
    """Test cleanup when start point was snapped (not temp)."""
    sketch = Sketch()
    existing_pid = sketch.add_point(50, 50)

    state = EllipseCommand.start_preview(
        sketch.registry, 0, 0, snapped_pid=existing_pid
    )

    EllipseCommand.cleanup_preview(sketch.registry, state)

    assert existing_pid in [p.id for p in sketch.registry.points]


def test_ellipse_preview_lifecycle():
    """Test full preview lifecycle: start -> update -> cleanup."""
    sketch = Sketch()

    state = EllipseCommand.start_preview(
        sketch.registry, 0, 0, snapped_pid=None
    )
    assert state.start_temp is True

    EllipseCommand.update_preview(
        sketch.registry,
        state,
        50,
        50,
        center_on_start=False,
        constrain_circle=False,
    )
    center_p = sketch.registry.get_point(state.center_id)
    assert center_p.x == 25.0
    assert center_p.y == 25.0

    EllipseCommand.cleanup_preview(sketch.registry, state)

    assert (
        len([e for e in sketch.registry.entities if isinstance(e, Ellipse)])
        == 0
    )


def test_ellipse_preview_get_preview_point_ids():
    """Test EllipsePreviewState.get_preview_point_ids."""
    state = EllipsePreviewState(
        start_id=1,
        start_temp=True,
        center_id=2,
        radius_x_id=3,
        radius_y_id=4,
        entity_id=5,
    )
    assert state.get_preview_point_ids() == {2, 3, 4}


def test_ellipse_preview_get_hidden_point_ids():
    """Test EllipsePreviewState.get_hidden_point_ids."""
    state = EllipsePreviewState(
        start_id=1,
        start_temp=True,
        center_id=2,
        radius_x_id=3,
        radius_y_id=4,
        entity_id=5,
    )
    assert state.get_hidden_point_ids() == {1}


def test_ellipse_creates_helper_lines():
    """Test that ellipse creation creates helper construction lines."""
    sketch = Sketch()
    start_pid = sketch.add_point(0, 0)
    cmd = EllipseCommand(sketch, start_pid, (100, 50))
    cmd.execute()

    ellipse = next(
        e for e in sketch.registry.entities if isinstance(e, Ellipse)
    )
    assert len(ellipse.helper_line_ids) == 2

    lines = [e for e in sketch.registry.entities if isinstance(e, Line)]
    assert len(lines) == 4

    construction_lines = [ln for ln in lines if ln.construction]
    assert len(construction_lines) == 2

    invisible_lines = [ln for ln in lines if ln.invisible]
    assert len(invisible_lines) == 2


def test_ellipse_creates_perpendicular_constraint():
    """Test that ellipse creation creates perpendicular constraint."""
    sketch = Sketch()
    start_pid = sketch.add_point(0, 0)
    cmd = EllipseCommand(sketch, start_pid, (100, 50))
    cmd.execute()

    assert len(sketch.constraints) == 1


def test_ellipse_constrain_circle_creates_equal_distance():
    """Test that constrain_circle adds an EqualDistanceConstraint on radii."""
    sketch = Sketch()
    start_pid = sketch.add_point(0, 0)
    cmd = EllipseCommand(sketch, start_pid, (100, 50), constrain_circle=True)
    cmd.execute()

    assert len(sketch.constraints) == 2
    equal_constr = next(
        c for c in sketch.constraints if isinstance(c, EqualDistanceConstraint)
    )
    assert equal_constr is not None
    ellipse = next(
        e for e in sketch.registry.entities if isinstance(e, Ellipse)
    )
    assert equal_constr.p1 == ellipse.center_idx
    assert equal_constr.p2 == ellipse.radius_x_pt_idx
    assert equal_constr.p3 == ellipse.center_idx
    assert equal_constr.p4 == ellipse.radius_y_pt_idx


def test_ellipse_no_constrain_circle_no_equal_distance():
    """Test that without constrain_circle, no EqualDistanceConstraint."""
    sketch = Sketch()
    start_pid = sketch.add_point(0, 0)
    cmd = EllipseCommand(sketch, start_pid, (100, 50))
    cmd.execute()

    assert not any(
        isinstance(c, EqualDistanceConstraint) for c in sketch.constraints
    )
