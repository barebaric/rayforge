from unittest.mock import MagicMock, patch

import pytest
from sketcher.core.commands import (
    RectangleCommand,
    RectanglePreviewState,
)
from sketcher.core.entities import Point
from sketcher.core.entities import Point as SketchPoint
from sketcher.core.snap import SnapLineType
from sketcher.ui_gtk.tools import RectangleTool


@pytest.fixture
def mock_element():
    """Create a mock SketchElement for testing tools."""
    element = MagicMock()
    element.sketch = MagicMock()
    element.sketch.registry._id_counter = 0
    element.hittester.get_hit_data.return_value = (None, None)
    element.execute_command = MagicMock()
    element.canvas = MagicMock()
    element.canvas.view_transform = MagicMock()
    element.canvas.view_transform.get_scale = MagicMock(
        return_value=(1.0, 1.0)
    )
    element.snap_engine = MagicMock()
    element.snap_engine.query = MagicMock(
        return_value=MagicMock(
            snapped=False,
            position=(0.0, 0.0),
            snap_lines=[],
            primary_snap_point=None,
        )
    )
    return element


@pytest.fixture
def rect_tool(mock_element):
    """Create a RectangleTool instance with a mocked element."""
    return RectangleTool(mock_element)


@pytest.mark.ui
def test_rectangle_tool_initialization(rect_tool):
    """Test tool's initial state."""
    assert rect_tool._preview_state is None
    assert rect_tool._ctrl_held is False
    assert rect_tool._press_world_pos is None
    assert rect_tool._in_press is False


@pytest.mark.ui
def test_first_click_no_hit_starts_preview(rect_tool, mock_element):
    """Test that first click on an empty space starts preview mode."""
    mock_element.sketch.registry.add_point.side_effect = [0, 1]
    mock_element.hittester.screen_to_model.return_value = (10, 20)

    with patch.object(
        RectangleCommand, "create_preview"
    ) as mock_create_preview:
        mock_create_preview.return_value = {"p2": 2, "line1": 10}
        result = rect_tool.on_press(100, 200, 1)

    assert result is True
    assert rect_tool._preview_state is not None
    assert rect_tool._preview_state.start_id == 0
    assert rect_tool._preview_state.start_temp is True
    assert rect_tool._preview_state.p_end_id == 1
    mock_element.sketch.registry.add_point.assert_called()


@pytest.mark.ui
def test_first_click_with_hit_starts_preview(rect_tool, mock_element):
    """
    Test that first click on an existing point starts preview mode.
    """
    mock_point = SketchPoint(5, 10, 20)
    mock_snap_point = MagicMock()
    mock_snap_point.line_type = SnapLineType.ENTITY_POINT
    mock_snap_point.source = mock_point
    mock_snap_point.x = 10.0
    mock_snap_point.y = 20.0
    mock_element.snap_engine.query.return_value = MagicMock(
        snapped=True,
        position=(10.0, 20.0),
        snap_lines=[],
        primary_snap_point=mock_snap_point,
    )
    mock_element.sketch.registry.add_point.return_value = 6
    mock_element.hittester.screen_to_model.return_value = (10, 20)

    with patch.object(
        RectangleCommand, "create_preview"
    ) as mock_create_preview:
        mock_create_preview.return_value = {"p2": 7, "line1": 10}
        result = rect_tool.on_press(100, 200, 1)

    assert result is True
    assert rect_tool._preview_state is not None
    assert rect_tool._preview_state.start_id == 5
    assert rect_tool._preview_state.start_temp is False
    assert rect_tool._preview_state.p_end_id == 6


@pytest.mark.ui
def test_second_click_no_hit_creates_rectangle(rect_tool, mock_element):
    """Test that second click creates final rectangle geometry."""
    # --- Setup first click state ---
    rect_tool._preview_state = RectanglePreviewState(
        start_id=0,
        start_temp=True,
        p_end_id=1,
        preview_ids={"p2": 2, "line1": 10},
    )
    mock_element.sketch.registry.get_point.side_effect = lambda pid: (
        Point(pid, 100, 50) if pid == 1 else Point(pid, 0, 0)
    )

    # --- Simulate second click ---
    mock_element.hittester.screen_to_model.return_value = (100, 50)
    result = rect_tool.on_press(100, 200, 1)

    assert result is True

    # Verify command execution
    mock_element.execute_command.assert_called_once()
    cmd = mock_element.execute_command.call_args[0][0]
    assert isinstance(cmd, RectangleCommand)

    # Verify command contents
    assert cmd.start_pid == 0
    assert cmd.end_pos == (100, 50)
    assert cmd.end_pid is None
    assert cmd.is_start_temp is True

    # Verify tool reset
    assert rect_tool._preview_state is None
    assert rect_tool._press_world_pos is None


@pytest.mark.ui
def test_second_click_with_hit_creates_rectangle(rect_tool, mock_element):
    """Test creating a rectangle by snapping to second corner a point."""
    # --- Setup first click state ---
    rect_tool._preview_state = RectanglePreviewState(
        start_id=0,
        start_temp=False,
        p_end_id=1,
        preview_ids={"p2": 2, "line1": 10},
    )
    mock_element.sketch.registry.get_point.side_effect = lambda pid: (
        Point(pid, 100, 50) if pid == 1 else Point(pid, 0, 0)
    )

    # --- Simulate second click ---
    mock_point = SketchPoint(7, 100, 50)
    mock_snap_point = MagicMock()
    mock_snap_point.line_type = SnapLineType.ENTITY_POINT
    mock_snap_point.source = mock_point
    mock_snap_point.x = 100.0
    mock_snap_point.y = 50.0
    mock_element.snap_engine.query.return_value = MagicMock(
        snapped=True,
        position=(100.0, 50.0),
        snap_lines=[],
        primary_snap_point=mock_snap_point,
    )
    mock_element.hittester.screen_to_model.return_value = (100, 50)
    result = rect_tool.on_press(100, 200, 1)

    assert result is True
    mock_element.execute_command.assert_called_once()
    cmd = mock_element.execute_command.call_args[0][0]
    assert isinstance(cmd, RectangleCommand)

    # Check command contents
    assert cmd.start_pid == 0
    assert cmd.end_pos == (100, 50)
    assert cmd.end_pid == 7
    assert cmd.is_start_temp is False


@pytest.mark.ui
def test_on_hover_motion_updates_preview(rect_tool, mock_element):
    """Test that hovering updates preview geometry."""
    rect_tool._preview_state = RectanglePreviewState(
        start_id=0,
        start_temp=True,
        p_end_id=1,
        preview_ids={"p2": 2, "line1": 10},
    )

    with patch.object(RectangleCommand, "update_preview") as mock_update:
        mock_element.hittester.screen_to_model.return_value = (75, 85)
        rect_tool.on_hover_motion(100, 200)

        mock_update.assert_called_once()
        call_args = mock_update.call_args
        assert call_args[0][2] == 75
        assert call_args[0][3] == 85
        mock_element.mark_dirty.assert_called_once()


@pytest.mark.ui
def test_on_deactivate_cleans_up(rect_tool, mock_element):
    """Test that deactivating tool cleans up temporary state."""
    rect_tool._preview_state = RectanglePreviewState(
        start_id=0,
        start_temp=True,
        p_end_id=1,
        preview_ids={"p2": 2, "line1": 10},
    )

    rect_tool.on_deactivate()

    assert rect_tool._preview_state is None
    mock_element.mark_dirty.assert_called()


@pytest.mark.ui
def test_degenerate_rectangle_aborts_creation(rect_tool, mock_element):
    """Test that a zero-width or zero-height rect is not created."""
    rect_tool._preview_state = RectanglePreviewState(
        start_id=0,
        start_temp=True,
        p_end_id=1,
        preview_ids={"p2": 2, "line1": 10},
    )
    mock_element.sketch.registry.get_point.return_value = Point(0, 10, 20)
    mock_element.sketch.remove_point_if_unused = MagicMock()

    # --- Simulate second click at nearly same spot ---
    mock_element.hittester.screen_to_model.return_value = (10, 20.0000001)
    result = rect_tool.on_press(100, 200, 1)

    assert result is True

    # The command should be executed...
    mock_element.execute_command.assert_called_once()
    cmd = mock_element.execute_command.call_args[0][0]
    assert isinstance(cmd, RectangleCommand)

    # ...but its internal logic should detect degenerate case and do
    # nothing except clean up temporary start point.
    cmd.sketch = mock_element.sketch
    cmd._do_execute()
    mock_element.sketch.remove_point_if_unused.assert_called_once_with(0)
    assert cmd.add_cmd is None


@pytest.mark.ui
def test_drag_release_creates_rectangle(rect_tool, mock_element):
    """Test that press-drag-release finishes the rectangle."""
    rect_tool._preview_state = RectanglePreviewState(
        start_id=0,
        start_temp=True,
        p_end_id=1,
        preview_ids={"p2": 2, "line1": 10},
    )
    rect_tool._press_world_pos = (100.0, 200.0)
    mock_element.sketch.registry.get_point.side_effect = lambda pid: (
        Point(pid, 100, 50) if pid == 1 else Point(pid, 0, 0)
    )

    rect_tool.on_release(140.0, 200.0)

    mock_element.execute_command.assert_called_once()
    cmd = mock_element.execute_command.call_args[0][0]
    assert cmd.end_pos == (100, 50)
    assert rect_tool._preview_state is None
    assert rect_tool._press_world_pos is None
    assert rect_tool._in_press is False


@pytest.mark.ui
def test_release_without_movement_keeps_preview(rect_tool, mock_element):
    """Test stray click does not commit and keeps the preview armed."""
    rect_tool._preview_state = RectanglePreviewState(
        start_id=0,
        start_temp=True,
        p_end_id=1,
        preview_ids={"p2": 2, "line1": 10},
    )
    rect_tool._press_world_pos = (100.0, 200.0)

    rect_tool.on_release(100.0, 200.0)

    mock_element.execute_command.assert_not_called()
    assert rect_tool._preview_state is not None
    assert rect_tool._press_world_pos == (100.0, 200.0)
    assert rect_tool._in_press is False


@pytest.mark.ui
def test_release_below_threshold_keeps_preview(rect_tool, mock_element):
    """Test release after sub-threshold jitter does not commit."""
    rect_tool._preview_state = RectanglePreviewState(
        start_id=0,
        start_temp=True,
        p_end_id=1,
        preview_ids={"p2": 2, "line1": 10},
    )
    rect_tool._press_world_pos = (100.0, 200.0)

    rect_tool.on_release(101.0, 200.5)

    mock_element.execute_command.assert_not_called()
    assert rect_tool._preview_state is not None


@pytest.mark.ui
def test_on_deactivate_clears_press_state(rect_tool, mock_element):
    """Test that deactivating resets the press state."""
    rect_tool._preview_state = RectanglePreviewState(
        start_id=0,
        start_temp=True,
        p_end_id=1,
        preview_ids={"p2": 2, "line1": 10},
    )
    rect_tool._press_world_pos = (100.0, 200.0)
    rect_tool._in_press = True

    rect_tool.on_deactivate()

    assert rect_tool._preview_state is None
    assert rect_tool._press_world_pos is None
    assert rect_tool._in_press is False


@pytest.mark.ui
def test_on_modifier_change_tracks_ctrl(rect_tool, mock_element):
    """Test that ctrl modifier state is tracked and triggers redraw."""
    rect_tool._preview_state = RectanglePreviewState(
        start_id=0,
        start_temp=True,
        p_end_id=1,
        preview_ids={"p2": 2, "line1": 10},
    )

    rect_tool.on_modifier_change(shift=False, ctrl=True)
    assert rect_tool._ctrl_held is True
    mock_element.mark_dirty.assert_called_once()

    rect_tool.on_modifier_change(shift=False, ctrl=True)
    mock_element.mark_dirty.assert_called_once()


@pytest.mark.ui
def test_ctrl_constrain_square_passed_to_command(rect_tool, mock_element):
    """Test that a ctrl-constrained preview commits as a square."""
    rect_tool._preview_state = RectanglePreviewState(
        start_id=0,
        start_temp=True,
        p_end_id=1,
        preview_ids={"p2": 2, "line1": 10},
    )
    rect_tool._preview_state.constrain_square = True
    rect_tool._ctrl_held = True
    mock_element.hittester.screen_to_model.return_value = (100, 50)
    mock_element.sketch.registry.get_point.side_effect = lambda pid: (
        Point(pid, 100, 50) if pid == 1 else Point(pid, 0, 0)
    )

    rect_tool.on_press(100, 200, 1)

    cmd = mock_element.execute_command.call_args[0][0]
    assert cmd.constrain_square is True


@pytest.mark.ui
def test_ctrl_updates_preview_square(rect_tool, mock_element):
    """Test that hovering with ctrl squashes the preview to a square."""
    rect_tool._preview_state = RectanglePreviewState(
        start_id=0,
        start_temp=True,
        p_end_id=1,
        preview_ids={"p2": 2, "line1": 10},
    )
    rect_tool._ctrl_held = True
    mock_element.hittester.screen_to_model.return_value = (75, 85)

    with patch.object(RectangleCommand, "update_preview") as mock_update:
        rect_tool.on_hover_motion(100, 200)

    assert mock_update.call_args[1]["constrain_square"] is True


@pytest.mark.ui
def test_commit_uses_preview_mode_not_live_modifier(rect_tool, mock_element):
    """Test commit matches the preview even if modifiers changed after."""
    rect_tool._preview_state = RectanglePreviewState(
        start_id=0,
        start_temp=True,
        p_end_id=1,
        preview_ids={"p2": 2, "line1": 10},
        center_on_start=True,
    )
    rect_tool._shift_held = False
    mock_element.hittester.screen_to_model.return_value = (100, 50)
    mock_element.sketch.registry.get_point.side_effect = lambda pid: (
        Point(pid, 100, 50) if pid == 1 else Point(pid, 0, 0)
    )

    rect_tool.on_press(100, 200, 1)

    cmd = mock_element.execute_command.call_args[0][0]
    assert cmd.center_on_start is True
    assert cmd.end_pos == (100, 50)


@pytest.mark.ui
def test_get_active_shortcuts_during_preview(rect_tool):
    """Test status bar shortcuts advertise ctrl and the click phase."""
    rect_tool._preview_state = RectanglePreviewState(
        start_id=0,
        start_temp=True,
        p_end_id=1,
        preview_ids={"p2": 2, "line1": 10},
    )

    shortcuts = rect_tool.get_active_shortcuts()
    keys = [s[0] for s in shortcuts]
    assert "Shift" in keys
    assert "Ctrl" in keys
    assert "Click" in keys

    rect_tool._in_press = True
    shortcuts = rect_tool.get_active_shortcuts()
    keys = [s[0] for s in shortcuts]
    assert "Click" not in keys
