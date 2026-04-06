from unittest.mock import MagicMock, patch

import pytest

from sketcher.core.commands import (
    RoundedRectCommand,
    RoundedRectPreviewState,
)
from sketcher.core.entities import Point
from sketcher.ui_gtk.tools import RoundedRectTool


@pytest.fixture
def mock_element():
    """Create a mock SketchElement for testing tools."""
    element = MagicMock()
    element.sketch = MagicMock()
    element.sketch.registry._id_counter = 0
    element.hittester.get_hit_data.return_value = (None, None)
    element.execute_command = MagicMock()
    return element


@pytest.fixture
def tool(mock_element):
    """Create a RoundedRectTool instance with a mocked element."""
    return RoundedRectTool(mock_element)


@pytest.mark.ui
def test_rounded_rect_tool_initialization(tool):
    """Test tool's initial state."""
    assert tool._preview_state is None


@pytest.mark.ui
def test_first_click_no_hit_starts_preview(tool, mock_element):
    """Test that first click on an empty space starts preview mode."""
    mock_element.sketch.registry.add_point.side_effect = [0, 1]
    mock_element.hittester.screen_to_model.return_value = (10, 20)

    with patch.object(
        RoundedRectCommand, "create_preview"
    ) as mock_create_preview:
        mock_create_preview.return_value = {"t2": 2, "line1": 10}
        result = tool.on_press(100, 200, 1)

    assert result is True
    assert tool._preview_state is not None
    assert tool._preview_state.start_id == 0
    assert tool._preview_state.start_temp is True
    assert tool._preview_state.p_end_id == 1
    mock_element.sketch.registry.add_point.assert_called()


@pytest.mark.ui
def test_first_click_with_hit_starts_preview(tool, mock_element):
    """Test a first click on an existing point starts preview mode."""
    mock_element.hittester.get_hit_data.return_value = ("point", 5)
    mock_element.sketch.registry.add_point.return_value = 6
    mock_element.hittester.screen_to_model.return_value = (10, 20)

    with patch.object(
        RoundedRectCommand, "create_preview"
    ) as mock_create_preview:
        mock_create_preview.return_value = {"t2": 7, "line1": 10}
        result = tool.on_press(100, 200, 1)

    assert result is True
    assert tool._preview_state is not None
    assert tool._preview_state.start_id == 5
    assert tool._preview_state.start_temp is False
    assert tool._preview_state.p_end_id == 6


@pytest.mark.ui
def test_second_click_creates_rounded_rectangle(tool, mock_element):
    """Test a second click creates final rounded rectangle geometry."""
    tool._preview_state = RoundedRectPreviewState(
        start_id=0,
        start_temp=True,
        p_end_id=1,
        preview_ids={"t2": 2, "line1": 10},
        radius=10.0,
    )
    mock_element.sketch.registry.get_point.return_value = Point(0, 0, 0)

    mock_element.hittester.screen_to_model.return_value = (100, 50)
    result = tool.on_press(100, 200, 1)

    assert result is True
    mock_element.execute_command.assert_called_once()
    cmd = mock_element.execute_command.call_args[0][0]
    assert isinstance(cmd, RoundedRectCommand)
    assert cmd.start_pid == 0
    assert cmd.end_pos == (100, 50)
    assert cmd.is_start_temp is True
    assert cmd.radius == tool.DEFAULT_RADIUS
    assert tool._preview_state is None


@pytest.mark.ui
def test_on_hover_motion_updates_preview(tool, mock_element):
    """Test that hovering updates preview geometry."""
    tool._preview_state = RoundedRectPreviewState(
        start_id=0,
        start_temp=True,
        p_end_id=1,
        preview_ids={"t2": 2, "line1": 10},
        radius=10.0,
    )

    with patch.object(RoundedRectCommand, "update_preview") as mock_update:
        mock_element.hittester.screen_to_model.return_value = (75, 85)
        tool.on_hover_motion(100, 200)

        mock_update.assert_called_once()
        call_args = mock_update.call_args
        assert call_args[0][2] == 75
        assert call_args[0][3] == 85
        mock_element.mark_dirty.assert_called_once()


@pytest.mark.ui
def test_on_deactivate_cleans_up(tool, mock_element):
    """Test that deactivating tool cleans up temporary state."""
    tool._preview_state = RoundedRectPreviewState(
        start_id=0,
        start_temp=True,
        p_end_id=1,
        preview_ids={"t2": 2, "line1": 10},
        radius=10.0,
    )

    tool.on_deactivate()

    assert tool._preview_state is None
    mock_element.mark_dirty.assert_called()


@pytest.mark.ui
def test_degenerate_rounded_rectangle_aborts_creation(tool, mock_element):
    """Test that a zero-width or zero-height rect is not created."""
    # --- Setup first click state ---
    tool._preview_state = RoundedRectPreviewState(
        start_id=0,
        start_temp=True,
        p_end_id=1,
        preview_ids={"t2": 2, "line1": 10},
        radius=10.0,
    )
    mock_element.sketch.registry.get_point.return_value = Point(0, 10, 20)
    mock_element.sketch.remove_point_if_unused = MagicMock()

    # --- Simulate second click at nearly same spot ---
    mock_element.hittester.screen_to_model.return_value = (10, 20.0000001)
    result = tool.on_press(100, 200, 1)

    assert result is True
    # The command should be executed...
    mock_element.execute_command.assert_called_once()
    cmd = mock_element.execute_command.call_args[0][0]
    assert isinstance(cmd, RoundedRectCommand)

    # ...but its internal logic should detect degenerate case and do
    # nothing except clean up temporary start point.
    cmd.sketch = mock_element.sketch
    cmd._do_execute()
    mock_element.sketch.remove_point_if_unused.assert_called_once_with(0)
    assert cmd.add_cmd is None
