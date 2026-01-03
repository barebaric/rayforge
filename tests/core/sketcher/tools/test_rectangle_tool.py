from unittest.mock import MagicMock, call
from typing import cast
import pytest

from rayforge.core.sketcher.commands import AddItemsCommand
from rayforge.core.sketcher.constraints import (
    HorizontalConstraint,
    VerticalConstraint,
)
from rayforge.core.sketcher.entities import Line, Point
from rayforge.core.sketcher.tools import RectangleTool


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
def rect_tool(mock_element):
    """Create a RectangleTool instance with a mocked element."""
    return RectangleTool(mock_element)


def test_rectangle_tool_initialization(rect_tool):
    """Test the tool's initial state."""
    assert rect_tool.start_id is None
    assert rect_tool.start_temp is False
    assert rect_tool._is_previewing is False
    assert not rect_tool._preview_ids


def test_first_click_no_hit_starts_preview(rect_tool, mock_element):
    """Test that the first click on an empty space starts the preview mode."""
    mock_element.sketch.add_point.side_effect = [0, 1]  # start_id, p_end_id
    mock_element.hittester.screen_to_model.return_value = (10, 20)
    rect_tool._update_preview_geometry = MagicMock()

    result = rect_tool.on_press(100, 200, 1)

    assert result is True
    assert rect_tool.start_id == 0
    assert rect_tool.start_temp is True
    assert rect_tool._is_previewing is True
    assert rect_tool._preview_ids["p_end"] == 1
    rect_tool._update_preview_geometry.assert_called_once_with(
        is_creation=True
    )
    mock_element.sketch.add_point.assert_has_calls(
        [call(10, 20), call(10, 20)]
    )


def test_first_click_with_hit_starts_preview(rect_tool, mock_element):
    """
    Test that the first click on an existing point starts the preview mode.
    """
    mock_element.hittester.get_hit_data.return_value = ("point", 5)
    mock_element.sketch.add_point.return_value = 6  # p_end_id
    mock_element.hittester.screen_to_model.return_value = (10, 20)
    rect_tool._update_preview_geometry = MagicMock()

    result = rect_tool.on_press(100, 200, 1)

    assert result is True
    assert rect_tool.start_id == 5
    assert rect_tool.start_temp is False
    assert rect_tool._is_previewing is True
    assert rect_tool._preview_ids["p_end"] == 6
    rect_tool._update_preview_geometry.assert_called_once_with(
        is_creation=True
    )


def test_second_click_no_hit_creates_rectangle(rect_tool, mock_element):
    """Test that the second click creates the final rectangle geometry."""
    # --- Setup first click state ---
    rect_tool.start_id = 0
    rect_tool.start_temp = True
    rect_tool._is_previewing = True
    mock_element.sketch.registry.get_point.return_value = Point(0, 0, 0)
    rect_tool._cleanup_temps = MagicMock()

    # --- Simulate second click ---
    mock_element.hittester.screen_to_model.return_value = (100, 50)
    result = rect_tool.on_press(100, 200, 1)

    assert result is True
    rect_tool._cleanup_temps.assert_called_once()

    # Verify command execution
    mock_element.execute_command.assert_called_once()
    cmd = mock_element.execute_command.call_args[0][0]
    assert isinstance(cmd, AddItemsCommand)

    # Check command contents
    assert len(cmd.points) == 4  # p1(temp), p2, p3, p4
    assert len(cmd.entities) == 4
    assert len(cmd.constraints) == 4
    assert all(isinstance(e, Line) for e in cmd.entities)
    assert (
        sum(isinstance(c, HorizontalConstraint) for c in cmd.constraints) == 2
    )
    assert sum(isinstance(c, VerticalConstraint) for c in cmd.constraints) == 2

    # Verify tool reset
    assert rect_tool.start_id is None
    assert rect_tool._is_previewing is False


def test_second_click_with_hit_creates_rectangle(rect_tool, mock_element):
    """Test creating a rectangle by snapping the second corner to a point."""
    # --- Setup first click state ---
    rect_tool.start_id = 0
    rect_tool.start_temp = False
    rect_tool._is_previewing = True
    mock_element.sketch.registry.get_point.side_effect = [
        Point(0, 0, 0),  # start_p
        Point(7, 100, 50),  # final_p
    ]
    rect_tool._cleanup_temps = MagicMock()

    # --- Simulate second click ---
    mock_element.hittester.get_hit_data.return_value = ("point", 7)
    mock_element.hittester.screen_to_model.return_value = (100, 50)
    result = rect_tool.on_press(100, 200, 1)

    assert result is True
    rect_tool._cleanup_temps.assert_called_once()

    # Verify command execution
    mock_element.execute_command.assert_called_once()
    cmd = mock_element.execute_command.call_args[0][0]
    assert isinstance(cmd, AddItemsCommand)

    # Check command contents (only 2 new points, start and end already exist)
    assert len(cmd.points) == 2  # p2, p4
    line_2 = cast(Line, cmd.entities[1])
    assert line_2.p2_idx == 7  # Line 2 should connect to p3 (ID 7)
    assert len(cmd.entities) == 4
    assert len(cmd.constraints) == 4


def test_on_hover_motion_updates_preview(rect_tool, mock_element):
    """Test that hovering updates the preview geometry."""
    # --- Setup preview state ---
    rect_tool.start_id = 0
    rect_tool._is_previewing = True
    rect_tool._preview_ids["p_end"] = 1
    mock_point = Point(1, 0, 0)
    mock_element.sketch.registry.get_point.return_value = mock_point
    rect_tool._update_preview_geometry = MagicMock()

    # --- Simulate hover ---
    mock_element.hittester.screen_to_model.return_value = (75, 85)
    rect_tool.on_hover_motion(100, 200)

    assert mock_point.x == 75
    assert mock_point.y == 85
    rect_tool._update_preview_geometry.assert_called_once()
    mock_element.mark_dirty.assert_called_once()


def test_on_deactivate_cleans_up(rect_tool, mock_element):
    """Test that deactivating the tool cleans up temporary state."""
    rect_tool.start_id = 0
    rect_tool.start_temp = True
    rect_tool._is_previewing = True
    rect_tool._cleanup_temps = MagicMock()

    rect_tool.on_deactivate()

    rect_tool._cleanup_temps.assert_called_once()
    mock_element.remove_point_if_unused.assert_called_once_with(0)
    assert rect_tool.start_id is None
    assert rect_tool.start_temp is False


def test_degenerate_rectangle_aborts_creation(rect_tool, mock_element):
    """Test that a zero-width or zero-height rect is not created."""
    # --- Setup first click state ---
    rect_tool.start_id = 0
    rect_tool.start_temp = True
    rect_tool._is_previewing = True
    mock_element.sketch.registry.get_point.return_value = Point(0, 10, 20)
    rect_tool._cleanup_temps = MagicMock()

    # --- Simulate second click at nearly the same spot ---
    mock_element.hittester.screen_to_model.return_value = (10, 20.0000001)
    result = rect_tool.on_press(100, 200, 1)

    assert result is True
    mock_element.execute_command.assert_not_called()
    # Verify tool reset and temp start point was cleaned up
    assert rect_tool.start_id is None
    mock_element.remove_point_if_unused.assert_called_once_with(0)
