from unittest.mock import MagicMock, call

import pytest

from rayforge.core.sketcher.commands import AddItemsCommand
from rayforge.core.sketcher.constraints import (
    EqualDistanceConstraint,
    EqualLengthConstraint,
    HorizontalConstraint,
    TangentConstraint,
    VerticalConstraint,
)
from rayforge.core.sketcher.entities import Arc, Line, Point
from rayforge.core.sketcher.tools import RoundedRectTool


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


def test_rounded_rect_tool_initialization(tool):
    """Test the tool's initial state."""
    assert tool.start_id is None
    assert tool.start_temp is False
    assert tool._is_previewing is False
    assert not tool._preview_ids


def test_first_click_no_hit_starts_preview(tool, mock_element):
    """Test that the first click on an empty space starts the preview mode."""
    mock_element.sketch.add_point.side_effect = [0, 1]  # start_id, p_end_id
    mock_element.hittester.screen_to_model.return_value = (10, 20)
    tool._update_preview_geometry = MagicMock()

    result = tool.on_press(100, 200, 1)

    assert result is True
    assert tool.start_id == 0
    assert tool.start_temp is True
    assert tool._is_previewing is True
    assert tool._preview_ids["p_end"] == 1
    tool._update_preview_geometry.assert_called_once_with(is_creation=True)
    mock_element.sketch.add_point.assert_has_calls(
        [call(10, 20), call(10, 20)]
    )


def test_first_click_with_hit_starts_preview(tool, mock_element):
    """Test a first click on an existing point starts the preview mode."""
    mock_element.hittester.get_hit_data.return_value = ("point", 5)
    mock_element.sketch.add_point.return_value = 6  # p_end_id
    mock_element.hittester.screen_to_model.return_value = (10, 20)
    tool._update_preview_geometry = MagicMock()

    result = tool.on_press(100, 200, 1)

    assert result is True
    assert tool.start_id == 5
    assert tool.start_temp is False
    assert tool._is_previewing is True
    assert tool._preview_ids["p_end"] == 6
    tool._update_preview_geometry.assert_called_once_with(is_creation=True)


def test_second_click_creates_rounded_rectangle(tool, mock_element):
    """Test a second click creates final rounded rectangle geometry."""
    # --- Setup first click state ---
    tool.start_id = 0
    tool.start_temp = True  # Important for checking point removal
    tool._is_previewing = True
    # Mock the removal of the temp start point
    mock_start_point = Point(0, 0, 0)
    mock_element.sketch.registry.get_point.return_value = mock_start_point
    mock_element.sketch.registry.points.remove = MagicMock()

    tool._cleanup_temps = MagicMock()

    # --- Simulate second click ---
    mock_element.hittester.screen_to_model.return_value = (100, 50)
    result = tool.on_press(100, 200, 1)

    assert result is True
    tool._cleanup_temps.assert_called_once()

    # Verify that the temp start point was removed from the registry
    mock_element.sketch.registry.points.remove.assert_called_once_with(
        mock_start_point
    )

    # Verify command execution
    mock_element.execute_command.assert_called_once()
    cmd = mock_element.execute_command.call_args[0][0]
    assert isinstance(cmd, AddItemsCommand)

    # Check command contents
    assert len(cmd.points) == 12  # 8 tangent points + 4 center points
    assert len(cmd.entities) == 8  # 4 lines + 4 arcs
    # 4 HV + 8 Tangent + 1 EqualLength + 4 EqualDistance = 17
    assert len(cmd.constraints) == 17
    assert sum(isinstance(e, Line) for e in cmd.entities) == 4
    assert sum(isinstance(e, Arc) for e in cmd.entities) == 4
    assert (
        sum(isinstance(c, HorizontalConstraint) for c in cmd.constraints) == 2
    )
    assert sum(isinstance(c, VerticalConstraint) for c in cmd.constraints) == 2
    assert sum(isinstance(c, TangentConstraint) for c in cmd.constraints) == 8
    assert (
        sum(isinstance(c, EqualLengthConstraint) for c in cmd.constraints) == 1
    )
    assert (
        sum(isinstance(c, EqualDistanceConstraint) for c in cmd.constraints)
        == 4
    )

    # Verify tool reset
    assert tool.start_id is None
    assert tool._is_previewing is False


def test_on_hover_motion_updates_preview(tool, mock_element):
    """Test that hovering updates the preview geometry."""
    # --- Setup preview state ---
    tool.start_id = 0
    tool._is_previewing = True
    tool._preview_ids["p_end"] = 1
    mock_point = Point(1, 0, 0)
    mock_element.sketch.registry.get_point.return_value = mock_point
    tool._update_preview_geometry = MagicMock()

    # --- Simulate hover ---
    mock_element.hittester.screen_to_model.return_value = (75, 85)
    tool.on_hover_motion(100, 200)

    assert mock_point.x == 75
    assert mock_point.y == 85
    tool._update_preview_geometry.assert_called_once()
    mock_element.mark_dirty.assert_called_once()


def test_on_deactivate_cleans_up(tool, mock_element):
    """Test that deactivating the tool cleans up temporary state."""
    tool.start_id = 0
    tool.start_temp = True
    tool._is_previewing = True
    tool._cleanup_temps = MagicMock()

    tool.on_deactivate()

    tool._cleanup_temps.assert_called_once()
    mock_element.remove_point_if_unused.assert_called_once_with(0)
    assert tool.start_id is None
    assert tool.start_temp is False


def test_degenerate_rounded_rectangle_aborts_creation(tool, mock_element):
    """Test that a zero-width or zero-height rect is not created."""
    # --- Setup first click state ---
    tool.start_id = 0
    tool.start_temp = True
    tool._is_previewing = True
    mock_start_point = Point(0, 10, 20)
    mock_element.sketch.registry.get_point.return_value = mock_start_point
    tool._cleanup_temps = MagicMock()

    # --- Simulate second click at nearly the same spot ---
    mock_element.hittester.screen_to_model.return_value = (10, 20.0000001)
    result = tool.on_press(100, 200, 1)

    assert result is True
    mock_element.execute_command.assert_not_called()
    # Verify tool reset and temp start point was cleaned up
    assert tool.start_id is None
    mock_element.remove_point_if_unused.assert_called_once_with(0)
