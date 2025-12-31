import pytest
from unittest.mock import Mock
from rayforge.core.sketcher.tools.line_tool import LineTool


@pytest.fixture
def mock_element():
    """Create a mock SketchElement for testing."""
    element = Mock()
    element.sketch = Mock()
    element.sketch.registry = Mock()
    element.sketch.registry.points = []
    element.sketch.registry.entities = []
    element.sketch.registry._id_counter = 0
    element.sketch.add_point = Mock(return_value=0)
    element.selection = Mock()
    element.selection.clear = Mock()
    element.selection.select_point = Mock()
    element.hittester = Mock()
    element.hittester.screen_to_model = Mock(return_value=(0.0, 0.0))
    element.hittester.get_hit_data = Mock(return_value=(None, None))
    element.remove_point_if_unused = Mock()
    element.mark_dirty = Mock()
    element.update_bounds_from_sketch = Mock()
    element.editor = None
    return element


@pytest.fixture
def line_tool(mock_element):
    """Create a LineTool instance for testing."""
    return LineTool(mock_element)


def test_line_tool_initialization(line_tool, mock_element):
    """Test that LineTool initializes correctly."""
    assert line_tool.element == mock_element
    assert line_tool.line_start_id is None
    assert line_tool.start_point_temp is False


def test_line_tool_on_deactivate_no_start(line_tool, mock_element):
    """Test on_deactivate when no start point exists."""
    line_tool.on_deactivate()

    assert line_tool.line_start_id is None
    assert line_tool.start_point_temp is False


def test_line_tool_on_deactivate_with_temp_start(line_tool, mock_element):
    """Test on_deactivate with temporary start point."""
    line_tool.line_start_id = 1
    line_tool.start_point_temp = True

    line_tool.on_deactivate()

    assert line_tool.line_start_id is None
    assert line_tool.start_point_temp is False
    mock_element.remove_point_if_unused.assert_called_once_with(1)


def test_line_tool_on_press_no_hit(line_tool, mock_element):
    """Test on_press when no point is hit."""
    mock_element.hittester.get_hit_data.return_value = (None, None)
    mock_element.hittester.screen_to_model.return_value = (10.0, 20.0)
    mock_element.sketch.add_point.return_value = 0

    result = line_tool.on_press(100.0, 200.0, 1)

    assert result is True
    assert line_tool.line_start_id == 0
    assert line_tool.start_point_temp is True


def test_line_tool_on_press_existing_point(line_tool, mock_element):
    """Test on_press when an existing point is hit."""
    mock_element.hittester.get_hit_data.return_value = ("point", 5)
    mock_element.hittester.screen_to_model.return_value = (10.0, 20.0)

    result = line_tool.on_press(100.0, 200.0, 1)

    assert result is True
    assert line_tool.line_start_id == 5
    assert line_tool.start_point_temp is False


def test_line_tool_on_drag(line_tool):
    """Test on_drag does nothing."""
    line_tool.on_drag(10.0, 20.0)
    assert True


def test_line_tool_on_release(line_tool):
    """Test on_release does nothing."""
    line_tool.on_release(10.0, 20.0)
    assert True


def test_line_tool_handle_click_first_point(line_tool, mock_element):
    """Test _handle_click for first point."""
    mock_element.hittester.screen_to_model.return_value = (10.0, 20.0)
    mock_element.sketch.add_point.return_value = 0

    result = line_tool._handle_click(None, 10.0, 20.0)

    assert result is True
    assert line_tool.line_start_id == 0
    assert line_tool.start_point_temp is True
    mock_element.selection.clear.assert_called_once()
    mock_element.selection.select_point.assert_called_once_with(0, False)


def test_line_tool_handle_click_second_point(line_tool, mock_element):
    """Test _handle_click for second point (completes line)."""
    line_tool.line_start_id = 1
    line_tool.start_point_temp = False
    mock_element.hittester.screen_to_model.return_value = (20.0, 30.0)
    mock_element.sketch.registry._id_counter = 2

    result = line_tool._handle_click(None, 20.0, 30.0)

    assert result is True
    assert line_tool.line_start_id is not None
    mock_element.mark_dirty.assert_called_once()
