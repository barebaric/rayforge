import pytest
from unittest.mock import Mock
from rayforge.core.sketcher.tools.circle_tool import CircleTool


@pytest.fixture
def mock_element():
    """Create a mock SketchElement for testing."""
    element = Mock()
    element.sketch = Mock()
    element.sketch.registry = Mock()
    element.sketch.registry.points = []
    element.sketch.registry.entities = []
    element.sketch.registry._id_counter = 0
    element.sketch.registry._entity_map = {}
    element.sketch.add_point = Mock(return_value=0)
    element.sketch.add_circle = Mock(return_value=0)
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
def circle_tool(mock_element):
    """Create a CircleTool instance for testing."""
    return CircleTool(mock_element)


def test_circle_tool_initialization(circle_tool, mock_element):
    """Test that CircleTool initializes correctly."""
    assert circle_tool.element == mock_element
    assert circle_tool.center_id is None
    assert circle_tool.center_temp is False
    assert circle_tool.temp_radius_id is None
    assert circle_tool.temp_entity_id is None


def test_circle_tool_cleanup_temps_no_temps(circle_tool):
    """Test cleanup when no temps exist."""
    circle_tool._cleanup_temps()
    assert circle_tool.temp_radius_id is None
    assert circle_tool.temp_entity_id is None


def test_circle_tool_on_deactivate(circle_tool, mock_element):
    """Test that on_deactivate cleans up state."""
    circle_tool.center_id = 1
    circle_tool.center_temp = True
    circle_tool.temp_radius_id = 2
    circle_tool.temp_entity_id = 3

    circle_tool.on_deactivate()

    assert circle_tool.center_id is None
    assert circle_tool.center_temp is False
    assert circle_tool.temp_radius_id is None
    assert circle_tool.temp_entity_id is None
    mock_element.mark_dirty.assert_called_once()


def test_circle_tool_on_press_no_hit(circle_tool, mock_element):
    """Test on_press when no point is hit."""
    mock_element.hittester.get_hit_data.return_value = (None, None)
    mock_element.hittester.screen_to_model.return_value = (10.0, 20.0)
    mock_element.sketch.add_point.return_value = 0

    result = circle_tool.on_press(100.0, 200.0, 1)

    assert result is True
    assert circle_tool.center_id == 0
    assert circle_tool.center_temp is True
    assert mock_element.sketch.add_point.call_count == 2


def test_circle_tool_on_drag(circle_tool):
    """Test on_drag does nothing."""
    circle_tool.on_drag(10.0, 20.0)
    assert True


def test_circle_tool_on_release(circle_tool):
    """Test on_release does nothing."""
    circle_tool.on_release(10.0, 20.0)
    assert True


def test_circle_tool_on_hover_motion_no_preview(circle_tool):
    """Test on_hover_motion when not in preview stage."""
    circle_tool.center_id = None
    circle_tool.on_hover_motion(100.0, 200.0)
    assert True


def test_circle_tool_on_hover_motion_with_preview(circle_tool, mock_element):
    """Test on_hover_motion updates preview when in preview stage."""
    circle_tool.center_id = 0
    circle_tool.temp_radius_id = 1
    circle_tool.temp_entity_id = 2

    mock_point = Mock()
    mock_point.x = 0.0
    mock_point.y = 0.0

    mock_element.sketch.registry.get_point.return_value = mock_point
    mock_element.hittester.screen_to_model.return_value = (10.0, 10.0)

    circle_tool.on_hover_motion(100.0, 200.0)

    assert mock_point.x == 10.0
    assert mock_point.y == 10.0
    mock_element.mark_dirty.assert_called_once()


def test_circle_tool_handle_click_center_point(circle_tool, mock_element):
    """Test _handle_click for setting center point."""
    mock_element.hittester.screen_to_model.return_value = (10.0, 20.0)
    mock_element.sketch.add_point.return_value = 0

    result = circle_tool._handle_click(None, 10.0, 20.0)

    assert result is True
    assert circle_tool.center_id == 0
    assert circle_tool.center_temp is True
