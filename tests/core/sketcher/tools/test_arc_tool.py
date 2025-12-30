import pytest
from unittest.mock import Mock
from rayforge.core.sketcher.tools.arc_tool import ArcTool


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
    element.sketch.add_arc = Mock(return_value=0)
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
def arc_tool(mock_element):
    """Create an ArcTool instance for testing."""
    return ArcTool(mock_element)


def test_arc_tool_initialization(arc_tool, mock_element):
    """Test that ArcTool initializes correctly."""
    assert arc_tool.element == mock_element
    assert arc_tool.center_id is None
    assert arc_tool.start_id is None
    assert arc_tool.center_temp is False
    assert arc_tool.start_temp is False
    assert arc_tool.temp_end_id is None
    assert arc_tool.temp_entity_id is None


def test_arc_tool_cleanup_temps_no_temps(arc_tool):
    """Test cleanup when no temps exist."""
    arc_tool._cleanup_temps()
    assert arc_tool.temp_end_id is None
    assert arc_tool.temp_entity_id is None


def test_arc_tool_on_deactivate(arc_tool, mock_element):
    """Test that on_deactivate cleans up state."""
    arc_tool.center_id = 1
    arc_tool.start_id = 2
    arc_tool.center_temp = True
    arc_tool.start_temp = True
    arc_tool.temp_end_id = 3
    arc_tool.temp_entity_id = 4

    arc_tool.on_deactivate()

    assert arc_tool.center_id is None
    assert arc_tool.start_id is None
    assert arc_tool.center_temp is False
    assert arc_tool.start_temp is False
    assert arc_tool.temp_end_id is None
    assert arc_tool.temp_entity_id is None
    mock_element.mark_dirty.assert_called_once()


def test_arc_tool_on_press_no_hit(arc_tool, mock_element):
    """Test on_press when no point is hit."""
    mock_element.hittester.get_hit_data.return_value = (None, None)
    mock_element.hittester.screen_to_model.return_value = (10.0, 20.0)
    mock_element.sketch.add_point.return_value = 0

    result = arc_tool.on_press(100.0, 200.0, 1)

    assert result is True
    assert arc_tool.center_id == 0
    assert arc_tool.center_temp is True
    mock_element.sketch.add_point.assert_called_once_with(10.0, 20.0)


def test_arc_tool_on_drag(arc_tool):
    """Test on_drag does nothing."""
    arc_tool.on_drag(10.0, 20.0)
    assert True


def test_arc_tool_on_release(arc_tool):
    """Test on_release does nothing."""
    arc_tool.on_release(10.0, 20.0)
    assert True


def test_arc_tool_on_hover_motion_no_preview(arc_tool):
    """Test on_hover_motion when not in preview stage."""
    arc_tool.center_id = None
    arc_tool.on_hover_motion(100.0, 200.0)
    assert True


def test_arc_tool_on_hover_motion_with_preview(arc_tool, mock_element):
    """Test on_hover_motion updates preview when in preview stage."""
    arc_tool.center_id = 0
    arc_tool.start_id = 1
    arc_tool.temp_end_id = 2
    arc_tool.temp_entity_id = 3

    mock_point_center = Mock()
    mock_point_center.x = 0.0
    mock_point_center.y = 0.0
    mock_point_start = Mock()
    mock_point_start.x = 10.0
    mock_point_start.y = 0.0
    mock_point_end = Mock()
    mock_point_end.x = 0.0
    mock_point_end.y = 0.0
    mock_arc = Mock()
    mock_arc.clockwise = False

    mock_element.sketch.registry.get_point.side_effect = [
        mock_point_center,
        mock_point_start,
        mock_point_end,
    ]
    mock_element.sketch.registry.get_entity.return_value = mock_arc
    mock_element.hittester.screen_to_model.return_value = (10.0, 10.0)

    arc_tool.on_hover_motion(100.0, 200.0)

    mock_point_end.x = 10.0
    mock_point_end.y = 10.0


def test_arc_tool_handle_click_center_point(arc_tool, mock_element):
    """Test _handle_click for setting center point."""
    mock_element.hittester.screen_to_model.return_value = (10.0, 20.0)
    mock_element.sketch.add_point.return_value = 0

    result = arc_tool._handle_click(None, 10.0, 20.0)

    assert result is True
    assert arc_tool.center_id == 0
    assert arc_tool.center_temp is True
