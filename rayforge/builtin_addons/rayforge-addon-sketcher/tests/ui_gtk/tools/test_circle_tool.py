import pytest
from unittest.mock import Mock, MagicMock, patch
from sketcher.ui_gtk.tools.circle_tool import CircleTool
from sketcher.core.commands.circle import CirclePreviewState


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
    element.sketch.registry.add_point = Mock(return_value=0)
    element.sketch.registry.add_circle = Mock(return_value=0)
    element.sketch.registry.get_point = Mock(
        side_effect=lambda pid: MagicMock(x=0.0, y=0.0)
    )
    element.sketch.remove_point_if_unused = Mock()
    element.selection = Mock()
    element.selection.clear = Mock()
    element.selection.select_point = Mock()
    element.hittester = Mock()
    element.hittester.screen_to_model = Mock(return_value=(0.0, 0.0))
    element.hittester.get_hit_data = Mock(return_value=(None, None))
    element.remove_point_if_unused = Mock()
    element.mark_dirty = Mock()
    element.update_bounds_from_sketch = Mock()
    element.execute_command = Mock()
    element.editor = None
    return element


@pytest.fixture
def circle_tool(mock_element):
    """Create a CircleTool instance for testing."""
    return CircleTool(mock_element)


def test_circle_tool_initialization(circle_tool, mock_element):
    """Test that CircleTool initializes correctly."""
    assert circle_tool.element == mock_element
    assert circle_tool._preview_state is None


def test_circle_tool_on_deactivate_no_preview(circle_tool, mock_element):
    """Test that on_deactivate works when no preview state."""
    circle_tool.on_deactivate()
    assert circle_tool._preview_state is None


def test_circle_tool_on_deactivate_with_preview(circle_tool, mock_element):
    """Test that on_deactivate cleans up preview state."""
    circle_tool._preview_state = CirclePreviewState(
        center_id=1,
        center_temp=True,
        radius_id=2,
        entity_id=3,
    )

    with patch(
        "sketcher.ui_gtk.tools.circle_tool.CircleCommand.cleanup_preview"
    ):
        circle_tool.on_deactivate()

    assert circle_tool._preview_state is None
    mock_element.remove_point_if_unused.assert_called_once_with(1)


def test_circle_tool_on_press_no_hit(circle_tool, mock_element):
    """Test on_press when no point is hit."""
    mock_element.hittester.get_hit_data.return_value = (None, None)
    mock_element.hittester.screen_to_model.return_value = (10.0, 20.0)
    mock_element.sketch.registry.add_point.side_effect = [0, 1]
    mock_element.sketch.registry.add_circle.return_value = 2

    with patch(
        "sketcher.ui_gtk.tools.circle_tool.CircleCommand.start_preview"
    ) as mock_start:
        mock_start.return_value = CirclePreviewState(
            center_id=0,
            center_temp=True,
            radius_id=1,
            entity_id=2,
        )
        result = circle_tool.on_press(100.0, 200.0, 1)

    assert result is True
    assert circle_tool._preview_state is not None
    assert circle_tool._preview_state.center_id == 0


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
    circle_tool._preview_state = None
    circle_tool.on_hover_motion(100.0, 200.0)
    assert True


def test_circle_tool_on_hover_motion_with_preview(circle_tool, mock_element):
    """Test on_hover_motion updates preview when in preview stage."""
    circle_tool._preview_state = CirclePreviewState(
        center_id=0,
        center_temp=True,
        radius_id=1,
        entity_id=2,
    )

    mock_point = Mock()
    mock_point.x = 0.0
    mock_point.y = 0.0
    mock_element.sketch.registry.get_point.return_value = mock_point
    mock_element.hittester.screen_to_model.return_value = (10.0, 10.0)

    circle_tool.on_hover_motion(100.0, 200.0)

    mock_element.mark_dirty.assert_called()


def test_circle_tool_handle_click_center_point(circle_tool, mock_element):
    """Test _handle_click for setting center point."""
    mock_element.hittester.screen_to_model.return_value = (10.0, 20.0)
    mock_element.sketch.registry.add_point.side_effect = [0, 1]
    mock_element.sketch.registry.add_circle.return_value = 2

    with patch(
        "sketcher.ui_gtk.tools.circle_tool.CircleCommand.start_preview"
    ) as mock_start:
        mock_start.return_value = CirclePreviewState(
            center_id=0,
            center_temp=True,
            radius_id=1,
            entity_id=2,
        )
        result = circle_tool._handle_click(None, 10.0, 20.0)

    assert result is True
    assert circle_tool._preview_state is not None
    assert circle_tool._preview_state.center_id == 0
    assert circle_tool._preview_state.center_temp is True
