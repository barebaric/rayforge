import pytest
from unittest.mock import Mock, MagicMock, patch
from rayforge.ui_gtk.sketcher.tools.bezier_tool import BezierTool
from rayforge.core.sketcher.commands.bezier import BezierPreviewState


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
    element.sketch.registry.add_bezier = Mock(return_value=0)
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
def bezier_tool(mock_element):
    """Create a BezierTool instance for testing."""
    return BezierTool(mock_element)


def test_bezier_tool_initialization(bezier_tool, mock_element):
    """Test that BezierTool initializes correctly."""
    assert bezier_tool.element == mock_element
    assert bezier_tool._preview_state is None


def test_bezier_tool_on_deactivate_no_preview(bezier_tool, mock_element):
    """Test that on_deactivate works when no preview state."""
    bezier_tool.on_deactivate()
    assert bezier_tool._preview_state is None


def test_bezier_tool_on_deactivate_with_preview(bezier_tool, mock_element):
    """Test that on_deactivate cleans up preview state."""
    bezier_tool._preview_state = BezierPreviewState(
        start_id=1,
        start_temp=True,
        cp1_id=2,
        cp1_temp=True,
        cp2_id=3,
        cp2_temp=True,
        end_id=4,
        end_temp=True,
        temp_entity_id=5,
    )

    with patch(
        "rayforge.ui_gtk.sketcher.tools.bezier_tool"
        ".BezierCommand.cleanup_preview"
    ):
        bezier_tool.on_deactivate()

    assert bezier_tool._preview_state is None
    mock_element.remove_point_if_unused.assert_called()


def test_bezier_tool_on_press_no_hit(bezier_tool, mock_element):
    """Test on_press when no point is hit."""
    mock_element.hittester.get_hit_data.return_value = (None, None)
    mock_element.hittester.screen_to_model.return_value = (10.0, 20.0)
    mock_element.sketch.registry.add_point.side_effect = [0]

    with patch(
        "rayforge.ui_gtk.sketcher.tools.bezier_tool"
        ".BezierCommand.start_preview"
    ) as mock_start:
        mock_start.return_value = BezierPreviewState(
            start_id=0,
            start_temp=True,
        )
        result = bezier_tool.on_press(100.0, 200.0, 1)

    assert result is True
    assert bezier_tool._preview_state is not None
    assert bezier_tool._preview_state.start_id == 0


def test_bezier_tool_on_drag(bezier_tool):
    """Test on_drag does nothing."""
    bezier_tool.on_drag(10.0, 20.0)
    assert True


def test_bezier_tool_on_release(bezier_tool):
    """Test on_release does nothing."""
    bezier_tool.on_release(10.0, 20.0)
    assert True


def test_bezier_tool_on_hover_motion_no_preview(bezier_tool):
    """Test on_hover_motion when not in preview stage."""
    bezier_tool._preview_state = None
    bezier_tool.on_hover_motion(100.0, 200.0)
    assert True


def test_bezier_tool_on_hover_motion_without_cp1(bezier_tool, mock_element):
    """Test on_hover_motion when cp1 not yet set."""
    bezier_tool._preview_state = BezierPreviewState(
        start_id=0,
        start_temp=True,
    )
    bezier_tool.on_hover_motion(100.0, 200.0)
    mock_element.mark_dirty.assert_not_called()


def test_bezier_tool_on_hover_motion_with_preview(bezier_tool, mock_element):
    """Test on_hover_motion updates preview when in preview stage."""
    bezier_tool._preview_state = BezierPreviewState(
        start_id=0,
        start_temp=True,
        cp1_id=1,
        cp1_temp=True,
        cp2_id=2,
        cp2_temp=True,
        end_id=3,
        end_temp=True,
        temp_entity_id=4,
    )

    mock_point = Mock()
    mock_point.x = 0.0
    mock_point.y = 0.0
    mock_element.sketch.registry.get_point.return_value = mock_point
    mock_element.hittester.screen_to_model.return_value = (10.0, 10.0)

    with patch(
        "rayforge.ui_gtk.sketcher.tools.bezier_tool"
        ".BezierCommand.update_preview"
    ):
        bezier_tool.on_hover_motion(100.0, 200.0)

    mock_element.mark_dirty.assert_called()


def test_bezier_tool_shortcut(bezier_tool):
    """Test that BezierTool has the correct shortcut."""
    assert BezierTool.SHORTCUT == ("gb", "Bezier")
