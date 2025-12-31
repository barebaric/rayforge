import pytest
from unittest.mock import Mock
from rayforge.core.sketcher.tools.select_tool import SelectTool


@pytest.fixture
def mock_element():
    """Create a mock SketchElement for testing."""
    element = Mock()
    element.sketch = Mock()
    element.sketch.registry = Mock()
    element.sketch.registry.points = []
    element.sketch.registry.entities = []
    element.sketch.constraints = []
    element.sketch.get_coincident_points = Mock(return_value=[])
    element.selection = Mock()
    element.selection.clear = Mock()
    element.selection.point_ids = []
    element.selection.entity_ids = []
    element.selection.constraint_idx = None
    element.selection.junction_pid = None
    element.selection.select_point = Mock()
    element.selection.select_entity = Mock()
    element.selection.select_constraint = Mock()
    element.selection.select_junction = Mock()
    element.selection.copy = Mock(return_value=Mock())
    element.hittester = Mock()
    element.hittester.get_hit_data = Mock(return_value=(None, None))
    element.hittester.get_objects_in_rect = Mock(return_value=([], []))
    element.canvas = None
    element.mark_dirty = Mock()
    element.get_world_transform = Mock()
    element.content_transform = Mock()
    element.constraint_edit_requested = Mock()
    return element


@pytest.fixture
def select_tool(mock_element):
    """Create a SelectTool instance for testing."""
    return SelectTool(mock_element)


def test_select_tool_initialization(select_tool, mock_element):
    """Test that SelectTool initializes correctly."""
    assert select_tool.element == mock_element
    assert select_tool.hovered_point_id is None
    assert select_tool.hovered_constraint_idx is None
    assert select_tool.hovered_junction_pid is None
    assert select_tool.is_box_selecting is False
    assert select_tool.drag_start_world_pos is None
    assert select_tool.drag_current_world_pos is None
    assert select_tool.dragged_point_id is None
    assert select_tool.drag_point_start_pos is None
    assert select_tool.dragged_entity is None
    assert select_tool.drag_start_model_pos is None


def test_select_tool_on_press_no_hit(select_tool, mock_element):
    """Test on_press when nothing is hit."""
    mock_element.hittester.get_hit_data.return_value = (None, None)
    mock_element.canvas = Mock()
    mock_element.canvas._shift_pressed = False
    mock_element.canvas._ctrl_pressed = False

    result = select_tool.on_press(100.0, 200.0, 1)

    assert result is False
    assert select_tool.is_box_selecting is True
    assert select_tool.drag_start_world_pos == (100.0, 200.0)


def test_select_tool_on_press_point_hit(select_tool, mock_element):
    """Test on_press when a point is hit."""
    mock_element.hittester.get_hit_data.return_value = ("point", 5)

    result = select_tool.on_press(100.0, 200.0, 1)

    assert result is False
    mock_element.selection.select_point.assert_called_once_with(5, False)


def test_select_tool_on_press_entity_hit(select_tool, mock_element):
    """Test on_press when an entity is hit."""
    from rayforge.core.sketcher.entities import Line

    mock_entity = Mock(spec=Line)
    mock_entity.id = 10
    mock_entity.p1_idx = 1
    mock_entity.p2_idx = 2
    mock_element.hittester.get_hit_data.return_value = ("entity", mock_entity)
    mock_element.hittester.screen_to_model.return_value = (100.0, 200.0)

    result = select_tool.on_press(100.0, 200.0, 1)

    assert result is False
    mock_element.selection.select_entity.assert_called_once()


def test_select_tool_on_drag_no_state(select_tool):
    """Test on_drag when no drag state is set."""
    select_tool.on_drag(10.0, 20.0)
    assert True


def test_select_tool_on_drag_box_select(select_tool, mock_element):
    """Test on_drag during box selection."""
    select_tool.is_box_selecting = True
    select_tool.drag_start_world_pos = (100.0, 200.0)
    mock_element.hittester.get_objects_in_rect.return_value = ([1, 2], [3])
    mock_element.hittester.screen_to_model.return_value = (150.0, 300.0)

    select_tool.on_drag(50.0, 100.0)

    assert select_tool.drag_current_world_pos == (150.0, 300.0)
    mock_element.mark_dirty.assert_called_once()


def test_select_tool_on_release_box_select(select_tool):
    """Test on_release after box selection."""
    select_tool.is_box_selecting = True
    select_tool.drag_start_world_pos = (100.0, 200.0)
    select_tool.drag_current_world_pos = (150.0, 300.0)

    select_tool.on_release(150.0, 300.0)

    assert select_tool.is_box_selecting is False
    assert select_tool.drag_start_world_pos is None
    assert select_tool.drag_current_world_pos is None


def test_select_tool_on_hover_motion_no_change(select_tool, mock_element):
    """Test on_hover_motion when hit type doesn't change."""
    mock_element.hittester.get_hit_data.return_value = ("point", 5)
    select_tool.hovered_point_id = 5

    select_tool.on_hover_motion(100.0, 200.0)

    mock_element.mark_dirty.assert_not_called()


def test_select_tool_on_hover_motion_change(select_tool, mock_element):
    """Test on_hover_motion when hit type changes."""
    mock_element.hittester.get_hit_data.return_value = ("point", 5)
    select_tool.hovered_point_id = None

    select_tool.on_hover_motion(100.0, 200.0)

    assert select_tool.hovered_point_id == 5
    mock_element.mark_dirty.assert_called_once()


def test_select_tool_prepare_point_drag(select_tool, mock_element):
    """Test _prepare_point_drag sets up drag state."""
    mock_point = Mock()
    mock_point.x = 10.0
    mock_point.y = 20.0
    mock_element.sketch.registry.get_point.return_value = mock_point
    mock_element.get_world_transform.return_value.invert.return_value = Mock()
    mock_element.content_transform.invert.return_value = Mock()

    select_tool._prepare_point_drag(5)

    assert select_tool.dragged_point_id == 5
    assert select_tool.drag_point_start_pos == (10.0, 20.0)
    assert select_tool.dragged_entity is None


def test_select_tool_draw_overlay_no_box(select_tool):
    """Test draw_overlay when not box selecting."""
    import cairo

    ctx = Mock(spec=cairo.Context)
    select_tool.is_box_selecting = False

    select_tool.draw_overlay(ctx)

    ctx.save.assert_not_called()


def test_select_tool_draw_overlay_with_box(select_tool, mock_element):
    """Test draw_overlay when box selecting."""
    import cairo

    ctx = Mock(spec=cairo.Context)
    select_tool.is_box_selecting = True
    select_tool.drag_start_world_pos = (100.0, 200.0)
    select_tool.drag_current_world_pos = (150.0, 300.0)
    mock_element.canvas = Mock()
    mock_element.canvas.view_transform = Mock()
    mock_element.canvas.view_transform.transform_point.return_value = (
        100.0,
        200.0,
    )

    select_tool.draw_overlay(ctx)

    ctx.save.assert_called_once()
    ctx.rectangle.assert_called_once()
