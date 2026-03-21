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
    element.sketch.registry.add_line = Mock(return_value=0)
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
    assert bezier_tool._press_pos is None
    assert bezier_tool._dragging is False


def test_bezier_tool_on_deactivate_no_preview(bezier_tool, mock_element):
    """Test that on_deactivate works when no preview state."""
    bezier_tool.on_deactivate()
    assert bezier_tool._preview_state is None
    assert bezier_tool._press_pos is None
    assert bezier_tool._dragging is False


def test_bezier_tool_on_deactivate_with_preview(bezier_tool, mock_element):
    """Test that on_deactivate cleans up preview state."""
    bezier_tool._preview_state = BezierPreviewState(
        start_id=1,
        start_temp=True,
        end_id=2,
        end_temp=True,
        temp_entity_id=3,
        is_line_preview=True,
    )

    with patch(
        "rayforge.ui_gtk.sketcher.tools.bezier_tool"
        ".BezierCommand.cleanup_preview"
    ):
        bezier_tool.on_deactivate()

    assert bezier_tool._preview_state is None
    mock_element.remove_point_if_unused.assert_called_once_with(1)


def test_bezier_tool_first_press_starts_preview(bezier_tool, mock_element):
    """Test first press starts line preview."""
    mock_element.hittester.get_hit_data.return_value = (None, None)
    mock_element.hittester.screen_to_model.return_value = (10.0, 20.0)

    with patch(
        "rayforge.ui_gtk.sketcher.tools.bezier_tool"
        ".BezierCommand.start_preview"
    ) as mock_start:
        mock_start.return_value = BezierPreviewState(
            start_id=0,
            start_temp=True,
            end_id=1,
            end_temp=True,
            temp_entity_id=2,
            is_line_preview=True,
        )
        result = bezier_tool.on_press(100.0, 200.0, 1)

    assert result is False
    assert bezier_tool._preview_state is not None
    assert bezier_tool._waypoint_model_pos == (10.0, 20.0)
    assert bezier_tool._press_pos == (100.0, 200.0)
    assert bezier_tool._dragging is False


def test_bezier_tool_on_drag_below_threshold(bezier_tool, mock_element):
    """Test on_drag does nothing below threshold."""
    bezier_tool._press_pos = (100.0, 200.0)
    bezier_tool._waypoint_model_pos = (10.0, 20.0)
    bezier_tool._preview_state = BezierPreviewState(
        start_id=0,
        start_temp=True,
        end_id=1,
        end_temp=True,
        temp_entity_id=2,
        is_line_preview=True,
    )

    bezier_tool.on_drag(2.0, 2.0)

    assert bezier_tool._dragging is False


def test_bezier_tool_on_drag_starts_bezier(bezier_tool, mock_element):
    """Test on_drag converts line to bezier above threshold."""
    bezier_tool._press_pos = (100.0, 200.0)
    bezier_tool._waypoint_model_pos = (10.0, 20.0)
    bezier_tool._preview_state = BezierPreviewState(
        start_id=0,
        start_temp=True,
        end_id=1,
        end_temp=True,
        temp_entity_id=2,
        is_line_preview=True,
    )

    def get_point_side_effect(pid):
        if pid == 0:
            return MagicMock(x=0.0, y=0.0)
        elif pid == 1:
            return MagicMock(x=10.0, y=20.0)
        return MagicMock(x=0.0, y=0.0)

    mock_element.sketch.registry.get_point.side_effect = get_point_side_effect
    mock_element.hittester.screen_to_model.return_value = (15.0, 25.0)

    with patch(
        "rayforge.ui_gtk.sketcher.tools.bezier_tool"
        ".BezierCommand.convert_to_bezier"
    ) as mock_convert:
        bezier_tool.on_drag(10.0, 10.0)

        mock_convert.assert_called_once_with(
            mock_element.sketch.registry,
            bezier_tool._preview_state,
            10.0,
            20.0,
            15.0,
            25.0,
            mirror_cp_offset=None,
        )

    assert bezier_tool._dragging is True


def test_bezier_tool_on_release_without_drag_creates_line(
    bezier_tool, mock_element
):
    """Test on_release creates line when not dragging and end has moved."""
    bezier_tool._press_pos = (100.0, 200.0)
    bezier_tool._waypoint_model_pos = (20.0, 30.0)
    bezier_tool._dragging = False
    bezier_tool._snapped_pid = None
    bezier_tool._preview_state = BezierPreviewState(
        start_id=0,
        start_temp=True,
        end_id=1,
        end_temp=True,
        temp_entity_id=2,
        is_line_preview=True,
    )

    def get_point_side_effect(pid):
        if pid == 0:
            return MagicMock(x=10.0, y=15.0)
        else:
            return MagicMock(x=20.0, y=30.0)

    mock_element.sketch.registry.get_point.side_effect = get_point_side_effect

    with patch(
        "rayforge.ui_gtk.sketcher.tools.bezier_tool"
        ".BezierCommand.cleanup_preview"
    ):
        bezier_tool.on_release(100.0, 200.0)

        mock_element.execute_command.assert_called_once()
        cmd = mock_element.execute_command.call_args[0][0]
        assert cmd.is_line is True


def test_bezier_tool_on_release_with_drag_creates_bezier(
    bezier_tool, mock_element
):
    """Test on_release creates bezier when dragging."""
    bezier_tool._press_pos = (100.0, 200.0)
    bezier_tool._waypoint_model_pos = (10.0, 20.0)
    bezier_tool._dragging = True
    bezier_tool._snapped_pid = None
    bezier_tool._preview_state = BezierPreviewState(
        start_id=0,
        start_temp=True,
        end_id=1,
        end_temp=True,
        temp_entity_id=4,
        is_line_preview=False,
        virtual_cp=(2.0, 2.0),
    )

    mock_start_pt = MagicMock(x=10.0, y=20.0)
    mock_end_pt = MagicMock(x=15.0, y=25.0)
    mock_entity = MagicMock(cp1=(1.0, 1.0), cp2=(-2.0, -2.0))

    def get_point_side_effect(pid):
        if pid == 0:
            return mock_start_pt
        elif pid == 1:
            return mock_end_pt
        return None

    def get_entity_side_effect(eid):
        if eid == 4:
            return mock_entity
        return None

    mock_element.sketch.registry.get_point.side_effect = get_point_side_effect
    mock_element.sketch.registry.get_entity.side_effect = (
        get_entity_side_effect
    )

    with patch(
        "rayforge.ui_gtk.sketcher.tools.bezier_tool"
        ".BezierCommand.cleanup_preview"
    ):
        bezier_tool.on_release(100.0, 200.0)

        mock_element.execute_command.assert_called_once()
        cmd = mock_element.execute_command.call_args[0][0]
        assert cmd.is_line is False


def test_bezier_tool_on_hover_motion_updates_preview(
    bezier_tool, mock_element
):
    """Test on_hover_motion updates preview when in preview stage."""
    bezier_tool._preview_state = BezierPreviewState(
        start_id=0,
        start_temp=True,
        end_id=1,
        end_temp=True,
        temp_entity_id=2,
        is_line_preview=True,
    )
    bezier_tool._in_press = False
    mock_element.hittester.screen_to_model.return_value = (15.0, 25.0)

    with patch(
        "rayforge.ui_gtk.sketcher.tools.bezier_tool"
        ".BezierCommand.update_preview"
    ):
        bezier_tool.on_hover_motion(100.0, 200.0)

    mock_element.mark_dirty.assert_called()


def test_bezier_tool_on_hover_motion_skips_when_in_press(
    bezier_tool, mock_element
):
    """Test on_hover_motion skips when in press sequence."""
    bezier_tool._preview_state = BezierPreviewState(
        start_id=0,
        start_temp=True,
        end_id=1,
        end_temp=True,
        temp_entity_id=2,
        is_line_preview=False,
    )
    bezier_tool._in_press = True

    bezier_tool.on_hover_motion(100.0, 200.0)

    mock_element.mark_dirty.assert_not_called()


def test_bezier_tool_shortcut(bezier_tool):
    """Test that BezierTool has the correct shortcut."""
    assert BezierTool.SHORTCUTS == ["gb"]
