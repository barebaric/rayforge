import pytest
from unittest.mock import Mock, MagicMock, patch
from sketcher.ui_gtk.tools.circle_tool import CircleTool
from sketcher.core.commands.ellipse import EllipsePreviewState
from sketcher.ui_gtk.tools.base import SketcherKey
from sketcher.core.snap import SnapLineType
from sketcher.core.entities import Point as SketchPoint


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
    element.sketch.registry.add_ellipse = Mock(return_value=0)
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
    element.canvas = Mock()
    element.canvas._shift_pressed = False
    element.canvas.view_transform = Mock()
    element.canvas.view_transform.get_scale = Mock(return_value=(1.0, 1.0))
    element.snap_engine = Mock()
    element.snap_engine.query = Mock(
        return_value=MagicMock(
            snapped=False,
            position=(0.0, 0.0),
            snap_lines=[],
            primary_snap_point=None,
        )
    )
    return element


@pytest.fixture
def circle_tool(mock_element):
    """Create a CircleTool instance for testing."""
    return CircleTool(mock_element)


@pytest.mark.ui
def test_circle_tool_initialization(circle_tool, mock_element):
    """Test that CircleTool initializes correctly."""
    assert circle_tool.element == mock_element
    assert circle_tool._preview_state is None
    assert circle_tool._ctrl_held is False
    assert circle_tool._shift_held is False


@pytest.mark.ui
def test_circle_tool_is_available(circle_tool):
    """Test is_available returns True only when target is None."""
    assert circle_tool.is_available(None, None) is True
    assert circle_tool.is_available("something", "point") is False


@pytest.mark.ui
def test_circle_tool_shortcut_is_active(circle_tool):
    """Test shortcut_is_active always returns True."""
    assert circle_tool.shortcut_is_active() is True


@pytest.mark.ui
def test_circle_tool_get_preview_state(circle_tool):
    """Test get_preview_state returns current preview state."""
    assert circle_tool.get_preview_state() is None

    circle_tool._preview_state = EllipsePreviewState(
        start_id=1,
        start_temp=True,
        center_id=2,
        radius_x_id=3,
        radius_y_id=4,
        entity_id=5,
    )
    assert circle_tool.get_preview_state() is not None


@pytest.mark.ui
def test_circle_tool_on_deactivate_no_preview(circle_tool, mock_element):
    """Test that on_deactivate works when no preview state."""
    circle_tool.on_deactivate()
    assert circle_tool._preview_state is None
    mock_element.mark_dirty.assert_not_called()


@pytest.mark.ui
def test_circle_tool_on_deactivate_with_preview(circle_tool, mock_element):
    """Test that on_deactivate cleans up preview state."""
    circle_tool._preview_state = EllipsePreviewState(
        start_id=1,
        start_temp=True,
        center_id=2,
        radius_x_id=3,
        radius_y_id=4,
        entity_id=5,
    )

    with patch(
        "sketcher.ui_gtk.tools.circle_tool.EllipseCommand.cleanup_preview"
    ):
        circle_tool.on_deactivate()

    assert circle_tool._preview_state is None
    mock_element.remove_point_if_unused.assert_called_once_with(1)
    mock_element.mark_dirty.assert_called_once()


@pytest.mark.ui
def test_circle_tool_on_deactivate_with_snapped_start(
    circle_tool, mock_element
):
    """Test on_deactivate when start was snapped (not temp)."""
    circle_tool._preview_state = EllipsePreviewState(
        start_id=1,
        start_temp=False,
        center_id=2,
        radius_x_id=3,
        radius_y_id=4,
        entity_id=5,
    )

    with patch(
        "sketcher.ui_gtk.tools.circle_tool.EllipseCommand.cleanup_preview"
    ):
        circle_tool.on_deactivate()

    mock_element.remove_point_if_unused.assert_not_called()


@pytest.mark.ui
def test_circle_tool_on_press_no_hit(circle_tool, mock_element):
    """Test on_press when no point is hit."""
    mock_element.hittester.get_hit_data.return_value = (None, None)
    mock_element.hittester.screen_to_model.return_value = (10.0, 20.0)

    with patch(
        "sketcher.ui_gtk.tools.circle_tool.EllipseCommand.start_preview"
    ) as mock_start:
        mock_start.return_value = EllipsePreviewState(
            start_id=0,
            start_temp=True,
            center_id=1,
            radius_x_id=2,
            radius_y_id=3,
            entity_id=4,
        )
        result = circle_tool.on_press(100.0, 200.0, 1)

    assert result is True
    assert circle_tool._preview_state is not None
    assert circle_tool._preview_state.start_id == 0


@pytest.mark.ui
def test_circle_tool_on_press_with_snapped_point(circle_tool, mock_element):
    """Test on_press when snapping to an existing point."""
    mock_element.hittester.screen_to_model.return_value = (10.0, 20.0)
    mock_point = SketchPoint(99, 10.0, 20.0)
    mock_snap_point = MagicMock()
    mock_snap_point.line_type = SnapLineType.ENTITY_POINT
    mock_snap_point.source = mock_point
    mock_snap_point.x = 10.0
    mock_snap_point.y = 20.0
    mock_element.snap_engine.query.return_value = MagicMock(
        snapped=True,
        position=(10.0, 20.0),
        snap_lines=[],
        primary_snap_point=mock_snap_point,
    )

    with patch(
        "sketcher.ui_gtk.tools.circle_tool.EllipseCommand.start_preview"
    ) as mock_start:
        mock_start.return_value = EllipsePreviewState(
            start_id=99,
            start_temp=False,
            center_id=1,
            radius_x_id=2,
            radius_y_id=3,
            entity_id=4,
        )
        result = circle_tool.on_press(100.0, 200.0, 1)

    assert result is True
    mock_start.assert_called_once_with(
        mock_element.sketch.registry, 10.0, 20.0, snapped_pid=99
    )


@pytest.mark.ui
def test_circle_tool_on_press_already_in_preview(circle_tool, mock_element):
    """Test on_press when already in preview mode does nothing."""
    circle_tool._preview_state = EllipsePreviewState(
        start_id=1,
        start_temp=True,
        center_id=2,
        radius_x_id=3,
        radius_y_id=4,
        entity_id=5,
    )
    mock_element.hittester.screen_to_model.return_value = (10.0, 20.0)

    result = circle_tool.on_press(100.0, 200.0, 1)

    assert result is True
    mock_element.mark_dirty.assert_not_called()


@pytest.mark.ui
def test_circle_tool_on_drag(circle_tool):
    """Test on_drag does nothing."""
    circle_tool.on_drag(10.0, 20.0)
    assert True


@pytest.mark.ui
def test_circle_tool_on_release_no_preview(circle_tool, mock_element):
    """Test on_release when no preview state does nothing."""
    circle_tool._preview_state = None
    circle_tool.on_release(10.0, 20.0)
    mock_element.execute_command.assert_not_called()


@pytest.mark.ui
def test_circle_tool_on_release_with_preview(circle_tool, mock_element):
    """Test on_release creates command and cleans up."""
    circle_tool._preview_state = EllipsePreviewState(
        start_id=1,
        start_temp=True,
        center_id=2,
        radius_x_id=3,
        radius_y_id=4,
        entity_id=5,
    )
    mock_element.hittester.screen_to_model.return_value = (50.0, 50.0)
    mock_element.hittester.get_hit_data.return_value = (None, None)

    with patch(
        "sketcher.ui_gtk.tools.circle_tool.EllipseCommand.cleanup_preview"
    ):
        with patch("sketcher.ui_gtk.tools.circle_tool.EllipseCommand"):
            circle_tool.on_release(100.0, 200.0)

    assert circle_tool._preview_state is None
    mock_element.execute_command.assert_called_once()
    mock_element.mark_dirty.assert_called()


@pytest.mark.ui
def test_circle_tool_on_release_with_snapped_endpoint(
    circle_tool, mock_element
):
    """Test on_release snaps to existing point."""
    circle_tool._preview_state = EllipsePreviewState(
        start_id=1,
        start_temp=True,
        center_id=2,
        radius_x_id=3,
        radius_y_id=4,
        entity_id=5,
    )
    mock_element.hittester.screen_to_model.return_value = (50.0, 50.0)
    mock_point = SketchPoint(99, 50.0, 50.0)
    mock_snap_point = MagicMock()
    mock_snap_point.line_type = SnapLineType.ENTITY_POINT
    mock_snap_point.source = mock_point
    mock_snap_point.x = 50.0
    mock_snap_point.y = 50.0
    mock_element.snap_engine.query.return_value = MagicMock(
        snapped=True,
        position=(50.0, 50.0),
        snap_lines=[],
        primary_snap_point=mock_snap_point,
    )

    with patch(
        "sketcher.ui_gtk.tools.circle_tool.EllipseCommand.cleanup_preview"
    ):
        with patch(
            "sketcher.ui_gtk.tools.circle_tool.EllipseCommand"
        ) as MockCmd:
            mock_cmd_instance = Mock()
            MockCmd.return_value = mock_cmd_instance
            circle_tool.on_release(100.0, 200.0)

            call_kwargs = MockCmd.call_args[1]
            assert call_kwargs["end_pid"] == 99


@pytest.mark.ui
def test_circle_tool_on_release_ignores_preview_points(
    circle_tool, mock_element
):
    """Test on_release ignores snapped points that are preview points."""
    circle_tool._preview_state = EllipsePreviewState(
        start_id=1,
        start_temp=True,
        center_id=2,
        radius_x_id=3,
        radius_y_id=4,
        entity_id=5,
    )
    mock_element.hittester.screen_to_model.return_value = (50.0, 50.0)
    mock_point = SketchPoint(3, 50.0, 50.0)
    mock_snap_point = MagicMock()
    mock_snap_point.line_type = SnapLineType.ENTITY_POINT
    mock_snap_point.source = mock_point
    mock_snap_point.x = 50.0
    mock_snap_point.y = 50.0
    mock_element.snap_engine.query.return_value = MagicMock(
        snapped=True,
        position=(50.0, 50.0),
        snap_lines=[],
        primary_snap_point=mock_snap_point,
    )

    with patch(
        "sketcher.ui_gtk.tools.circle_tool.EllipseCommand.cleanup_preview"
    ):
        with patch(
            "sketcher.ui_gtk.tools.circle_tool.EllipseCommand"
        ) as MockCmd:
            mock_cmd_instance = Mock()
            MockCmd.return_value = mock_cmd_instance
            circle_tool.on_release(100.0, 200.0)

            call_kwargs = MockCmd.call_args[1]
            assert call_kwargs["end_pid"] is None


@pytest.mark.ui
def test_circle_tool_on_release_with_modifiers(circle_tool, mock_element):
    """Test on_release passes modifier state to command."""
    circle_tool._preview_state = EllipsePreviewState(
        start_id=1,
        start_temp=True,
        center_id=2,
        radius_x_id=3,
        radius_y_id=4,
        entity_id=5,
    )
    circle_tool._shift_held = True
    circle_tool._ctrl_held = True
    mock_element.hittester.screen_to_model.return_value = (50.0, 50.0)
    mock_element.hittester.get_hit_data.return_value = (None, None)

    with patch(
        "sketcher.ui_gtk.tools.circle_tool.EllipseCommand.cleanup_preview"
    ):
        with patch(
            "sketcher.ui_gtk.tools.circle_tool.EllipseCommand"
        ) as MockCmd:
            mock_cmd_instance = Mock()
            MockCmd.return_value = mock_cmd_instance
            circle_tool.on_release(100.0, 200.0)

            call_kwargs = MockCmd.call_args[1]
            assert call_kwargs["center_on_start"] is True
            assert call_kwargs["constrain_circle"] is True


@pytest.mark.ui
def test_circle_tool_on_hover_motion_no_preview(circle_tool):
    """Test on_hover_motion when not in preview stage."""
    circle_tool._preview_state = None
    circle_tool.on_hover_motion(100.0, 200.0)
    assert True


@pytest.mark.ui
def test_circle_tool_on_hover_motion_with_preview(circle_tool, mock_element):
    """Test on_hover_motion updates preview when in preview stage."""
    circle_tool._preview_state = EllipsePreviewState(
        start_id=0,
        start_temp=True,
        center_id=1,
        radius_x_id=2,
        radius_y_id=3,
        entity_id=4,
    )

    mock_point = Mock()
    mock_point.x = 0.0
    mock_point.y = 0.0
    mock_element.sketch.registry.get_point.return_value = mock_point
    mock_element.hittester.screen_to_model.return_value = (10.0, 10.0)

    with patch(
        "sketcher.ui_gtk.tools.circle_tool.EllipseCommand.update_preview"
    ):
        circle_tool.on_hover_motion(100.0, 200.0)

    mock_element.mark_dirty.assert_called()


@pytest.mark.ui
def test_circle_tool_on_hover_motion_with_modifiers(circle_tool, mock_element):
    """Test on_hover_motion passes modifier state to update_preview."""
    circle_tool._preview_state = EllipsePreviewState(
        start_id=0,
        start_temp=True,
        center_id=1,
        radius_x_id=2,
        radius_y_id=3,
        entity_id=4,
    )
    circle_tool._shift_held = True
    circle_tool._ctrl_held = True
    mock_element.hittester.screen_to_model.return_value = (10.0, 10.0)

    with patch(
        "sketcher.ui_gtk.tools.circle_tool.EllipseCommand.update_preview"
    ) as mock_update:
        circle_tool.on_hover_motion(100.0, 200.0)

        call_kwargs = mock_update.call_args[1]
        assert call_kwargs["center_on_start"] is True
        assert call_kwargs["constrain_circle"] is True


@pytest.mark.ui
def test_circle_tool_on_hover_motion_error_deactivates(
    circle_tool, mock_element
):
    """Test on_hover_motion deactivates on IndexError."""
    circle_tool._preview_state = EllipsePreviewState(
        start_id=0,
        start_temp=True,
        center_id=1,
        radius_x_id=2,
        radius_y_id=3,
        entity_id=4,
    )
    mock_element.hittester.screen_to_model.return_value = (10.0, 10.0)

    with patch(
        "sketcher.ui_gtk.tools.circle_tool.EllipseCommand.update_preview",
        side_effect=IndexError,
    ):
        circle_tool.on_hover_motion(100.0, 200.0)

    assert circle_tool._preview_state is None


@pytest.mark.ui
def test_circle_tool_on_hover_motion_key_error_deactivates(
    circle_tool, mock_element
):
    """Test on_hover_motion deactivates on KeyError."""
    circle_tool._preview_state = EllipsePreviewState(
        start_id=0,
        start_temp=True,
        center_id=1,
        radius_x_id=2,
        radius_y_id=3,
        entity_id=4,
    )
    mock_element.hittester.screen_to_model.return_value = (10.0, 10.0)

    with patch(
        "sketcher.ui_gtk.tools.circle_tool.EllipseCommand.update_preview",
        side_effect=KeyError,
    ):
        circle_tool.on_hover_motion(100.0, 200.0)

    assert circle_tool._preview_state is None


@pytest.mark.ui
def test_circle_tool_handle_key_event_escape(circle_tool, mock_element):
    """Test handle_key_event with Escape deactivates."""
    circle_tool._preview_state = EllipsePreviewState(
        start_id=0,
        start_temp=True,
        center_id=1,
        radius_x_id=2,
        radius_y_id=3,
        entity_id=4,
    )

    result = circle_tool.handle_key_event(SketcherKey.ESCAPE)

    assert result is True
    assert circle_tool._preview_state is None


@pytest.mark.ui
def test_circle_tool_handle_key_event_other_key(circle_tool):
    """Test handle_key_event with non-Escape key returns False."""
    circle_tool._preview_state = EllipsePreviewState(
        start_id=0,
        start_temp=True,
        center_id=1,
        radius_x_id=2,
        radius_y_id=3,
        entity_id=4,
    )

    result = circle_tool.handle_key_event(SketcherKey.RETURN)

    assert result is False


@pytest.mark.ui
def test_circle_tool_handle_key_event_no_preview(circle_tool):
    """Test handle_key_event with no preview state returns False."""
    result = circle_tool.handle_key_event(SketcherKey.ESCAPE)
    assert result is False


@pytest.mark.ui
def test_circle_tool_on_modifier_change_no_preview(circle_tool):
    """Test on_modifier_change with no preview does nothing."""
    circle_tool.on_modifier_change(shift=True, ctrl=True)
    assert circle_tool._shift_held is False
    assert circle_tool._ctrl_held is False


@pytest.mark.ui
def test_circle_tool_on_modifier_change_with_preview(
    circle_tool, mock_element
):
    """Test on_modifier_change updates state and marks dirty."""
    circle_tool._preview_state = EllipsePreviewState(
        start_id=0,
        start_temp=True,
        center_id=1,
        radius_x_id=2,
        radius_y_id=3,
        entity_id=4,
    )

    circle_tool.on_modifier_change(shift=True, ctrl=True)

    assert circle_tool._shift_held is True
    assert circle_tool._ctrl_held is True
    mock_element.mark_dirty.assert_called_once()


@pytest.mark.ui
def test_circle_tool_on_modifier_change_no_change(circle_tool, mock_element):
    """Test on_modifier_change doesn't mark dirty if no change."""
    circle_tool._preview_state = EllipsePreviewState(
        start_id=0,
        start_temp=True,
        center_id=1,
        radius_x_id=2,
        radius_y_id=3,
        entity_id=4,
    )
    circle_tool._shift_held = True
    circle_tool._ctrl_held = True

    circle_tool.on_modifier_change(shift=True, ctrl=True)

    mock_element.mark_dirty.assert_not_called()


@pytest.mark.ui
def test_circle_tool_on_modifier_change_partial(circle_tool, mock_element):
    """Test on_modifier_change with only one modifier changed."""
    circle_tool._preview_state = EllipsePreviewState(
        start_id=0,
        start_temp=True,
        center_id=1,
        radius_x_id=2,
        radius_y_id=3,
        entity_id=4,
    )
    circle_tool._shift_held = False
    circle_tool._ctrl_held = False

    circle_tool.on_modifier_change(shift=True, ctrl=False)

    assert circle_tool._shift_held is True
    assert circle_tool._ctrl_held is False
    mock_element.mark_dirty.assert_called_once()


@pytest.mark.ui
def test_circle_tool_get_active_shortcuts_no_preview(circle_tool):
    """Test get_active_shortcuts with no preview."""
    shortcuts = circle_tool.get_active_shortcuts()
    assert shortcuts == []


@pytest.mark.ui
def test_circle_tool_get_active_shortcuts_with_preview(circle_tool):
    """Test get_active_shortcuts returns shortcuts during preview."""
    circle_tool._preview_state = EllipsePreviewState(
        start_id=0,
        start_temp=True,
        center_id=1,
        radius_x_id=2,
        radius_y_id=3,
        entity_id=4,
    )

    shortcuts = circle_tool.get_active_shortcuts()

    assert len(shortcuts) == 4
    keys = [s[0] for s in shortcuts]
    assert "Shift" in keys
    assert "Ctrl" in keys
    assert "Tab" in keys
    assert "Esc" in keys


@pytest.mark.ui
def test_circle_tool_class_attributes():
    """Test CircleTool class attributes."""
    assert CircleTool.ICON == "sketch-circle-symbolic"
    assert CircleTool.LABEL == "Ellipse"
    assert CircleTool.SHORTCUTS == ["gc"]
    assert CircleTool.CURSOR_ICON == "sketch-circle-symbolic"
