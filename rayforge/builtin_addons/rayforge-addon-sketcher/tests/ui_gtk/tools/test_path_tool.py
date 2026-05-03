import pytest
from unittest.mock import Mock, MagicMock, patch
from sketcher.ui_gtk.tools.path_tool import PathTool
from sketcher.core.commands.bezier import BezierPreviewState
from sketcher.core.constraints import HorizontalConstraint, VerticalConstraint
from sketcher.core.entities import Point as SketchPoint
from sketcher.core.snap.types import SnapLine, SnapLineType, SnapResult


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
def path_tool(mock_element):
    """Create a PathTool instance for testing."""
    return PathTool(mock_element)


@pytest.mark.ui
def test_path_tool_initialization(path_tool, mock_element):
    """Test that BezierTool initializes correctly."""
    assert path_tool.element == mock_element
    assert path_tool._preview_state is None
    assert path_tool._press_pos is None
    assert path_tool._dragging is False


@pytest.mark.ui
def test_path_tool_on_deactivate_no_preview(path_tool, mock_element):
    """Test that on_deactivate works when no preview state."""
    path_tool.on_deactivate()
    assert path_tool._preview_state is None
    assert path_tool._press_pos is None
    assert path_tool._dragging is False


@pytest.mark.ui
def test_path_tool_on_deactivate_with_preview(path_tool, mock_element):
    """Test that on_deactivate cleans up preview state."""
    path_tool._preview_state = BezierPreviewState(
        start_id=1,
        start_temp=True,
        end_id=2,
        end_temp=True,
        temp_entity_id=3,
        is_line_preview=True,
    )

    with patch(
        "sketcher.ui_gtk.tools.path_tool.BezierCommand.cleanup_preview"
    ):
        path_tool.on_deactivate()

    assert path_tool._preview_state is None
    mock_element.remove_point_if_unused.assert_called_once_with(1)


@pytest.mark.ui
def test_path_tool_first_press_starts_preview(path_tool, mock_element):
    """Test first press starts line preview."""
    mock_element.hittester.get_hit_data.return_value = (None, None)
    mock_element.hittester.screen_to_model.return_value = (10.0, 20.0)

    with patch(
        "sketcher.ui_gtk.tools.path_tool.BezierCommand.start_preview"
    ) as mock_start:
        mock_start.return_value = BezierPreviewState(
            start_id=0,
            start_temp=True,
            end_id=1,
            end_temp=True,
            temp_entity_id=2,
            is_line_preview=True,
        )
        result = path_tool.on_press(100.0, 200.0, 1)

    assert result is False
    assert path_tool._preview_state is not None
    assert path_tool._waypoint_model_pos == (10.0, 20.0)
    assert path_tool._press_pos == (100.0, 200.0)
    assert path_tool._dragging is False


@pytest.mark.ui
def test_path_tool_on_drag_below_threshold(path_tool, mock_element):
    """Test on_drag does nothing below threshold."""
    path_tool._press_pos = (100.0, 200.0)
    path_tool._waypoint_model_pos = (10.0, 20.0)
    path_tool._preview_state = BezierPreviewState(
        start_id=0,
        start_temp=True,
        end_id=1,
        end_temp=True,
        temp_entity_id=2,
        is_line_preview=True,
    )

    path_tool.on_drag(2.0, 2.0)

    assert path_tool._dragging is False


@pytest.mark.ui
def test_path_tool_on_drag_starts_bezier(path_tool, mock_element):
    """Test on_drag converts line to bezier above threshold."""
    path_tool._press_pos = (100.0, 200.0)
    path_tool._waypoint_model_pos = (10.0, 20.0)
    path_tool._preview_state = BezierPreviewState(
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
        "sketcher.ui_gtk.tools.path_tool.BezierCommand.convert_to_bezier"
    ) as mock_convert:
        path_tool.on_drag(10.0, 10.0)

        mock_convert.assert_called_once_with(
            mock_element.sketch.registry,
            path_tool._preview_state,
            10.0,
            20.0,
            15.0,
            25.0,
            mirror_cp_offset=None,
        )

    assert path_tool._dragging is True


@pytest.mark.ui
def test_path_tool_on_release_without_drag_creates_line(
    path_tool, mock_element
):
    """Test on_release creates line when not dragging and end has moved."""
    path_tool._press_pos = (100.0, 200.0)
    path_tool._waypoint_model_pos = (20.0, 30.0)
    path_tool._dragging = False
    path_tool._snapped_pid = None
    path_tool._preview_state = BezierPreviewState(
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
        "sketcher.ui_gtk.tools.path_tool.BezierCommand.cleanup_preview"
    ):
        path_tool.on_release(100.0, 200.0)

        mock_element.execute_command.assert_called_once()
        cmd = mock_element.execute_command.call_args[0][0]
        assert cmd.is_line is True


@pytest.mark.ui
def test_path_tool_on_release_with_drag_creates_bezier(
    path_tool, mock_element
):
    """Test on_release creates bezier when dragging."""
    path_tool._press_pos = (100.0, 200.0)
    path_tool._waypoint_model_pos = (10.0, 20.0)
    path_tool._dragging = True
    path_tool._snapped_pid = None
    path_tool._preview_state = BezierPreviewState(
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
        "sketcher.ui_gtk.tools.path_tool.BezierCommand.cleanup_preview"
    ):
        path_tool.on_release(100.0, 200.0)

        mock_element.execute_command.assert_called_once()
        cmd = mock_element.execute_command.call_args[0][0]
        assert cmd.is_line is False


@pytest.mark.ui
def test_path_tool_on_hover_motion_updates_preview(path_tool, mock_element):
    """Test on_hover_motion updates preview when in preview stage."""
    path_tool._preview_state = BezierPreviewState(
        start_id=0,
        start_temp=True,
        end_id=1,
        end_temp=True,
        temp_entity_id=2,
        is_line_preview=True,
    )
    path_tool._in_press = False
    mock_element.hittester.screen_to_model.return_value = (15.0, 25.0)

    with patch("sketcher.ui_gtk.tools.path_tool.BezierCommand.update_preview"):
        path_tool.on_hover_motion(100.0, 200.0)

    mock_element.mark_dirty.assert_called()


@pytest.mark.ui
def test_path_tool_on_hover_motion_skips_when_in_press(
    path_tool, mock_element
):
    """Test on_hover_motion skips when in press sequence."""
    path_tool._preview_state = BezierPreviewState(
        start_id=0,
        start_temp=True,
        end_id=1,
        end_temp=True,
        temp_entity_id=2,
        is_line_preview=False,
    )
    path_tool._in_press = True

    path_tool.on_hover_motion(100.0, 200.0)

    mock_element.mark_dirty.assert_not_called()


@pytest.mark.ui
def test_path_tool_shortcut(path_tool):
    """Test that PathTool has correct shortcut."""
    assert PathTool.SHORTCUTS == ["gp", "gl"]


def _setup_line_finalization(path_tool, mock_element, start_id=0):
    path_tool._press_pos = (100.0, 200.0)
    path_tool._dragging = False
    path_tool._snapped_pid = None
    path_tool._preview_state = BezierPreviewState(
        start_id=start_id,
        start_temp=True,
        end_id=1,
        end_temp=True,
        temp_entity_id=2,
        is_line_preview=True,
    )
    start_pt = SketchPoint(start_id, 0.0, 0.0)
    end_pt = MagicMock(x=100.0, y=0.0)

    def get_point(pid):
        if pid == start_id:
            return start_pt
        return end_pt

    mock_element.sketch.registry.get_point.side_effect = get_point
    mock_element.sketch.registry._id_counter = 10
    mock_element.sketch.constraints = []
    return start_pt


@pytest.mark.ui
def test_auto_constraint_horizontal(path_tool, mock_element):
    """Line snapped to horizontal guide gets a HorizontalConstraint."""
    start_pt = _setup_line_finalization(path_tool, mock_element)
    path_tool.current_snap_result = SnapResult(
        snapped=True,
        position=(100.0, 0.0),
        snap_lines=[
            SnapLine(
                is_horizontal=True,
                coordinate=0.0,
                line_type=SnapLineType.ENTITY_POINT,
                source=start_pt,
            ),
        ],
    )

    with patch(
        "sketcher.ui_gtk.tools.path_tool.BezierCommand.cleanup_preview"
    ):
        path_tool.on_release(100.0, 200.0)

        mock_element.execute_command.assert_called_once()
        cmd = mock_element.execute_command.call_args[0][0]
        horiz = [
            c for c in cmd.constraints
            if isinstance(c, HorizontalConstraint)
        ]
        assert len(horiz) == 1
        assert horiz[0].p1 == 0


@pytest.mark.ui
def test_auto_constraint_vertical(path_tool, mock_element):
    """Line snapped to vertical guide gets a VerticalConstraint."""
    start_pt = _setup_line_finalization(path_tool, mock_element)
    path_tool.current_snap_result = SnapResult(
        snapped=True,
        position=(0.0, 100.0),
        snap_lines=[
            SnapLine(
                is_horizontal=False,
                coordinate=0.0,
                line_type=SnapLineType.ENTITY_POINT,
                source=start_pt,
            ),
        ],
    )

    with patch(
        "sketcher.ui_gtk.tools.path_tool.BezierCommand.cleanup_preview"
    ):
        path_tool.on_release(100.0, 200.0)

        mock_element.execute_command.assert_called_once()
        cmd = mock_element.execute_command.call_args[0][0]
        vert = [
            c for c in cmd.constraints
            if isinstance(c, VerticalConstraint)
        ]
        assert len(vert) == 1
        assert vert[0].p1 == 0


@pytest.mark.ui
def test_auto_constraint_no_snap_no_constraint(
    path_tool, mock_element
):
    """Line without snap result gets no axis constraint."""
    _setup_line_finalization(path_tool, mock_element)
    path_tool.current_snap_result = None

    with patch(
        "sketcher.ui_gtk.tools.path_tool.BezierCommand.cleanup_preview"
    ):
        path_tool.on_release(100.0, 200.0)

        mock_element.execute_command.assert_called_once()
        cmd = mock_element.execute_command.call_args[0][0]
        axis = [
            c for c in cmd.constraints
            if isinstance(c, (HorizontalConstraint, VerticalConstraint))
        ]
        assert len(axis) == 0


@pytest.mark.ui
def test_auto_constraint_from_any_existing_point(
    path_tool, mock_element
):
    """Snap line from any point (not just start) creates constraint."""
    _setup_line_finalization(path_tool, mock_element, start_id=0)
    other_pt = SketchPoint(99, 50.0, 0.0)
    mock_element.sketch.constraints = []
    path_tool.current_snap_result = SnapResult(
        snapped=True,
        position=(100.0, 0.0),
        snap_lines=[
            SnapLine(
                is_horizontal=True,
                coordinate=0.0,
                line_type=SnapLineType.ENTITY_POINT,
                source=other_pt,
            ),
        ],
    )

    with patch(
        "sketcher.ui_gtk.tools.path_tool.BezierCommand.cleanup_preview"
    ):
        path_tool.on_release(100.0, 200.0)

        mock_element.execute_command.assert_called_once()
        cmd = mock_element.execute_command.call_args[0][0]
        horiz = [
            c for c in cmd.constraints
            if isinstance(c, HorizontalConstraint)
        ]
        assert len(horiz) == 1
        assert horiz[0].p1 == 99


@pytest.mark.ui
def test_auto_constraint_skips_duplicate(
    path_tool, mock_element
):
    """Does not add axis constraint if one already exists."""
    _setup_line_finalization(path_tool, mock_element, start_id=0)
    start_pt = SketchPoint(0, 0.0, 0.0)
    existing = [HorizontalConstraint(0, 10)]
    mock_element.sketch.constraints = existing
    path_tool.current_snap_result = SnapResult(
        snapped=True,
        position=(100.0, 0.0),
        snap_lines=[
            SnapLine(
                is_horizontal=True,
                coordinate=0.0,
                line_type=SnapLineType.ENTITY_POINT,
                source=start_pt,
            ),
        ],
    )

    with patch(
        "sketcher.ui_gtk.tools.path_tool.BezierCommand.cleanup_preview"
    ):
        path_tool.on_release(100.0, 200.0)

        mock_element.execute_command.assert_called_once()
        cmd = mock_element.execute_command.call_args[0][0]
        horiz = [
            c for c in cmd.constraints
            if isinstance(c, HorizontalConstraint)
        ]
        assert len(horiz) == 0


@pytest.mark.ui
def test_auto_constraint_skips_self_constraint(
    path_tool, mock_element
):
    """Does not add constraint where source equals end point."""
    end_pt = SketchPoint(99, 100.0, 0.0)
    _setup_line_finalization(path_tool, mock_element, start_id=0)
    mock_element.sketch.constraints = []
    path_tool._snapped_pid = 99

    def get_point(pid):
        if pid == 0:
            return SketchPoint(0, 0.0, 0.0)
        return end_pt

    mock_element.sketch.registry.get_point.side_effect = get_point
    path_tool.current_snap_result = SnapResult(
        snapped=True,
        position=(100.0, 0.0),
        snap_lines=[
            SnapLine(
                is_horizontal=True,
                coordinate=0.0,
                line_type=SnapLineType.ENTITY_POINT,
                source=end_pt,
            ),
        ],
    )

    with patch(
        "sketcher.ui_gtk.tools.path_tool.BezierCommand.cleanup_preview"
    ):
        path_tool.on_release(100.0, 200.0)

        mock_element.execute_command.assert_called_once()
        cmd = mock_element.execute_command.call_args[0][0]
        axis = [
            c for c in cmd.constraints
            if isinstance(c, (HorizontalConstraint, VerticalConstraint))
        ]
        assert len(axis) == 0
