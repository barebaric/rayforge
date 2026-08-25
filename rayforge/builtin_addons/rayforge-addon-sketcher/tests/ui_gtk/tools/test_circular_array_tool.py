import math
from unittest.mock import MagicMock, patch

import cairo
import pytest
from sketcher.core.commands import CreatePatternCommand, EditPatternCommand
from sketcher.core.constraints import RotationalConstraint
from sketcher.core.entities import Circle
from sketcher.core.params import ParameterContext
from sketcher.core.patterns import CircularPatternParams, SketchArrayMode
from sketcher.core.patterns.definition import PatternDefinition
from sketcher.core.sketch import Sketch
from sketcher.ui_gtk.tools import CircularArrayTool
from sketcher.ui_gtk.tools.base import SketcherKey


def make_pattern(sketch=None, guide_circle_id=99):
    return PatternDefinition(
        uid="test-uid",
        mode=SketchArrayMode.CIRCULAR,
        guide_circle_id=guide_circle_id,
        members=[],
        count=8,
        total_angle_deg=270.0,
        rotate_copies=False,
    )


@pytest.fixture
def sketch_with_selection():
    sketch = Sketch()
    p0 = sketch.registry.add_point(5, 0)
    p1 = sketch.registry.add_point(25, 5)
    line = sketch.registry.add_line(p0, p1)
    return sketch, line, p0, p1


@pytest.fixture
def mock_element(sketch_with_selection):
    sketch, line, _p0, _p1 = sketch_with_selection
    element = MagicMock()
    element.sketch = sketch
    element.editor = MagicMock()
    element.editor.parent_window = MagicMock()
    element.execute_command = MagicMock()
    element.selection.entity_ids = [line]
    element.hittester.screen_to_model.return_value = (5.0, -7.0)
    return element


@pytest.fixture
def tool(mock_element):
    return CircularArrayTool(mock_element)


def test_initialization(tool):
    assert tool.ICON == "sketch-array-symbolic"
    assert tool.LABEL is not None
    assert "gy" in tool.SHORTCUTS


def test_is_available_requires_selection(tool, mock_element):
    mock_element.selection.entity_ids = []
    assert tool.is_available(None, None) is False
    mock_element.selection.entity_ids = [3]
    assert tool.is_available(None, None) is True


def test_on_activate_without_selection_switches_back(tool, mock_element):
    mock_element.selection.entity_ids = []
    tool.on_activate()
    mock_element.set_tool.assert_called_once_with("select")


@patch.object(CircularArrayTool, "_show_dialog")
def test_on_activate_captures_seeds(
    mock_show_dialog, tool, mock_element, sketch_with_selection
):
    _sketch, line, _p0, _p1 = sketch_with_selection
    tool.on_activate()
    mock_show_dialog.assert_called_once()
    assert tool._seed_entity_ids == [line]


@patch.object(CircularArrayTool, "_show_dialog")
def test_default_params_derived_from_seed(
    mock_show_dialog, tool, sketch_with_selection
):
    _sketch, _line, _p0, _p1 = sketch_with_selection
    tool.on_activate()
    params = tool._params
    assert isinstance(params, CircularPatternParams)
    assert params.count == 6
    assert params.total_angle_deg == 360.0
    assert params.rotate_copies is True
    # Radius is derived from the anchor point (first seed point here),
    # matching what CreatePatternCommand will apply.
    assert params.radius == pytest.approx(5.0)


@patch.object(CircularArrayTool, "_show_dialog")
def test_creation_radius_always_matches_anchor(
    mock_show_dialog, tool, mock_element, sketch_with_selection
):
    """User-entered radius must not diverge from the anchor-derived
    circle at creation time (that mismatch caused the jump on apply)."""
    tool.on_activate()
    tool._count_row = MagicMock()
    tool._count_row.get_int_value.return_value = 6
    tool._angle_row = MagicMock()
    tool._angle_row.get_value.return_value = 360.0
    tool._center_x_row = MagicMock()
    tool._center_x_row.get_value_in_base_units.return_value = 0.0
    tool._center_y_row = MagicMock()
    tool._center_y_row.get_value_in_base_units.return_value = 0.0
    tool._radius_row = MagicMock()
    tool._radius_row.get_value_in_base_units.return_value = 200.0
    tool._rotate_switch = MagicMock()
    tool._rotate_switch.get_active.return_value = True
    assert tool._params is not None
    tool._params.radius = 200.0
    tool._sync_params()
    assert tool._params.radius == pytest.approx(5.0)


@patch.object(CircularArrayTool, "_show_dialog")
def test_collect_command_builds_create_command(
    mock_show_dialog, tool, sketch_with_selection
):
    _sketch, line, _p0, _p1 = sketch_with_selection
    tool.on_activate()
    cmd = tool._collect_command()
    assert isinstance(cmd, CreatePatternCommand)
    assert cmd.seed_entity_ids == [line]
    assert not isinstance(cmd, EditPatternCommand)


@patch.object(CircularArrayTool, "_show_dialog")
def test_edit_target_builds_edit_command(
    mock_show_dialog, tool, mock_element, sketch_with_selection
):
    """Edit mode pre-fills params from the master and builds an edit."""
    sketch, line, _p0, _p1 = sketch_with_selection

    # Master circle geometry: center (10, 2), radius point (30, 2).
    c = sketch.registry.add_point(10.0, 2.0)
    r = sketch.registry.add_point(30.0, 2.0)
    circle = sketch.registry.add_circle(c, r, construction=True)

    pattern = make_pattern(sketch, guide_circle_id=circle)
    pattern.members = [(0, [line])]
    pattern.count = 8
    pattern.total_angle_deg = 270.0
    pattern.rotate_copies = False

    tool.set_edit_target(pattern)
    tool.on_activate()

    assert tool._is_editing
    params = tool._params
    assert params.count == 8
    assert params.total_angle_deg == 270.0
    assert params.rotate_copies is False
    assert params.center == pytest.approx((10.0, 2.0))
    assert params.radius == pytest.approx(20.0)

    cmd = tool._collect_command()
    assert isinstance(cmd, EditPatternCommand)
    assert cmd.pattern is pattern


@patch.object(CircularArrayTool, "_show_dialog")
def test_apply_executes_command_and_selects_copies(
    mock_show_dialog, tool, mock_element, sketch_with_selection
):
    sketch, _line, _p0, _p1 = sketch_with_selection
    tool.on_activate()

    created = MagicMock()
    created.id = 42
    created.pattern_copy = True

    fake_cmd = MagicMock()
    fake_cmd.created_entity_ids = [42]
    mock_element.execute_command.side_effect = lambda cmd: None

    with (
        patch.object(tool, "_collect_command", return_value=fake_cmd),
        patch.object(tool, "_close_dialog"),
        patch.object(
            sketch.registry,
            "get_entity",
            side_effect=lambda eid: created if eid == 42 else None,
        ),
    ):
        tool._on_apply()

    mock_element.set_tool.assert_called_once_with("select")
    mock_element.selection.select_entity.assert_called_once_with(
        created, False
    )


def test_draw_overlay_noop_without_dialog(tool):
    tool._dialog = None
    tool._params = None
    surface = cairo.ImageSurface(cairo.FORMAT_ARGB32, 10, 10)
    ctx = cairo.Context(surface)
    tool.draw_overlay(ctx)


@patch.object(CircularArrayTool, "_show_dialog")
def test_on_press_does_not_move_center(
    mock_show_dialog, tool, mock_element, sketch_with_selection
):
    """Canvas clicks must never relocate the pattern center (they
    conflict with panning/navigation while the dialog is open)."""
    tool.on_activate()
    tool._dialog = MagicMock()
    center_before = tool._params.center

    tool.on_press(100.0, 200.0, 1)

    assert tool._params.center == center_before
    mock_element.mark_dirty.assert_not_called()


def test_escape_deactivates_tool(tool, mock_element):
    tool._dialog = MagicMock()
    handled = tool.handle_key_event(SketcherKey.ESCAPE)
    assert handled is True
    mock_element.set_tool.assert_called_once_with("select")


def test_close_request_deactivates_tool_and_allows_close(tool, mock_element):
    tool._dialog = MagicMock()
    assert tool._on_close_request() is False
    assert tool._dialog is None
    mock_element.set_tool.assert_called_once_with("select")


@patch.object(CircularArrayTool, "_show_dialog")
def test_full_edit_flow_through_tool_does_not_collapse(
    mock_show_dialog,
):
    """
    Integration: double-click edit flow (tool -> EditPatternCommand)
    must leave all members at distinct slot angles with zero constraint
    residual - never collapsed onto one position.
    """
    sketch = Sketch()
    p0 = sketch.registry.add_point(30, 0)
    p1 = sketch.registry.add_point(50, 0)
    line = sketch.registry.add_line(p0, p1)
    params = CircularPatternParams(
        count=6,
        total_angle_deg=360.0,
        center=(0.0, 0.0),
        radius=40.0,
        rotate_copies=True,
    )
    create_cmd = CreatePatternCommand(
        sketch, SketchArrayMode.CIRCULAR, params, [line]
    )
    create_cmd.execute()
    sketch.solve()
    pattern = sketch.patterns[0]

    element = MagicMock()
    element.sketch = sketch
    element.editor.parent_window = MagicMock()
    element.editor.history_manager.execute.side_effect = lambda cmd: (
        cmd.execute()
    )
    element.execute_command.side_effect = lambda cmd: cmd.execute()
    element.selection.entity_ids = []
    element.hittester.screen_to_model.return_value = (0.0, 0.0)

    tool = CircularArrayTool(element)
    tool.set_edit_target(pattern)
    tool.on_activate()

    # User changes the count in the dialog and hits Apply.
    assert tool._params is not None
    tool._params.count = 8
    tool._on_apply()

    sketch.solve()

    circle = next(e for e in sketch.registry.entities if isinstance(e, Circle))
    center = sketch.registry.get_point(circle.center_idx)

    living = pattern.living_entity_ids(sketch.registry)
    assert len(living) == 8

    angles = []
    for eid in living:
        entity = sketch.registry.get_entity(eid)
        assert entity is not None
        pid = entity.get_point_ids()[0]
        pt = sketch.registry.get_point(pid)
        angles.append(
            round(
                math.degrees(math.atan2(pt.y - center.y, pt.x - center.x))
                / 45.0
            )
            * 45.0
            % 360.0
        )
    assert sorted(set(angles)) == [
        0.0,
        45.0,
        90.0,
        135.0,
        180.0,
        225.0,
        270.0,
        315.0,
    ]

    worst = max(
        max(abs(v) for v in c.error(sketch.registry, ParameterContext()))
        for c in sketch.constraints
        if isinstance(c, RotationalConstraint)
    )
    assert worst < 1e-6


@patch.object(CircularArrayTool, "_show_dialog")
def test_cancel_clears_edit_target(
    mock_show_dialog, tool, mock_element, sketch_with_selection
):
    pattern = make_pattern()
    tool.set_edit_target(pattern)
    tool.on_activate()
    assert tool._is_editing

    tool.on_deactivate()
    assert tool._edit_target is None
