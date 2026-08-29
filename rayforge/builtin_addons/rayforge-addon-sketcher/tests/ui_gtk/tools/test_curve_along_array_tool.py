from unittest.mock import MagicMock, patch

import cairo
import pytest
from sketcher.core.arrays import (
    CurveAlongArray,
    CurveAlongArrayStrategy,
    path_length,
)
from sketcher.core.commands import CreateArrayCommand, EditArrayCommand
from sketcher.core.sketch import Sketch
from sketcher.ui_gtk.tools import CurveAlongArrayTool
from sketcher.ui_gtk.tools.base import SketcherKey


@pytest.fixture
def sketch_with_template_and_path():
    """A horizontal line seed and a path line to distribute along."""
    sketch = Sketch()
    # Guide path: a horizontal line from (0,0) to (30,0). Selected FIRST.
    pp0 = sketch.registry.add_point(0.0, 0.0)
    pp1 = sketch.registry.add_point(30.0, 0.0)
    path = sketch.registry.add_line(pp0, pp1)
    # Seed: a small vertical line at the origin. Selected SECOND.
    sp0 = sketch.registry.add_point(0.0, -2.0)
    sp1 = sketch.registry.add_point(0.0, 2.0)
    seed = sketch.registry.add_line(sp0, sp1)
    return sketch, seed, path


@pytest.fixture
def mock_element(sketch_with_template_and_path):
    sketch, seed, path = sketch_with_template_and_path
    element = MagicMock()
    element.sketch = sketch
    element.editor = MagicMock()
    element.editor.parent_window = MagicMock()
    element.execute_command = MagicMock()
    # Selection order: [guide path, seed] -> path first, seed second.
    element.selection.entity_ids = [path, seed]
    element.hittester.screen_to_model.return_value = (0.0, 0.0)
    return element


@pytest.fixture
def tool(mock_element):
    return CurveAlongArrayTool(mock_element)


def test_initialization(tool):
    assert tool.ICON is not None
    assert tool.LABEL is not None
    assert "gw" in tool.SHORTCUTS
    assert tool.ARRAY_TYPE is CurveAlongArray


def test_is_available_requires_two_selections(tool, mock_element):
    mock_element.selection.entity_ids = [3]
    assert tool.is_available(None, None) is False
    mock_element.selection.entity_ids = [3, 4]
    assert tool.is_available(None, None) is True


def test_on_activate_without_selection_switches_back(tool, mock_element):
    mock_element.selection.entity_ids = [3]
    tool.on_activate()
    mock_element.set_tool.assert_called_once_with("select")


@patch.object(CurveAlongArrayTool, "_show_dialog")
def test_on_activate_splits_selection_into_guide_and_template(
    mock_show_dialog, tool, mock_element, sketch_with_template_and_path
):
    _sketch, seed, path = sketch_with_template_and_path
    tool.on_activate()
    mock_show_dialog.assert_called_once()
    assert tool._path_entity_id == path
    assert tool._template_entity_ids == [seed]


@patch.object(CurveAlongArrayTool, "_show_dialog")
def test_default_params_use_first_selected_as_path(
    mock_show_dialog, tool, sketch_with_template_and_path
):
    _sketch, _seed, path = sketch_with_template_and_path
    tool.on_activate()
    strategy = tool._strategy
    assert isinstance(strategy, CurveAlongArrayStrategy)
    assert strategy.count == 6
    assert strategy.path_entity_id == path
    assert strategy.align_to_tangent is True
    assert strategy.spacing == 0.0


@patch.object(CurveAlongArrayTool, "_show_dialog")
def test_collect_command_builds_create_command(
    mock_show_dialog, tool, sketch_with_template_and_path
):
    _sketch, seed, _path = sketch_with_template_and_path
    tool.on_activate()
    cmd = tool._collect_command()
    assert isinstance(cmd, CreateArrayCommand)
    assert isinstance(cmd.strategy, CurveAlongArrayStrategy)
    assert cmd.template_entity_ids == [seed]
    assert not isinstance(cmd, EditArrayCommand)


@patch.object(CurveAlongArrayTool, "_show_dialog")
def test_edit_target_builds_edit_command(
    mock_show_dialog, tool, mock_element, sketch_with_template_and_path
):
    _sketch, seed, path = sketch_with_template_and_path
    array = CurveAlongArray(
        uid="test",
        guide_circle_id=path,
        members=[(0, [seed])],
        count=5,
        path_entity_id=path,
        align_to_tangent=False,
        offset_to_start=2.0,
        spacing=5.0,
    )
    tool.set_edit_target(array)
    tool.on_activate()

    assert tool._is_editing
    params = tool._strategy
    assert params.count == 5
    assert params.path_entity_id == path
    assert params.align_to_tangent is False
    assert params.offset_to_start == 2.0
    assert params.spacing == 5.0

    cmd = tool._collect_command()
    assert isinstance(cmd, EditArrayCommand)
    assert cmd.array is array


def test_draw_overlay_noop_without_dialog(tool):
    tool._dialog = None
    tool._strategy = None
    surface = cairo.ImageSurface(cairo.Format.ARGB32, 10, 10)
    ctx = cairo.Context(surface)
    tool.draw_overlay(ctx)


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


@patch.object(CurveAlongArrayTool, "_show_dialog")
def test_cancel_clears_edit_target(
    mock_show_dialog, tool, mock_element, sketch_with_template_and_path
):
    _sketch, _seed, path = sketch_with_template_and_path
    array = CurveAlongArray(
        uid="t",
        guide_circle_id=path,
        members=[],
        count=4,
        path_entity_id=path,
    )
    tool.set_edit_target(array)
    tool.on_activate()
    assert tool._is_editing
    tool.on_deactivate()
    assert tool._edit_target is None


@patch.object(CurveAlongArrayTool, "_show_dialog")
def test_on_press_does_not_move_path(
    mock_show_dialog, tool, mock_element, sketch_with_template_and_path
):
    tool.on_activate()
    tool._dialog = MagicMock()
    path_before = tool._strategy.path_entity_id
    tool.on_press(100.0, 200.0, 1)
    assert tool._strategy.path_entity_id == path_before


# ----------------------------------------------------------------------
# Spacing <-> count linkage
# ----------------------------------------------------------------------


def _mock_spin_row(value):
    row = MagicMock()
    row.get_value_in_base_units.return_value = value
    row.get_int_value.return_value = int(value)
    return row


def _setup_mock_rows(tool, count=None, spacing=None, offset=None, align=True):
    """Creates mock dialog rows for tools whose _show_dialog is patched."""
    tool._count_row = _mock_spin_row(count if count is not None else 6)
    tool._spacing_row = _mock_spin_row(spacing if spacing is not None else 0.0)
    tool._offset_row = _mock_spin_row(offset if offset is not None else 0.0)
    tool._align_switch = MagicMock()
    tool._align_switch.get_active.return_value = align


@patch.object(CurveAlongArrayTool, "_show_dialog")
def test_spacing_drives_count_from_path_length(
    mock_show_dialog, tool, sketch_with_template_and_path
):
    """Setting spacing derives count from usable path length."""
    _sketch, _seed, _path = sketch_with_template_and_path
    tool.on_activate()
    # Simulate the spacing row reporting 10.0 on a 30-unit path:
    # 30/10 + 1 = 4 copies.
    _setup_mock_rows(tool, spacing=10.0, offset=0.0)
    tool._on_spacing_changed()
    assert tool._strategy.count == 4


@patch.object(CurveAlongArrayTool, "_show_dialog")
def test_count_change_clears_spacing(
    mock_show_dialog, tool, sketch_with_template_and_path
):
    """Editing count resets spacing to 0 (count takes over)."""
    _sketch, _seed, _path = sketch_with_template_and_path
    tool.on_activate()
    tool._strategy.spacing = 10.0
    _setup_mock_rows(tool, count=8, spacing=10.0)
    tool._on_count_changed()
    assert tool._strategy.spacing == 0.0


@patch.object(CurveAlongArrayTool, "_show_dialog")
def test_spacing_respects_start_offset(
    mock_show_dialog, tool, sketch_with_template_and_path
):
    _sketch, _seed, _path = sketch_with_template_and_path
    tool.on_activate()
    # Path 30, offset 10 -> usable 20. Spacing 5 -> 20/5 + 1 = 5.
    _setup_mock_rows(tool, spacing=5.0, offset=10.0)
    tool._on_spacing_changed()
    assert tool._strategy.count == 5


# ----------------------------------------------------------------------
# Integration
# ----------------------------------------------------------------------


@patch.object(CurveAlongArrayTool, "_show_dialog")
def test_full_edit_flow_does_not_collapse(mock_show_dialog):
    """Integration: editing a curve array through the tool keeps
    members distributed along the path."""
    sketch = Sketch()
    pp0 = sketch.registry.add_point(0.0, 0.0)
    pp1 = sketch.registry.add_point(40.0, 0.0)
    path = sketch.registry.add_line(pp0, pp1)
    sp0 = sketch.registry.add_point(0.0, -1.0)
    sp1 = sketch.registry.add_point(0.0, 1.0)
    seed = sketch.registry.add_line(sp0, sp1)

    strategy = CurveAlongArrayStrategy(
        count=4,
        path_entity_id=path,
        align_to_tangent=True,
    )
    create_cmd = CreateArrayCommand(sketch, strategy, [seed])
    create_cmd.execute()
    array_def = sketch.arrays[0]

    element = MagicMock()
    element.sketch = sketch
    element.editor.parent_window = MagicMock()
    element.execute_command.side_effect = lambda cmd: cmd.execute()
    element.selection.entity_ids = []
    element.hittester.screen_to_model.return_value = (0.0, 0.0)

    tool = CurveAlongArrayTool(element)
    tool.set_edit_target(array_def)
    tool.on_activate()

    assert tool._strategy is not None
    tool._strategy.count = 6
    tool._on_apply()

    living = array_def.living_entity_ids(sketch.registry)
    assert len(living) == 6

    # Each member sits at a distinct x along the 40-unit path.
    centers_x = []
    for eid in living:
        entity = sketch.registry.get_entity(eid)
        assert entity is not None
        pids = entity.get_point_ids()
        xs = [sketch.registry.get_point(p).x for p in pids]
        centers_x.append(round(sum(xs) / len(xs), 1))
    assert sorted(centers_x) == [0.0, 8.0, 16.0, 24.0, 32.0, 40.0]


# ----------------------------------------------------------------------
# path_length helper
# ----------------------------------------------------------------------


def test_path_length_measures_line(sketch_with_template_and_path):
    sketch, _seed, path = sketch_with_template_and_path
    assert path_length(sketch.registry, path) == pytest.approx(30.0)


def test_path_length_zero_for_missing_entity(sketch_with_template_and_path):
    sketch, _seed, _path = sketch_with_template_and_path
    assert path_length(sketch.registry, 9999) == 0.0
