# flake8: noqa: E402
import os
import sys

import pytest
from unittest.mock import MagicMock

if sys.platform.startswith("linux"):
    os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
    if not os.environ.get("DISPLAY"):
        pytest.skip(
            "DISPLAY not set on Linux, skipping UI tests. Run with xvfb-run.",
            allow_module_level=True,
        )

import gi

gi.require_version("Gtk", "4.0")
gi.require_version("Adw", "1")

from rayforge.core.sketcher import Sketch
from rayforge.core.sketcher.commands import ChamferCommand, FilletCommand
from rayforge.core.sketcher.entities import Line
from rayforge.ui_gtk.sketcher.sketchelement import SketchElement


@pytest.fixture
def sketch_with_corner():
    """Creates a sketch with two lines forming a corner at (0,0)."""
    s = Sketch()
    p1_id = s.add_point(-100, 0)
    corner_pid = s.add_point(0, 0)
    p3_id = s.add_point(0, 100)

    line1_id = s.add_line(p1_id, corner_pid)
    line2_id = s.add_line(corner_pid, p3_id)

    return s, corner_pid, line1_id, line2_id


@pytest.fixture
def element_with_corner(sketch_with_corner):
    """Creates a SketchElement with a corner and mocked editor."""
    sketch, _, _, _ = sketch_with_corner
    element = SketchElement(sketch=sketch)
    element.editor = MagicMock()
    element.execute_command = element.editor.history_manager.execute
    return element


@pytest.mark.ui
def test_is_action_supported_chamfer_valid(
    element_with_corner, sketch_with_corner
):
    """Test chamfer is supported when a valid corner junction is selected."""
    _, corner_pid, _, _ = sketch_with_corner
    element_with_corner.selection.select_junction(corner_pid, is_multi=False)

    assert element_with_corner.is_action_supported("chamfer") is True


@pytest.mark.parametrize(
    "setup_selection",
    [
        "no_selection",
        "point_selection",
        "single_line_junction",
        "triple_line_junction",
    ],
)
@pytest.mark.ui
def test_is_action_supported_chamfer_invalid(setup_selection):
    """Test chamfer is not supported for invalid selections."""
    s = Sketch()
    p1 = s.add_point(0, 0)
    p2 = s.add_point(10, 0)
    p3 = s.add_point(10, 10)
    p4 = s.add_point(0, 10)
    s.add_line(p1, p2)

    element = SketchElement(sketch=s)

    if setup_selection == "no_selection":
        pass
    elif setup_selection == "point_selection":
        element.selection.select_point(p1, is_multi=False)
    elif setup_selection == "single_line_junction":
        element.selection.select_junction(p1, is_multi=False)
    elif setup_selection == "triple_line_junction":
        s.add_line(p1, p3)
        s.add_line(p1, p4)
        element.selection.select_junction(p1, is_multi=False)

    assert element.is_action_supported("chamfer") is False


@pytest.mark.ui
def test_add_chamfer_action_executes_command(
    element_with_corner, sketch_with_corner
):
    """Test that add_chamfer_action creates and executes a ChamferCommand."""
    _, corner_pid, _, _ = sketch_with_corner
    element_with_corner.selection.select_junction(corner_pid, is_multi=False)

    element_with_corner.add_chamfer_action()

    element_with_corner.execute_command.assert_called_once()
    command_instance = element_with_corner.execute_command.call_args[0][0]
    assert isinstance(command_instance, ChamferCommand)


@pytest.mark.ui
def test_add_chamfer_action_on_short_lines(
    element_with_corner, sketch_with_corner
):
    """Test that chamfer action is aborted if lines are too short."""
    sketch, corner_pid, line1_id, _ = sketch_with_corner
    element = element_with_corner

    line1 = sketch.registry.get_entity(line1_id)

    assert isinstance(line1, Line)
    p1 = sketch.registry.get_point(line1.p1_idx)
    p1.x = -1e-7

    element.selection.select_junction(corner_pid, is_multi=False)
    element.add_chamfer_action()

    element.execute_command.assert_not_called()


@pytest.mark.ui
def test_is_action_supported_fillet_valid(
    element_with_corner, sketch_with_corner
):
    """Test fillet is supported when a valid corner junction is selected."""
    _, corner_pid, _, _ = sketch_with_corner
    element_with_corner.selection.select_junction(corner_pid, is_multi=False)

    assert element_with_corner.is_action_supported("fillet") is True


@pytest.mark.parametrize(
    "setup_selection",
    [
        "no_selection",
        "point_selection",
        "single_line_junction",
        "triple_line_junction",
    ],
)
@pytest.mark.ui
def test_is_action_supported_fillet_invalid(setup_selection):
    """Test fillet is not supported for invalid selections."""
    s = Sketch()
    p1 = s.add_point(0, 0)
    p2 = s.add_point(10, 0)
    p3 = s.add_point(10, 10)
    p4 = s.add_point(0, 10)
    s.add_line(p1, p2)

    element = SketchElement(sketch=s)

    if setup_selection == "no_selection":
        pass
    elif setup_selection == "point_selection":
        element.selection.select_point(p1, is_multi=False)
    elif setup_selection == "single_line_junction":
        element.selection.select_junction(p1, is_multi=False)
    elif setup_selection == "triple_line_junction":
        s.add_line(p1, p3)
        s.add_line(p1, p4)
        element.selection.select_junction(p1, is_multi=False)

    assert element.is_action_supported("fillet") is False


@pytest.mark.ui
def test_add_fillet_action_executes_command(
    element_with_corner, sketch_with_corner
):
    """Test that add_fillet_action creates and executes a FilletCommand."""
    _, corner_pid, _, _ = sketch_with_corner
    element_with_corner.selection.select_junction(corner_pid, is_multi=False)

    element_with_corner.add_fillet_action()

    element_with_corner.execute_command.assert_called_once()
    command_instance = element_with_corner.execute_command.call_args[0][0]
    assert isinstance(command_instance, FilletCommand)


@pytest.mark.ui
def test_add_fillet_action_on_short_lines(
    element_with_corner, sketch_with_corner
):
    """Test that fillet action is aborted if lines are too short."""
    sketch, corner_pid, line1_id, _ = sketch_with_corner
    element = element_with_corner

    line1 = sketch.registry.get_entity(line1_id)

    assert isinstance(line1, Line)
    p1 = sketch.registry.get_point(line1.p1_idx)
    p1.x = -1e-7

    element.selection.select_junction(corner_pid, is_multi=False)
    element.add_fillet_action()

    element.execute_command.assert_not_called()
