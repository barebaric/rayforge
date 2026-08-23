# flake8: noqa: E402
import os
import sys
from unittest.mock import MagicMock

import pytest

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

from sketcher.core import Sketch
from sketcher.core.commands import MoveEntitiesCommand
from sketcher.ui_gtk.sketchelement import SketchElement

from rayforge.core.undo import HistoryManager


@pytest.fixture
def element_with_line():
    s = Sketch()
    p1 = s.add_point(0, 0)
    p2 = s.add_point(10, 0)
    line_id = s.add_line(p1, p2)

    element = SketchElement(sketch=s)
    element.editor = MagicMock()
    history = HistoryManager()
    element.execute_command = history.execute
    return element, s, p1, p2, line_id, history


def _get_executed_command(history: HistoryManager):
    return history.undo_stack[-1] if history.undo_stack else None


@pytest.mark.ui
def test_nudge_moves_selected_entity(element_with_line):
    element, sketch, p1, p2, _, _ = element_with_line
    registry = sketch.registry
    entity_id = registry.entities[0].id
    element.selection.select_entity(
        registry.get_entity(entity_id), is_multi=False
    )

    assert element.nudge_selection(1.0, -2.0) is True
    assert (registry.get_point(p1).x, registry.get_point(p1).y) == (
        1.0,
        -2.0,
    )
    assert (registry.get_point(p2).x, registry.get_point(p2).y) == (
        11.0,
        -2.0,
    )


@pytest.mark.ui
def test_nudge_creates_undoable_command(element_with_line):
    element, sketch, p1, p2, _, history = element_with_line
    registry = sketch.registry
    entity = registry.entities[0]
    element.selection.select_entity(entity, is_multi=False)

    element.nudge_selection(5.0, 0.0)

    cmd = _get_executed_command(history)
    assert isinstance(cmd, MoveEntitiesCommand)

    history.undo()
    assert (registry.get_point(p1).x, registry.get_point(p1).y) == (
        0.0,
        0.0,
    )
    assert (registry.get_point(p2).x, registry.get_point(p2).y) == (
        10.0,
        0.0,
    )


@pytest.mark.ui
def test_nudge_without_selection_returns_false(element_with_line):
    element, _, _, _, _, _ = element_with_line
    assert element.nudge_selection(1.0, 1.0) is False


@pytest.mark.ui
def test_nudge_skips_fixed_points(element_with_line):
    element, sketch, p1, p2, _, _ = element_with_line
    registry = sketch.registry
    registry.get_point(p1).fixed = True
    element.selection.point_ids.extend([p1, p2])

    assert element.nudge_selection(1.0, 1.0) is True
    assert (registry.get_point(p1).x, registry.get_point(p1).y) == (
        0.0,
        0.0,
    )
    assert (registry.get_point(p2).x, registry.get_point(p2).y) == (
        11.0,
        1.0,
    )
