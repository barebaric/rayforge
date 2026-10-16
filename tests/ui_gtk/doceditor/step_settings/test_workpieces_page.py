# flake8: noqa: E402
"""Tests for the read-only Workpieces settings page."""

from collections import deque

import gi
import pytest

gi.require_version("Gtk", "4.0")
gi.require_version("Adw", "1")

from rayforge.core.step import Step
from rayforge.core.workpiece import WorkPiece
from rayforge.ui_gtk.doceditor.step_settings.dialog import StepSettingsDialog
from rayforge.ui_gtk.doceditor.step_settings.pages.workpieces import (
    WorkpiecesPage,
)


@pytest.fixture
def attached_step(editor):
    """A step on the editor's active layer, with two workpieces."""
    step = Step(typelabel="Fake", name="fake")
    layer = editor.doc.active_layer
    assert layer.workflow is not None
    layer.workflow.add_step(step)
    wp_a = WorkPiece(name="wp-a")
    wp_b = WorkPiece(name="wp-b")
    wp_a.uid = "wp-a"
    wp_b.uid = "wp-b"
    layer.add_child(wp_a)
    layer.add_child(wp_b)
    return step, layer


def _row_titles(page):
    """Titles of all rows in the page's group, descending into the
    wrapper boxes Adw.PreferencesGroup adds around its rows."""
    titles = []
    queue = deque([page._group.get_first_child()])
    while queue:
        child = queue.popleft()
        while child is not None:
            if hasattr(child, "get_title"):
                titles.append(child.get_title())
            else:
                queue.append(child.get_first_child())
            child = child.get_next_sibling()
    return titles


@pytest.mark.ui
def test_page_lists_all_workpieces_without_filter(editor, attached_step):
    step, _layer = attached_step
    page = WorkpiecesPage(editor, step)

    assert _row_titles(page) == [
        "All workpieces on this layer",
        "wp-a",
        "wp-b",
    ]
    page._cleanup()


@pytest.mark.ui
def test_page_marks_excluded_workpieces(editor, attached_step):
    step, _layer = attached_step
    step.workpiece_uids = ["wp-b"]
    page = WorkpiecesPage(editor, step)

    assert _row_titles(page) == [
        "Processes 1 of 2 workpieces",
        "wp-a",
        "wp-b",
    ]
    page._cleanup()


@pytest.mark.ui
def test_page_refreshes_when_filter_changes(editor, attached_step):
    step, _layer = attached_step
    page = WorkpiecesPage(editor, step)

    step.set_workpiece_uids(["wp-a"])
    assert _row_titles(page)[0] == "Processes 1 of 2 workpieces"

    step.set_workpiece_uids([])
    assert _row_titles(page)[0] == "All workpieces on this layer"
    page._cleanup()


@pytest.mark.ui
def test_dialog_adds_page_for_layer_step(editor, attached_step):
    step, _layer = attached_step
    dialog = StepSettingsDialog(editor, step)

    assert dialog.workpieces_view is not None
    assert dialog.btn_workpieces is not None
    dialog.close()


@pytest.mark.ui
def test_dialog_omits_page_for_detached_step(editor, step):
    dialog = StepSettingsDialog(editor, step)

    assert dialog.workpieces_view is None
    assert dialog.btn_workpieces is None
    dialog.close()
