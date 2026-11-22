"""Regression tests for clicks during in-place rename (issue #582).

The layer column's capture-phase click gesture must ignore clicks that
land on a row while its name editor is open. Previously, double-clicking
a word in the rename entry of a sketch workpiece opened the sketch
editor modal page, hiding the focused entry mid-event and hanging GTK's
focus handling.
"""

from typing import Any
from unittest.mock import MagicMock

import pytest
from blinker import Signal
from gi.repository import Gtk

from rayforge.core.workpiece import WorkPiece
from rayforge.ui_gtk.doceditor.layer_column import LayerColumn
from rayforge.ui_gtk.doceditor.workpiece_row import WorkpieceRow

pytestmark = pytest.mark.ui


class _SketchAsset:
    edit_item_action = "edit-sketch-item"


def _column_with_row(sketch_workpiece: WorkPiece):
    """Builds a bare LayerColumn with one row, bypassing Gtk __init__."""
    column: Any = LayerColumn.__new__(LayerColumn)
    column.layer = MagicMock()
    column.doc = MagicMock()
    column.doc.active_layer = column.layer
    column._row_items = {}
    column._ordered_items = []
    column._selected_uids = set()
    column._selection_anchor = None
    column._pressed_clicked_item = None
    column.edit_item_requested = Signal()
    column.select_items_requested = Signal()

    row = Gtk.ListBoxRow()
    row_child = WorkpieceRow(sketch_workpiece, on_rename=lambda item, n: None)
    row.set_child(row_child)
    column._row_items[row] = sketch_workpiece
    column._ordered_items.append(sketch_workpiece)
    return column, row_child


def _pick_click_target(column, row_child):
    """Makes the column's pick() resolve to a widget inside the row."""
    entry = Gtk.Entry()
    row_child.append(entry)
    column.pick = lambda x, y, flags: entry


@pytest.fixture
def sketch_workpiece():
    wp = WorkPiece("New Sketch")
    wp.geometry_provider_uid = "sketch-uid"
    return wp


@pytest.mark.ui
def test_double_click_in_rename_entry_is_ignored(sketch_workpiece):
    column, row_child = _column_with_row(sketch_workpiece)
    _pick_click_target(column, row_child)
    row_child.start_rename()
    assert row_child.is_renaming()

    emissions = []

    def on_edit_requested(sender, **kw):
        emissions.append(kw)

    column.edit_item_requested.connect(on_edit_requested, weak=False)

    column._on_column_clicked(MagicMock(), 2, 5.0, 5.0)

    assert emissions == []


@pytest.mark.ui
def test_single_click_in_rename_entry_does_not_change_selection(
    sketch_workpiece,
):
    column, row_child = _column_with_row(sketch_workpiece)
    _pick_click_target(column, row_child)
    row_child.start_rename()

    emissions = []

    def on_select_items(sender, **kw):
        emissions.append(kw)

    column.select_items_requested.connect(on_select_items, weak=False)

    gesture = MagicMock()
    gesture.get_current_event.return_value = None
    column._on_column_clicked(gesture, 1, 5.0, 5.0)

    assert emissions == []


@pytest.mark.ui
def test_double_click_outside_rename_opens_editor(sketch_workpiece):
    column, row_child = _column_with_row(sketch_workpiece)
    _pick_click_target(column, row_child)
    column.doc.get_asset_by_uid = lambda uid: _SketchAsset()

    emissions = []

    def on_edit_requested(sender, **kw):
        emissions.append(kw["item"])

    column.edit_item_requested.connect(on_edit_requested, weak=False)

    column._on_column_clicked(MagicMock(), 2, 5.0, 5.0)

    assert emissions == [sketch_workpiece]
