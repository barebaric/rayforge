"""Regression tests for the in-place rename close path (issue #633).

Committing a rename with Enter removes the focused editor entry, and
GTK answers that removal with a focus change. The close routine must
ignore the focus change it causes itself: previously a second close
ran while the first was mid-flight, appending the name label to the
row twice and wedging GTK into an endless loop of internal errors.

The tests run without real windows or focus: repeatedly creating and
destroying focused entries destabilizes GTK's IME handling on Windows,
and the simulated focus-leave is what exercises the re-entrancy guard.
"""

import pytest
from gi.repository import Gdk, Gtk

from rayforge.core.group import Group
from rayforge.core.workpiece import WorkPiece
from rayforge.ui_gtk.doceditor.group_row import GroupRow
from rayforge.ui_gtk.doceditor.workpiece_row import WorkpieceRow

pytestmark = pytest.mark.ui


class ReentrantFocusRow(WorkpieceRow):
    """Row that fires the focus-leave handler in the middle of closing,
    as GTK does when the focused entry is removed."""

    def __init__(self, workpiece, on_rename=None):
        super().__init__(workpiece, on_rename=on_rename)
        self._fire_leave_during_close = False

    def _replace_name_widget(self):
        if self._fire_leave_during_close:
            self._fire_leave_during_close = False
            self._on_rename_focus_out()
        super()._replace_name_widget()


class ReentrantFocusGroupRow(GroupRow):
    """GroupRow variant of ReentrantFocusRow."""

    def __init__(self, group, on_rename=None):
        super().__init__(group, on_rename=on_rename)
        self._fire_leave_during_close = False

    def _replace_name_widget(self):
        if self._fire_leave_during_close:
            self._fire_leave_during_close = False
            self._on_rename_focus_out()
        super()._replace_name_widget()


def _children(widget) -> list:
    result = []
    child = widget.get_first_child()
    while child:
        result.append(child)
        child = child.get_next_sibling()
    return result


def _label_count(row) -> int:
    return _children(row).count(row.name_label)


def _focus_controller(entry) -> Gtk.EventControllerFocus:
    controllers = entry.observe_controllers()
    for pos in range(controllers.get_n_items()):
        controller = controllers.get_item(pos)
        if isinstance(controller, Gtk.EventControllerFocus):
            return controller
    raise AssertionError("no focus controller on rename entry")


def _make_row(row_class, item):
    renames: list = []
    return row_class(
        item, on_rename=lambda i, name: renames.append(name)
    ), renames


def test_enter_commit_applies_rename_once():
    row, renames = _make_row(WorkpieceRow, WorkPiece("Old Name"))

    row.start_rename()
    entry = row._rename_entry
    assert entry is not None

    entry.set_text("New Name")
    entry.emit("activate")

    assert renames == ["New Name"]
    assert not row.is_renaming()
    assert _label_count(row) == 1
    assert entry.get_parent() is None
    assert not any(isinstance(c, Gtk.Entry) for c in _children(row))


def test_enter_commit_applies_rename_once_for_group():
    row, renames = _make_row(GroupRow, Group("Old Name"))

    row.start_rename()
    entry = row._rename_entry
    assert entry is not None

    entry.set_text("New Name")
    entry.emit("activate")

    assert renames == ["New Name"]
    assert not row.is_renaming()
    assert _label_count(row) == 1
    assert entry.get_parent() is None


def test_focus_leave_during_close_is_ignored():
    row, renames = _make_row(ReentrantFocusRow, WorkPiece("Old Name"))

    row.start_rename()
    entry = row._rename_entry
    assert entry is not None
    entry.set_text("New Name")
    row._fire_leave_during_close = True
    entry.emit("activate")

    assert renames == ["New Name"]
    assert not row.is_renaming()
    assert _label_count(row) == 1
    assert not any(isinstance(c, Gtk.Entry) for c in _children(row))


def test_focus_leave_during_close_is_ignored_for_group():
    row, renames = _make_row(ReentrantFocusGroupRow, Group("Old Name"))

    row.start_rename()
    entry = row._rename_entry
    assert entry is not None
    entry.set_text("New Name")
    row._fire_leave_during_close = True
    entry.emit("activate")

    assert renames == ["New Name"]
    assert not row.is_renaming()
    assert _label_count(row) == 1
    assert not any(isinstance(c, Gtk.Entry) for c in _children(row))


def test_focus_leave_during_cancel_does_not_rename():
    row, renames = _make_row(ReentrantFocusRow, WorkPiece("Old Name"))

    row.start_rename()
    entry = row._rename_entry
    assert entry is not None
    entry.set_text("New Name")
    row._fire_leave_during_close = True
    row._on_rename_key_pressed(None, Gdk.KEY_Escape, 0, 0)

    assert renames == []
    assert row.name_label.get_text() == "Old Name"
    assert not row.is_renaming()
    assert _label_count(row) == 1


def test_focus_leave_during_cancel_does_not_rename_for_group():
    row, renames = _make_row(ReentrantFocusGroupRow, Group("Old Name"))

    row.start_rename()
    entry = row._rename_entry
    assert entry is not None
    entry.set_text("New Name")
    row._fire_leave_during_close = True
    row._on_rename_key_pressed(None, Gdk.KEY_Escape, 0, 0)

    assert renames == []
    assert row.name_label.get_text() == "Old Name"
    assert not row.is_renaming()
    assert _label_count(row) == 1


def test_deferred_focus_leave_after_close_is_ignored():
    row, renames = _make_row(WorkpieceRow, WorkPiece("Old Name"))

    row.start_rename()
    entry = row._rename_entry
    assert entry is not None
    controller = _focus_controller(entry)

    entry.set_text("New Name")
    entry.emit("activate")
    controller.emit("leave")

    assert renames == ["New Name"]
    assert not row.is_renaming()
    assert _label_count(row) == 1
