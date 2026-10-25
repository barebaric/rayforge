"""Regression tests for the in-place rename close path (issue #633).

Committing a rename with Enter removes the focused editor entry, and
GTK answers that removal with a focus change. The close routine must
ignore the focus change it causes itself: previously a second close
ran while the first was mid-flight, appending the name label to the
row twice and wedging GTK into an endless loop of internal errors.
"""

from typing import Any

import pytest
from gi.repository import Gdk, GLib, Gtk

from rayforge.core.group import Group
from rayforge.core.workpiece import WorkPiece
from rayforge.ui_gtk.doceditor.group_row import GroupRow
from rayforge.ui_gtk.doceditor.workpiece_row import WorkpieceRow

pytestmark = pytest.mark.ui


class ReentrantFocusRow(WorkpieceRow):
    """Row that fires the focus-leave handler in the middle of closing,
    as GTK does on some platforms when the focused entry is removed."""

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


def _process_events():
    ctx = GLib.main_context_default()
    while ctx.pending():
        ctx.iteration(False)


def _children(widget) -> list:
    result = []
    child = widget.get_first_child()
    while child:
        result.append(child)
        child = child.get_next_sibling()
    return result


def _label_count(row) -> int:
    return _children(row).count(row.name_label)


@pytest.fixture
def rename_env():
    renames: list = []
    holder: dict[str, Any] = {}

    def build(row_class, item):
        row = row_class(
            item, on_rename=lambda item, name: renames.append(name)
        )
        window = Gtk.Window()
        window.set_child(row)
        window.present()
        _process_events()
        holder["window"] = window
        holder["row"] = row
        return row

    yield build, renames, holder

    window = holder.get("window")
    if window is not None:
        window.destroy()
    _process_events()


@pytest.mark.ui
@pytest.mark.parametrize(
    "row_class,item_factory",
    [
        (WorkpieceRow, lambda: WorkPiece("Old Name")),
        (GroupRow, lambda: Group("Old Name")),
    ],
)
def test_enter_commit_applies_rename_once(rename_env, row_class, item_factory):
    build, renames, _ = rename_env
    row = build(row_class, item_factory())

    row.start_rename()
    _process_events()
    entry = row._rename_entry
    assert entry is not None

    entry.set_text("New Name")
    entry.emit("activate")
    _process_events()

    assert renames == ["New Name"]
    assert not row.is_renaming()
    assert _label_count(row) == 1
    assert entry.get_parent() is None
    assert not any(isinstance(c, Gtk.Entry) for c in _children(row))


@pytest.mark.ui
@pytest.mark.parametrize(
    "row_class,item_factory",
    [
        (ReentrantFocusRow, lambda: WorkPiece("Old Name")),
        (ReentrantFocusGroupRow, lambda: Group("Old Name")),
    ],
)
def test_focus_leave_during_close_is_ignored(
    rename_env, row_class, item_factory
):
    build, renames, _ = rename_env
    row = build(row_class, item_factory())

    row.start_rename()
    _process_events()
    row._rename_entry.set_text("New Name")
    row._fire_leave_during_close = True

    row._rename_entry.emit("activate")
    _process_events()

    assert renames == ["New Name"]
    assert not row.is_renaming()
    assert _label_count(row) == 1
    assert not any(isinstance(c, Gtk.Entry) for c in _children(row))


@pytest.mark.ui
@pytest.mark.parametrize(
    "row_class,item_factory",
    [
        (ReentrantFocusRow, lambda: WorkPiece("Old Name")),
        (ReentrantFocusGroupRow, lambda: Group("Old Name")),
    ],
)
def test_focus_leave_during_cancel_does_not_rename(
    rename_env, row_class, item_factory
):
    build, renames, _ = rename_env
    row = build(row_class, item_factory())

    row.start_rename()
    _process_events()
    row._rename_entry.set_text("New Name")
    row._fire_leave_during_close = True

    row._on_rename_key_pressed(None, Gdk.KEY_Escape, 0, 0)
    _process_events()

    assert renames == []
    assert row.name_label.get_text() == "Old Name"
    assert not row.is_renaming()
    assert _label_count(row) == 1


@pytest.mark.ui
def test_deferred_focus_leave_after_close_is_ignored(rename_env):
    build, renames, _ = rename_env
    row = build(WorkpieceRow, WorkPiece("Old Name"))

    row.start_rename()
    _process_events()
    entry = row._rename_entry
    controller = _focus_controller(entry)

    entry.set_text("New Name")
    entry.emit("activate")
    _process_events()
    controller.emit("leave")
    _process_events()

    assert renames == ["New Name"]
    assert not row.is_renaming()
    assert _label_count(row) == 1


def _focus_controller(entry) -> Gtk.EventControllerFocus:
    controllers = entry.observe_controllers()
    for pos in range(controllers.get_n_items()):
        controller = controllers.get_item(pos)
        if isinstance(controller, Gtk.EventControllerFocus):
            return controller
    raise AssertionError("no focus controller on rename entry")
