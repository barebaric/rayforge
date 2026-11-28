# flake8: noqa: E402
"""UI tests for the inline rename entry of the tree rows."""

import time
import uuid

import gi

gi.require_version("Gtk", "4.0")
gi.require_version("Gdk", "4.0")
gi.require_version("Adw", "1")

import pytest
from gi.repository import Adw, Gdk, GLib, Gtk

from rayforge.core.workpiece import WorkPiece
from rayforge.doceditor.tree.workpiece_node import WorkPieceNode


class _Callbacks:
    def __init__(self):
        self.renamed = []

    def on_row_activated(self, spec):
        pass

    def on_eye_toggled(self, spec):
        pass

    def on_twisty_toggled(self, spec):
        pass

    def on_row_renamed(self, spec, name):
        self.renamed.append(name)

    def on_row_context_menu(self, spec, row, x, y):
        pass


def _pump(seconds: float = 0.2):
    context = GLib.MainContext.default()
    end = time.monotonic() + seconds
    while time.monotonic() < end:
        context.iteration(False)
        time.sleep(0.005)


def _walk(widget):
    yield widget
    child = widget.get_first_child()
    while child:
        yield from _walk(child)
        child = child.get_next_sibling()


def _find_entry(widget):
    return next((w for w in _walk(widget) if isinstance(w, Gtk.Entry)), None)


def _make_row():
    workpiece = WorkPiece("Part")
    callbacks = _Callbacks()
    row = WorkPieceNode(workpiece, 0, set(), set(), callbacks)
    return row, callbacks


def _start_rename(row):
    row.start_rename()
    entry = _find_entry(row)
    assert entry is not None
    return entry


@pytest.mark.ui
def test_enter_commits_and_removes_the_entry():
    row, callbacks = _make_row()
    entry = _start_rename(row)

    entry.set_text("Renamed")
    entry.emit("activate")
    _pump()

    assert _find_entry(row) is None
    assert row._name_label.get_visible()
    assert callbacks.renamed == ["Renamed"]


@pytest.mark.ui
def test_escape_cancels_the_rename():
    row, callbacks = _make_row()
    entry = _start_rename(row)

    entry.set_text("Renamed")
    controller = next(
        c
        for c in entry.observe_controllers()
        if isinstance(c, Gtk.EventControllerKey)
    )
    handled = controller.emit("key-pressed", Gdk.KEY_Escape, 0, 0)
    _pump()

    assert handled
    assert _find_entry(row) is None
    assert row._name_label.get_visible()
    assert callbacks.renamed == []


def _wait_for(predicate, timeout: float = 3.0) -> bool:
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        if predicate():
            return True
        _pump(0.05)
    return predicate()


@pytest.mark.ui
def test_tabbing_out_commits_and_removes_the_entry():
    """Focus leaving the inline entry finishes the rename."""
    app = Adw.Application(
        application_id=f"org.rayforge.test.rename.{uuid.uuid4().hex}"
    )
    app.register(None)
    app.activate()
    window = Gtk.ApplicationWindow(application=app)
    row, callbacks = _make_row()
    button = Gtk.Button(label="Elsewhere")

    listbox = Gtk.ListBox()
    listbox.set_selection_mode(Gtk.SelectionMode.NONE)
    listbox.append(row)
    box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL)
    box.append(listbox)
    box.append(button)
    window.set_child(box)
    window.present()

    try:
        assert _wait_for(window.get_mapped)

        entry = _start_rename(row)
        assert _wait_for(
            lambda: not isinstance(window.get_focus(), Gtk.ListBoxRow)
        )

        entry.set_text("Renamed")
        button.grab_focus()
        assert _wait_for(lambda: _find_entry(row) is None)

        assert row._name_label.get_visible()
        assert callbacks.renamed == ["Renamed"]
    finally:
        window.destroy()
        app.quit()
