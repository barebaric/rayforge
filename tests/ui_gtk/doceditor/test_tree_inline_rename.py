# flake8: noqa: E402
"""UI tests for the inline rename entry of the tree rows."""

import time

import gi

gi.require_version("Gtk", "4.0")
gi.require_version("Gdk", "4.0")

import pytest
from gi.repository import Gdk, GLib, Gtk

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


def _pump(seconds: float = 0.2):
    context = GLib.MainContext.default()
    end = time.monotonic() + seconds
    while time.monotonic() < end:
        context.iteration(False)
        time.sleep(0.005)


def _find_entry(widget):
    if isinstance(widget, Gtk.Entry):
        return widget
    child = widget.get_first_child()
    while child:
        found = _find_entry(child)
        if found is not None:
            return found
        child = child.get_next_sibling()
    return None


@pytest.fixture
def rename_setup():
    window = Gtk.Window()
    workpiece = WorkPiece("Part")
    callbacks = _Callbacks()
    row = WorkPieceNode(workpiece, 0, set(), set(), callbacks)
    button = Gtk.Button(label="Elsewhere")

    listbox = Gtk.ListBox()
    listbox.set_selection_mode(Gtk.SelectionMode.NONE)
    listbox.append(row)
    box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL)
    box.append(listbox)
    box.append(button)
    window.set_child(box)
    window.present()
    _pump()
    yield row, button, callbacks
    window.destroy()


def _start_rename(row):
    row.start_rename()
    _pump()
    entry = _find_entry(row)
    assert entry is not None
    return entry


@pytest.mark.ui
def test_tabbing_out_commits_and_removes_the_entry(rename_setup):
    """Focus leaving the inline entry finishes the rename."""
    row, button, callbacks = rename_setup
    entry = _start_rename(row)
    assert not isinstance(row.get_root().get_focus(), Gtk.ListBoxRow)

    entry.set_text("Renamed")
    button.grab_focus()
    _pump()

    assert _find_entry(row) is None
    assert row._name_label.get_visible()
    assert callbacks.renamed == ["Renamed"]


@pytest.mark.ui
def test_enter_commits_and_removes_the_entry(rename_setup):
    row, _button, callbacks = rename_setup
    entry = _start_rename(row)

    entry.set_text("Renamed")
    entry.emit("activate")
    _pump()

    assert _find_entry(row) is None
    assert row._name_label.get_visible()
    assert callbacks.renamed == ["Renamed"]


@pytest.mark.ui
def test_escape_cancels_the_rename(rename_setup):
    row, _button, callbacks = rename_setup
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
