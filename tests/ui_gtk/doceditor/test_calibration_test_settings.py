# flake8: noqa: E402
"""
The calibration tests in the real main window: Tools → Create … adds
the test on its own layer and selects it, and the item properties panel
shows a pointer to its settings. Opening that pointer presents the
settings dialog of the test's layer, where a change rebuilds the test
and keeps it selected.
"""

import os
import sys
import time
from typing import Any

import pytest

if sys.platform.startswith("linux"):
    if not os.environ.get("DISPLAY") and not os.environ.get("WAYLAND_DISPLAY"):
        pytest.skip(
            "No display on Linux, skipping UI tests. Run with xvfb-run.",
            allow_module_level=True,
        )

import gi

gi.require_version("Gtk", "4.0")
gi.require_version("Adw", "1")
gi.require_version("Gdk", "4.0")
from gi.repository import Adw, GLib, Gtk

from rayforge.core.workpiece import WorkPiece
from rayforge.ui_gtk.doceditor.layer_settings_dialog import (
    LayerSettingsDialog,
)
from rayforge.ui_gtk.mainwindow import MainWindow
from rayforge.ui_gtk.shared.pref_rows import SpinRow
from rayforge.ui_gtk.varset.varsetwidget import VarSetWidget

ACTIONS = ["material_test", "interval_test", "focus_test"]


def process_events_for_duration(duration_sec: float):
    end_time = time.monotonic() + duration_sec
    context = GLib.main_context_default()
    while time.monotonic() < end_time:
        while context.pending():
            context.iteration(False)
        time.sleep(0.01)


@pytest.fixture
def app_and_window(ui_context_initializer, request):
    from rayforge.ui_gtk import sim3d

    sim3d.initialize()

    win = None

    class TestApp(Adw.Application):
        def do_activate(self):
            nonlocal win
            win = MainWindow(application=self)
            win.set_default_size(1280, 800)
            self.win = win

    test_name = request.node.name.replace("_", "-").replace("[", "-")
    app_id = f"org.rayforge.rayforge.test.{test_name.rstrip(']')}"
    app = TestApp(application_id=app_id)
    app.register(None)
    app.activate()
    process_events_for_duration(0.5)

    win = app.win
    win.present()
    process_events_for_duration(0.5)

    yield app, win

    if win:
        win.doc_editor.cleanup()
        win.close()
        app.quit()
    process_events_for_duration(0.2)


def _provider(win) -> Any:
    for provider, _widgets in win.item_props_widget._provider_widget_map:
        if type(provider).__name__ == "CalibrationPropertyProvider":
            return provider
    raise AssertionError("Test Settings provider not registered")


def _selected(win) -> list[WorkPiece]:
    return [
        item
        for item in win.surface.get_selected_items()
        if isinstance(item, WorkPiece)
    ]


def _dialog(win) -> LayerSettingsDialog | None:
    """The layer settings dialog of the main window, once it is open."""
    found = [
        w
        for w in Gtk.Window.list_toplevels()
        if isinstance(w, LayerSettingsDialog)
        and w.get_visible()
        and w.get_transient_for() is win
    ]
    assert len(found) <= 1
    return found[0] if found else None


def _varset_in(widget):
    """The VarSetWidget in the widget tree, for the Test Settings rows."""
    if isinstance(widget, VarSetWidget):
        return widget
    child = widget.get_first_child()
    while child is not None:
        found = _varset_in(child)
        if found is not None:
            return found
        child = child.get_next_sibling()
    return None


def _create_test(win, action):
    win.lookup_action(action).activate(None)
    process_events_for_duration(1.5)


@pytest.mark.ui
@pytest.mark.parametrize("action", ACTIONS)
def test_create_points_to_the_layer_settings(app_and_window, action):
    _app, win = app_and_window
    layers_before = len(win.doc_editor.doc.layers)

    _create_test(win, action)

    doc = win.doc_editor.doc
    assert len(doc.layers) == layers_before + 1
    layer = doc.active_layer
    assert set(_selected(win)) == set(layer.all_workpieces)
    provider = _provider(win)
    assert provider.settings_row.get_visible()
    assert provider.settings_row.get_title() == "Open Layer Settings"
    assert win.item_revealer.get_reveal_child()
    assert _dialog(win) is None


@pytest.mark.ui
@pytest.mark.parametrize("action", ACTIONS)
def test_card_button_opens_the_layer_settings(app_and_window, action):
    _app, win = app_and_window
    _create_test(win, action)
    provider = _provider(win)

    provider.settings_row.emit("activated")
    process_events_for_duration(1.0)

    dialog = _dialog(win)
    assert dialog is not None
    varset = _varset_in(dialog)
    assert varset is not None
    assert varset.get_title() == "Test Settings"
    assert varset.keys()


@pytest.mark.ui
def test_change_in_the_dialog_rebuilds_and_keeps_the_selection(
    app_and_window,
):
    _app, win = app_and_window
    _create_test(win, "interval_test")
    provider = _provider(win)
    provider.settings_row.emit("activated")
    process_events_for_duration(1.0)
    dialog = _dialog(win)
    assert dialog is not None
    varset = _varset_in(dialog)
    assert varset is not None
    old = set(_selected(win))

    row = varset.row_for("count")
    assert isinstance(row, SpinRow)
    row.get_spin_button().set_value(3)
    process_events_for_duration(1.5)

    layer = win.doc_editor.doc.active_layer
    new = set(_selected(win))
    assert len(layer.all_workpieces) == 4
    assert new == set(layer.all_workpieces)
    assert new.isdisjoint(old)

    win.doc_editor.history_manager.undo()
    process_events_for_duration(0.5)
    assert len(layer.all_workpieces) == 6


@pytest.mark.ui
@pytest.mark.parametrize("deselect", ["nothing", "other"])
def test_scrolling_stops_when_the_test_is_deselected(app_and_window, deselect):
    _app, win = app_and_window
    _create_test(win, "interval_test")
    provider = _provider(win)
    assert provider._scroll_handlers

    if deselect == "nothing":
        win.surface.unselect_all()
    else:
        other = WorkPiece("Other")
        other.set_size(5.0, 5.0)
        win.doc_editor.doc.layers[0].add_child(other)
        process_events_for_duration(0.3)
        win.surface.select_items([other])
    process_events_for_duration(1.0)

    assert provider._scroll_handlers == []
