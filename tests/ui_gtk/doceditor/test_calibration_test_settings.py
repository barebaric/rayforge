# flake8: noqa: E402
"""
The calibration tests in the real main window: Tools → Create … adds
the test on its own layer, selects it and shows its Test Settings in
the item properties panel without opening a window; a change there
rebuilds the test and keeps it selected.
"""

import os
import sys
import time

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
from gi.repository import Adw, GLib

from rayforge.core.workpiece import WorkPiece
from rayforge.ui_gtk.mainwindow import MainWindow

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


def _provider(win):
    for provider, _widgets, expander in win.item_props_widget._separate_groups:
        if type(provider).__name__ == "CalibrationPropertyProvider":
            return provider, expander
    raise AssertionError("Test Settings provider not registered")


def _selected(win) -> list[WorkPiece]:
    return [
        item
        for item in win.surface.get_selected_items()
        if isinstance(item, WorkPiece)
    ]


def _windows(app, win):
    return [w for w in app.get_windows() if w is not win and w.get_visible()]


@pytest.mark.ui
@pytest.mark.parametrize("action", ACTIONS)
def test_create_shows_settings_in_the_panel(app_and_window, action):
    app, win = app_and_window
    layers_before = len(win.doc_editor.doc.layers)

    win.lookup_action(action).activate(None)
    process_events_for_duration(1.5)

    doc = win.doc_editor.doc
    assert len(doc.layers) == layers_before + 1
    layer = doc.active_layer
    assert set(_selected(win)) == set(layer.all_workpieces)
    assert win.item_revealer.get_reveal_child()
    provider, expander = _provider(win)
    assert expander.get_visible()
    assert provider.varset_widget.keys()
    assert _windows(app, win) == []
    scrolled = provider._scrolled_window()
    offset = provider._card_offset(scrolled)
    adjustment = scrolled.get_vadjustment()
    top = adjustment.get_upper() - adjustment.get_page_size()
    assert adjustment.get_value() == pytest.approx(min(offset, top), abs=1)

    assert provider._scroll_handlers
    provider.stop_following()
    assert provider._scroll_handlers == []


@pytest.mark.ui
def test_change_rebuilds_and_keeps_the_selection(app_and_window):
    _app, win = app_and_window
    win.lookup_action("interval_test").activate(None)
    process_events_for_duration(1.5)
    provider, expander = _provider(win)
    old = set(_selected(win))

    row = provider.varset_widget.row_for("count")
    row.get_spin_button().set_value(3)
    process_events_for_duration(1.5)

    layer = win.doc_editor.doc.active_layer
    new = set(_selected(win))
    assert len(layer.all_workpieces) == 4
    assert new == set(layer.all_workpieces)
    assert new.isdisjoint(old)
    assert expander.get_visible()
    assert win.item_revealer.get_reveal_child()

    win.doc_editor.history_manager.undo()
    process_events_for_duration(0.5)
    assert len(layer.all_workpieces) == 6
