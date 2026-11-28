# flake8: noqa: E402
"""Regression test: stepping the X Position spin row with several
items selected must not re-enter the transform command in a loop.

The properties panel refreshes when each item's matrix changes. A
multi-item move executes one command per item, so a synchronous
refresh would read the group bounding box while only part of the move
has been applied, feed that intermediate value back into the spin row,
and re-trigger the move endlessly. The panel therefore refreshes on
idle, and programmatic spin row writes mark their value as echoed so
the deferred GTK change notification is not mistaken for a user edit.
"""

import faulthandler
import os
import sys
import time

import pytest

from rayforge.core.workpiece import WorkPiece

if sys.platform.startswith("linux"):
    if not os.environ.get("DISPLAY"):
        pytest.skip(
            "DISPLAY not set on Linux, skipping UI tests. Run with xvfb-run.",
            allow_module_level=True,
        )

import gi

gi.require_version("Gtk", "4.0")
gi.require_version("Adw", "1")
gi.require_version("Gdk", "4.0")
from gi.repository import Adw, GLib

from rayforge.ui_gtk.doceditor.property_providers import (
    register_builtin_providers,
)
from rayforge.ui_gtk.doceditor.property_providers.transform import (
    TransformPropertyProvider,
)
from rayforge.ui_gtk.mainwindow import MainWindow

register_builtin_providers()


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

    test_name = request.node.name.replace("_", "-")
    app_id = f"org.rayforge.rayforge.test.{test_name}"
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


@pytest.mark.ui
def test_multi_select_x_spin_step_settles(app_and_window, ui_task_mgr):
    faulthandler.dump_traceback_later(60, exit=True)
    _app, win = app_and_window
    editor = win.doc_editor

    wp1 = WorkPiece("a")
    wp1.pos = (40.0, 50.0)
    wp1.set_size(30.0, 20.0)
    wp2 = WorkPiece("b")
    wp2.pos = (80.0, 55.0)
    wp2.set_size(15.0, 15.0)
    editor.doc.active_layer.add_child(wp1)
    editor.doc.active_layer.add_child(wp2)
    process_events_for_duration(0.3)

    win.surface.select_items([wp1, wp2])
    process_events_for_duration(0.3)

    provider = next(
        p
        for p, _w in win.item_props_widget._provider_widget_map
        if isinstance(p, TransformPropertyProvider)
    )
    adj = provider.x_row.get_adjustment()

    def pump_until_idle(max_iterations=20000):
        ctx = GLib.main_context_default()
        for _ in range(max_iterations):
            if not ctx.pending():
                return
            ctx.iteration(False)
        raise AssertionError("Main context never went idle")

    start_x = adj.get_value()
    for _ in range(3):
        adj.set_value(adj.get_value() + adj.get_step_increment())
        pump_until_idle()

    bbox_min_x = min(
        wp1.get_world_transform().transform_point((0, 0))[0],
        wp2.get_world_transform().transform_point((0, 0))[0],
    )
    assert adj.get_value() == pytest.approx(start_x + 3)
    assert bbox_min_x == pytest.approx(start_x + 3)

    faulthandler.cancel_dump_traceback_later()
