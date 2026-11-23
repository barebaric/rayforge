# flake8: noqa: E402
import os
import sys
import time
from typing import Any

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
gi.require_version("Gdk", "4.0")
from gi.repository import Adw, GLib

from rayforge.ui_gtk import sim3d
from rayforge.ui_gtk.mainwindow import MainWindow


def _process_events_for(duration: float) -> None:
    end = time.monotonic() + duration
    context = GLib.main_context_default()
    while time.monotonic() < end:
        while context.pending():
            context.iteration(False)
        time.sleep(0.01)


@pytest.fixture
def app_and_window(ui_context_initializer, request):
    sim3d.initialize()
    assert sim3d.initialized, "Canvas3D failed to initialize"

    window: Any = None

    class TestApp(Adw.Application):
        def do_activate(self):
            nonlocal window
            window = MainWindow(application=self)
            window.set_default_size(1280, 800)
            self.win = window

    name = request.node.name.replace("_", "-")
    app = TestApp(application_id=f"org.rayforge.rayforge.test.{name}")
    app.register(None)
    app.activate()
    _process_events_for(0.5)

    assert getattr(app, "win", None) is not None
    window = app.win
    window.present()
    _process_events_for(0.5)

    yield app, window

    if window:
        window.doc_editor.cleanup()
        window.close()
        app.quit()
    _process_events_for(0.2)


@pytest.mark.ui
def test_barcode_actions_and_asset_browser(app_and_window):
    _app, window = app_and_window
    editor = window.doc_editor
    browser = window.bottom_panel.asset_browser
    actions = window.action_manager

    assert actions.get_action("add-barcode") is not None
    assert actions.get_action("new-barcode-asset") is not None
    assert actions.get_action("activate-barcode") is not None

    workpieces_before = len(editor.doc.all_workpieces)
    actions.get_action("add-barcode").activate(None)
    _process_events_for(0.1)
    assert len(editor.doc.all_workpieces) == workpieces_before + 1
    workpiece = editor.doc.all_workpieces[-1]
    asset = workpiece.get_geometry_provider()
    assert asset is not None
    assert asset.provider_type_name == "barcode"
    assert asset.uid in browser._cards
    assert workpiece in window.surface.get_selected_workpieces()

    assets_before = len(editor.doc.get_assets_by_type("barcode"))
    actions.get_action("new-barcode-asset").activate(None)
    _process_events_for(0.1)
    assert len(editor.doc.get_assets_by_type("barcode")) == assets_before + 1
    assert len(editor.doc.all_workpieces) == workpieces_before + 1

    library_asset = list(editor.doc.get_assets_by_type("barcode").values())[-1]
    browser.asset_activated.send(browser, asset=library_asset)
    _process_events_for(0.1)
    assert len(editor.doc.all_workpieces) == workpieces_before + 2
    assert editor.doc.all_workpieces[-1] in (
        window.surface.get_selected_workpieces()
    )
