# flake8: noqa: E402
import logging
import os
import sys
import time
from pathlib import Path
import pytest
import threading

# Platform-Specific Setup
if sys.platform.startswith("linux"):
    os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
    if not os.environ.get("DISPLAY"):
        pytest.skip(
            "DISPLAY not set on Linux, skipping UI tests. Run with xvfb-run.",
            allow_module_level=True,
        )


# Gtk imports must happen AFTER the platform setup and display check.
import gi

gi.require_version("Gtk", "4.0")
gi.require_version("Adw", "1")
gi.require_version("Gdk", "4.0")
from gi.repository import Adw, GLib

from rayforge.ui_gtk.mainwindow import MainWindow
from rayforge.core.vectorization_spec import TraceSpec

logger = logging.getLogger(__name__)


# Helper functions adapted for robust testing


def process_events_for_duration(duration_sec: float):
    """Processes all pending GTK events for a given duration without blocking."""
    end_time = time.monotonic() + duration_sec
    context = GLib.main_context_default()
    while time.monotonic() < end_time:
        while context.pending():
            context.iteration(False)
        time.sleep(0.01)


def wait_for_document_to_settle(window: MainWindow, timeout: int = 45) -> bool:
    """
    Waits for the 'document_settled' signal in a thread-safe manner.
    """
    settled_event = threading.Event()

    def on_settled(sender):
        logger.info("Received 'document_settled' signal.")
        settled_event.set()

    handler_id = window.doc_editor.document_settled.connect(on_settled)

    logger.info("Waiting for document to settle...")
    start_time = time.monotonic()

    while not settled_event.is_set():
        process_events_for_duration(0.1)
        if time.monotonic() - start_time > timeout:
            logger.error("Timeout waiting for document_settled signal.")
            window.doc_editor.document_settled.disconnect(handler_id)
            return False

    window.doc_editor.document_settled.disconnect(handler_id)
    return window.doc_editor.doc.has_result()


def activate_simulation_mode(window) -> bool:
    """Activate simulation mode via its Gio.Action."""
    action = window.action_manager.get_action("simulate_mode")
    if not action:
        return False

    action.change_state(GLib.Variant.new_boolean(True))
    process_events_for_duration(1.5)
    return window.surface.is_simulation_mode()


@pytest.fixture
def assets_path() -> Path:
    return Path(__file__).parent.parent / "tests"


@pytest.fixture
def test_file_path(assets_path: Path) -> Path:
    path = assets_path / "image" / "png" / "color.png"
    assert path.exists()
    return path


@pytest.fixture
def app_and_window(ui_context_initializer):
    """Sets up the Adw.Application and MainWindow without blocking."""
    from rayforge.ui_gtk import canvas3d

    canvas3d.initialize()
    assert canvas3d.initialized, "Canvas3D failed to initialize"

    win = None

    class TestApp(Adw.Application):
        def do_activate(self):
            nonlocal win
            win = MainWindow(application=self)
            win.set_default_size(1280, 800)
            self.win = win

    app = TestApp(application_id="org.rayforge.rayforge.test")
    app.register(None)
    app.activate()
    process_events_for_duration(0.5)

    assert hasattr(app, "win") and app.win is not None
    win = app.win
    win.present()
    process_events_for_duration(0.5)

    yield app, win

    # Teardown
    if win:
        win.doc_editor.cleanup()
        win.close()
        app.quit()
    process_events_for_duration(0.2)


@pytest.mark.ui
def test_main_window_simulation_mode_activation(
    app_and_window, test_file_path
):
    """
    Tests that activating simulation mode correctly changes the application's
    internal state and UI structure. This is a more robust alternative to
    screenshot testing.
    """
    _app, win = app_and_window

    # 1. Load a file and wait for the document pipeline to be ready.
    win.doc_editor.file.load_file_from_path(
        filename=test_file_path,
        mime_type="image/png",
        vectorization_spec=TraceSpec(),
    )
    assert wait_for_document_to_settle(win), (
        "Document did not settle or has no result"
    )

    # 2. Assert the initial state (before activation).
    assert not win.surface.is_simulation_mode(), (
        "Should not be in sim mode initially"
    )
    assert win.simulator_cmd.preview_controls is None, (
        "Sim controls should not exist initially"
    )

    # 3. Activate simulation mode.
    assert activate_simulation_mode(win), "Failed to activate sim mode"

    # 4. Assert the final state (after activation).
    assert win.surface.is_simulation_mode(), "Surface failed to enter sim mode"
    assert win.simulator_cmd.preview_controls is not None, (
        "Sim preview controls were not created"
    )
    assert win.simulator_cmd.preview_controls.is_visible(), (
        "Sim preview controls are not visible"
    )
    logger.info(
        "Successfully verified simulation mode activation and UI state."
    )
