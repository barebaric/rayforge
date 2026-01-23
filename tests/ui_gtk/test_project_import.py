# flake8: noqa: E402
import logging
import os
import sys
from pathlib import Path
import pytest

# Platform-Specific Setup
if sys.platform.startswith("linux"):
    os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
    if not os.environ.get("DISPLAY"):
        pytest.skip(
            "DISPLAY not set on Linux, skipping UI tests. Run with xvfb-run.",
            allow_module_level=True,
        )


# Gtk imports must happen AFTER platform setup and display check.
import gi

gi.require_version("Gtk", "4.0")
gi.require_version("Adw", "1")
gi.require_version("Gdk", "4.0")
from gi.repository import Adw, GLib

from rayforge.ui_gtk.mainwindow import MainWindow
from rayforge.ui_gtk.canvas2d.elements.workpiece import WorkPieceElement
from rayforge.core.workpiece import WorkPiece

logger = logging.getLogger(__name__)


def process_events_for_duration(duration_sec: float):
    """Processes all pending GTK events for a given duration without blocking."""
    import time

    end_time = time.monotonic() + duration_sec
    context = GLib.main_context_default()
    while time.monotonic() < end_time:
        while context.pending():
            context.iteration(False)
        time.sleep(0.01)


@pytest.fixture
def assets_path() -> Path:
    return Path(__file__).parent.parent / "doceditor" / "assets"


@pytest.fixture
def workpieces_project_path(assets_path: Path) -> Path:
    path = assets_path / "workpieces_project.ryp"
    assert path.exists()
    return path


@pytest.fixture
def app_and_window(ui_context_initializer, request):
    """Sets up Adw.Application and MainWindow without blocking."""
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

    test_name = request.node.name.replace("_", "-")
    app_id = f"org.rayforge.rayforge.test.{test_name}"
    app = TestApp(application_id=app_id)
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
def test_project_import_detects_broken_visibility(
    app_and_window, workpieces_project_path
):
    """
    Tests that loading a project file (workpieces_project.ryp) loads
    document but workpieces are not visible on the surface.

    This test detects the bug where importing shows a notification that
    the file was loaded, but the content does not become visible.
    """
    _app, win = app_and_window

    # 1. Load the project file
    success = win.doc_editor.file.load_project_from_path(
        workpieces_project_path
    )
    assert success, "Failed to load project file"

    # Process events to allow the signal to propagate
    process_events_for_duration(0.5)

    # 2. Verify that the document was loaded with workpieces
    doc = win.doc_editor.doc
    workpieces_in_doc = list(doc.get_descendants(of_type=WorkPiece))
    assert len(workpieces_in_doc) > 0, (
        "Document should contain workpieces after loading"
    )
    logger.info(f"Document contains {len(workpieces_in_doc)} workpieces")

    # 3. Check the surface for WorkPieceElements
    # The bug is that the surface doesn't get updated, so there are
    # no WorkPieceElements even though the document has workpieces.
    workpiece_elements = list(win.surface.find_by_type(WorkPieceElement))
    logger.info(f"Surface has {len(workpiece_elements)} WorkPieceElements")

    # This assertion will fail when the bug is present
    # (i.e., workpieces are in the doc but not on the surface)
    assert len(workpiece_elements) == len(workpieces_in_doc), (
        f"Surface should have {len(workpieces_in_doc)} WorkPieceElements "
        f"matching the document, but found {len(workpiece_elements)}. "
        "This indicates the broken import bug."
    )
