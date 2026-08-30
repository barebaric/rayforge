# flake8: noqa: E402
import logging
import os
import sys
import time

import pytest

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

from rayforge.config import BUILTIN_DEVICES_DIR
from rayforge.context import get_context
from rayforge.machine.device.profile import DeviceProfile
from rayforge.machine.models.machine import Machine
from rayforge.ui_gtk.mainwindow import MainWindow

logger = logging.getLogger(__name__)


def process_events_for_duration(duration_sec: float):
    """Processes all pending GTK events for a given duration."""
    end_time = time.monotonic() + duration_sec
    context = GLib.main_context_default()
    while time.monotonic() < end_time:
        while context.pending():
            context.iteration(False)
        time.sleep(0.01)


@pytest.fixture
def first_run_app_and_window(ui_context_initializer, request):
    """Sets up an app/window in a fresh (first-run) configuration."""
    from rayforge.ui_gtk import sim3d

    sim3d.initialize()
    assert sim3d.initialized, "Canvas3D failed to initialize"

    context = ui_context_initializer
    # Simulate a fresh install: a single untouched placeholder machine.
    machine_mgr = context.machine_mgr
    if not machine_mgr.machines:
        machine_mgr.create_default_machine()

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

    if win:
        win.doc_editor.cleanup()
        win.close()
        app.quit()
    process_events_for_duration(0.2)


@pytest.fixture
def setup_app_and_window(ui_context_initializer, request):
    """Sets up an app/window whose setup is already completed."""
    from rayforge.ui_gtk import sim3d

    sim3d.initialize()
    assert sim3d.initialized, "Canvas3D failed to initialize"

    context = ui_context_initializer
    context.machine_mgr.create_default_machine()
    context.config.setup_completed = True

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

    if win:
        win.doc_editor.cleanup()
        win.close()
        app.quit()
    process_events_for_duration(0.2)


@pytest.mark.ui
def test_first_launch_presents_wizard(first_run_app_and_window):
    """On first launch the setup wizard opens automatically."""
    _app, win = first_run_app_and_window
    GLib.idle_add(win._maybe_present_first_run_wizard)
    process_events_for_duration(0.5)

    assert win._setup_wizard is not None
    assert win._setup_wizard.get_visible()


@pytest.mark.ui
def test_first_launch_wizard_cancel_falls_back(first_run_app_and_window):
    """Cancelling the wizard keeps the placeholder and sets the flag."""
    _app, win = first_run_app_and_window
    from rayforge.context import get_context

    ctx = get_context()
    config = ctx.config
    placeholder_id = next(iter(ctx.machine_mgr.machines))

    GLib.idle_add(win._maybe_present_first_run_wizard)
    process_events_for_duration(0.5)
    assert win._setup_wizard is not None

    win._setup_wizard.close()
    process_events_for_duration(0.5)

    assert config.setup_completed is True
    assert win._setup_wizard is None
    # The placeholder machine is still there so the app stays usable.
    assert placeholder_id in ctx.machine_mgr.machines


def _machine_with_pending_profile_review(context) -> Machine:
    """Adds a real machine whose source profile has unreviewed changes."""
    profile = DeviceProfile.from_path(
        BUILTIN_DEVICES_DIR / "sculpfun-icube-3w"
    )
    machine = profile.create_machine(context)
    machine.reviewed_profile_hash = None
    base_speed = profile.machine_config.max_cut_speed or 0
    machine.max_cut_speed = base_speed + 100
    return machine


@pytest.mark.ui
def test_startup_defers_profile_reviews_while_consent_pending(
    setup_app_and_window,
):
    """While the consent dialog is up, no profile-review dialogs may
    present; otherwise one of the modals ends up hidden behind the
    other but keeps the input grab, leaving buttons unresponsive."""
    _app, win = setup_app_and_window
    ctx = get_context()
    machine = _machine_with_pending_profile_review(ctx)
    presented = []
    win._present_profile_reviews = lambda queue, on_done=None: (
        presented.append(queue)
    )

    # The map handler already ran (and disconnected) when the fixture
    # presented the window; reattach it to simulate a fresh startup.
    win.connect("map", win._trigger_startup_tasks)
    win._trigger_startup_tasks(None)
    process_events_for_duration(0.5)
    assert presented == []

    win._on_consent_responded(None, "accept")
    process_events_for_duration(0.5)
    assert len(presented) == 1
    assert presented[0][0][0] is machine


@pytest.mark.ui
def test_closing_wizard_triggers_profile_reviews(first_run_app_and_window):
    """The profile-review check runs only after the setup wizard is
    closed, not alongside it at startup."""
    _app, win = first_run_app_and_window
    calls = []
    original = win._check_profile_updates

    def spy():
        calls.append(1)
        return original()

    win._check_profile_updates = spy

    win._maybe_present_first_run_wizard()
    process_events_for_duration(0.5)
    assert win._setup_wizard is not None
    assert not calls

    win._setup_wizard.close()
    process_events_for_duration(0.5)
    assert calls


@pytest.mark.ui
def test_banner_shown_while_placeholder_active(setup_app_and_window):
    """The banner is visible while the active machine is a placeholder."""
    _app, win = setup_app_and_window
    assert win.setup_banner.get_revealed() is True


@pytest.mark.ui
def test_banner_opens_wizard_manually(setup_app_and_window):
    """The banner button opens the setup wizard."""
    _app, win = setup_app_and_window
    win.setup_banner.emit("button-clicked")
    process_events_for_duration(0.3)

    assert win._setup_wizard is not None
