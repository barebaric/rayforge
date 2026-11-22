# flake8: noqa: E402
import asyncio
import logging
import os
import sys
import threading
import time
from pathlib import Path
from typing import Any

import pytest

from rayforge.context import get_context
from rayforge.core.workpiece import WorkPiece
from rayforge.machine.models.machine import Machine

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

logger = logging.getLogger(__name__)


# Helper functions adapted for robust testing


def process_events_for_duration(duration_sec: float):
    """
    Processes all pending GTK events for a given duration without blocking.
    """
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


@pytest.fixture
def assets_path() -> Path:
    return Path(__file__).parent.parent


@pytest.fixture
def test_file_path(assets_path: Path) -> Path:
    path = assets_path / "image" / "png" / "color.png"
    assert path.exists()
    return path


@pytest.fixture
def app_and_window(ui_context_initializer, request):
    """Sets up the Adw.Application and MainWindow without blocking."""
    from rayforge.ui_gtk import sim3d

    sim3d.initialize()
    assert sim3d.initialized, "Canvas3D failed to initialize"

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
def test_rename_shortcut_reveals_layer_tab(app_and_window, assets_path):
    """F2 rename works for any selection and shows the layer tab."""
    _app, win = app_and_window
    project = assets_path / "doceditor" / "assets" / "workpieces_project.ryp"
    assert win.doc_editor.file.load_project_from_path(project)
    process_events_for_duration(0.5)

    wp = next(iter(win.doc_editor.doc.get_descendants(of_type=WorkPiece)))

    def row_widget() -> Any:
        for col in win.bottom_panel.layers_tab._columns:
            for row, item in col._row_items.items():
                if item is wp:
                    return row.get_child()
        return None

    # A canvas-style selection: rename works and reveals the layer tab.
    win.bottom_panel.set_visible(False)
    win.surface.select_items([wp])
    win.on_menu_rename(None, None)
    area = win.bottom_panel.dock_layout.find_item_area("layers")
    assert win.bottom_panel.get_visible()
    assert area.get_active_item() == "layers"
    widget = row_widget()
    assert widget is not None
    assert widget._rename_entry is not None


@pytest.mark.ui
def test_machine_settings_dialog_is_single_instance(app_and_window):
    """Machine settings opens as a single reusable dialog instance."""
    _app, win = app_and_window
    get_context().machine_mgr.create_default_machine()

    win.show_machine_settings(None, None)
    process_events_for_duration(0.3)
    assert win._machine_settings_dialog is not None
    first = win._machine_settings_dialog

    win.show_machine_settings(None, None)
    process_events_for_duration(0.3)
    assert win._machine_settings_dialog is first

    first.close()
    process_events_for_duration(0.3)
    assert win._machine_settings_dialog is None

    win.show_machine_settings(None, None)
    process_events_for_duration(0.3)
    assert win._machine_settings_dialog is not None
    assert win._machine_settings_dialog is not first


@pytest.mark.ui
def test_machine_settings_dialog_recreated_after_machine_switch(
    app_and_window,
):
    """Switching the active machine invalidates the open dialog."""
    _app, win = app_and_window
    mgr = get_context().machine_mgr
    machine_a = mgr.create_default_machine()
    get_context().config.set_machine(machine_a)

    win.show_machine_settings(None, None)
    process_events_for_duration(0.3)
    first = win._machine_settings_dialog
    assert first is not None
    assert first.machine is machine_a

    machine_b = mgr.create_default_machine()
    get_context().config.set_machine(machine_b)

    win.show_machine_settings(None, None)
    process_events_for_duration(0.3)
    assert win._machine_settings_dialog is not first
    assert win._machine_settings_dialog.machine is machine_b
    # The stale dialog for the old machine was closed automatically.
    assert not first.get_visible()

    # Closing the new dialog still resets the tracked reference.
    win._machine_settings_dialog.close()
    process_events_for_duration(0.3)
    assert win._machine_settings_dialog is None


def _enable_pointer_alignment(machine):
    from rayforge.machine.models.machine import Origin

    machine.set_axis_extents(400, 300)
    machine.set_origin(Origin.BOTTOM_LEFT)
    head = machine.get_default_laser_head()
    assert head is not None
    head.set_pointer_offset(10.0, 20.0)
    head.set_pointer_offset_enabled(True)
    machine.set_pointer_alignment(True)


@pytest.mark.ui
def test_click_to_move_shifted_with_alignment(app_and_window, mocker):
    """Click-to-move commands are shifted by -offset while pointer
    alignment is on, so the pointer dot lands on the clicked spot."""
    _app, win = app_and_window
    machine = get_context().config.machine
    assert machine is not None
    _enable_pointer_alignment(machine)
    machine.set_gcode_precision(2)

    move_to = mocker.patch.object(win.machine_cmd, "move_to")

    win._on_move_head_requested(None, x=100.126, y=50.124)

    move_to.assert_called_once()
    args = move_to.call_args.args
    assert args[0] is machine
    assert args[1] == pytest.approx(90.13)
    assert args[2] == pytest.approx(30.12)


@pytest.mark.ui
def test_click_to_move_shifted_with_alignment_workarea_origin(
    app_and_window, mocker
):
    """Click-to-move keeps the pointer shift when the workarea origin
    is the active reference."""
    _app, win = app_and_window
    machine = get_context().config.machine
    assert machine is not None
    _enable_pointer_alignment(machine)
    machine.set_wcs_origin_is_workarea_origin(True)
    machine.set_work_margins(5.0, 7.0, 0.0, 0.0)
    machine.set_gcode_precision(2)

    move_to = mocker.patch.object(win.machine_cmd, "move_to")

    win._on_move_head_requested(None, x=100.126, y=50.124)

    move_to.assert_called_once()
    args = move_to.call_args.args
    assert args[0] is machine
    assert args[1] == pytest.approx(85.13)
    assert args[2] == pytest.approx(30.12)


@pytest.mark.ui
def test_move_head_here_shifted_with_alignment_workarea_origin(
    app_and_window, mocker
):
    """Move-Head-Here keeps the pointer shift when the workarea origin
    is the active reference."""
    _app, win = app_and_window
    machine = get_context().config.machine
    assert machine is not None
    _enable_pointer_alignment(machine)
    machine.set_wcs_origin_is_workarea_origin(True)
    machine.set_work_margins(5.0, 7.0, 0.0, 0.0)
    win.surface.right_click_machine_pos = (100.0, 50.0)

    move_to = mocker.patch.object(win.machine_cmd, "move_to")

    win.on_move_head_here_clicked(None, None)

    move_to.assert_called_once()
    args = move_to.call_args.args
    assert args[0] is machine
    assert args[1] == pytest.approx(85.0)
    assert args[2] == pytest.approx(30.0)


@pytest.mark.ui
def test_click_to_set_work_origin_uses_machine_precision(
    app_and_window, mocker
):
    """Click-to-zero rounds coordinates to the machine's G-code precision."""
    _app, win = app_and_window
    machine = get_context().config.machine
    assert machine is not None
    machine.set_gcode_precision(3)

    set_work_origin = mocker.patch.object(machine, "set_work_origin")
    add_coroutine = mocker.patch(
        "rayforge.ui_gtk.mainwindow.task_mgr.add_coroutine"
    )

    win._on_work_zero_requested(None, x=12.3456, y=7.8914)

    add_coroutine.assert_called_once()
    asyncio.run(add_coroutine.call_args.args[0](None))
    set_work_origin.assert_awaited_once_with(12.346, 7.891, 0.0)


@pytest.mark.ui
def test_move_head_here_shifted_with_alignment(app_and_window, mocker):
    """Move-Head-Here commands are shifted by -offset while pointer
    alignment is on, so the pointer dot lands on the chosen spot."""
    _app, win = app_and_window
    machine = get_context().config.machine
    assert machine is not None
    _enable_pointer_alignment(machine)
    win.surface.right_click_machine_pos = (100.0, 50.0)

    move_to = mocker.patch.object(win.machine_cmd, "move_to")

    win.on_move_head_here_clicked(None, None)

    move_to.assert_called_once()
    args = move_to.call_args.args
    assert args[0] is machine
    assert args[1] == pytest.approx(90.0)
    assert args[2] == pytest.approx(30.0)


def _add_machine(mgr, context, name: str) -> Machine:
    machine = Machine(context)
    machine.name = name
    machine.auto_connect = False
    mgr.add_machine(machine)
    return machine


@pytest.mark.ui
def test_machine_menu_lists_machines(app_and_window):
    """The Machine menu lists every configured machine, sorted by name."""
    _app, win = app_and_window
    context = get_context()
    mgr = context.machine_mgr
    machine_b = _add_machine(mgr, context, "Machine B")
    machine_a = _add_machine(mgr, context, "Machine A")
    context.config.set_machine(machine_a)
    process_events_for_duration(0.3)

    section = win.menu_model.dynamic_machines_section
    labels = [
        section.get_item_attribute_value(i, "label", None).get_string()
        for i in range(section.get_n_items())
    ]
    assert len(labels) == len(mgr.get_machines())
    assert {machine_a.name, machine_b.name} <= set(labels)
    assert labels == sorted(labels)

    action = win.action_manager.get_action("select-machine")
    assert action.get_state().get_string() == machine_a.id


@pytest.mark.ui
def test_machine_menu_selection_switches_active_machine(app_and_window):
    """Activating a machine entry makes that machine the active one."""
    _app, win = app_and_window
    context = get_context()
    mgr = context.machine_mgr
    machine_a = _add_machine(mgr, context, "Machine A")
    machine_b = _add_machine(mgr, context, "Machine B")
    context.config.set_machine(machine_a)
    process_events_for_duration(0.3)

    action = win.action_manager.get_action("select-machine")
    action.activate(GLib.Variant.new_string(machine_b.id))
    process_events_for_duration(0.8)

    assert context.config.machine is machine_b
    assert action.get_state().get_string() == machine_b.id


@pytest.mark.ui
def test_machine_add_action_opens_wizard(app_and_window):
    """The Machine menu's Add Machine entry opens the setup wizard."""
    _app, win = app_and_window

    win.action_manager.get_action("machine-add").activate(None)
    process_events_for_duration(0.3)

    try:
        assert win._setup_wizard is not None
    finally:
        if win._setup_wizard is not None:
            win._setup_wizard.close()


@pytest.mark.ui
def test_machine_manage_action_opens_machines_page(app_and_window, mocker):
    """The Manage Machines entry opens Settings on the Machines page."""
    _app, win = app_and_window
    settings_cls = mocker.patch("rayforge.ui_gtk.mainwindow.SettingsWindow")

    win.action_manager.get_action("machine-manage").activate(None)

    settings_cls.assert_called_once()
    assert settings_cls.call_args.kwargs["initial_page"] == "machines"
