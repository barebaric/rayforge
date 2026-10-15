"""Screenshot: Machine settings - Camera page and dialogs."""

import logging
import subprocess
import time

from gi.repository import GLib, Gtk
from utils import (
    align_mock_camera,
    get_target,
    inject_mock_image,
    open_machine_settings,
    restore_config,
    run_on_main_thread,
    take_screenshot,
)

from rayforge.camera.models.camera import Camera
from rayforge.context import get_context
from rayforge.ui_gtk.camera.alignment_dialog import CameraAlignmentDialog
from rayforge.ui_gtk.camera.image_settings_dialog import (
    CameraImageSettingsDialog,
)
from rayforge.ui_gtk.camera.lens_calibration_dialog import (
    LensCalibrationDialog,
)
from rayforge.ui_gtk.camera.wizard.wizard import CameraWizard
from rayforge.uiscript import app, win

logger = logging.getLogger(__name__)

PAGE = "camera"
TARGET = get_target(f"machine-settings:{PAGE}")

WIZARD_PAGES = ("card", "capture")

DIALOGS = {
    "image-settings": {"dialog_cls": CameraImageSettingsDialog},
    "lens-calibration": {"dialog_cls": LensCalibrationDialog},
    "image-alignment": {"dialog_cls": CameraAlignmentDialog},
}


def parse_target(target: str) -> dict | None:
    """Parse target into a handler config, or None for the settings page."""
    parts = target.split(":")
    # machine-settings:camera -> None (page screenshot)
    # machine-settings:camera:image-settings -> dialog
    # machine-settings:camera:calibration-wizard:card -> wizard
    if len(parts) <= 2:
        return None
    sub = ":".join(parts[2:])
    if sub.startswith("lens-calibration:wizard-"):
        wizard_page = sub.split(":wizard-")[1] if ":wizard-" in sub else ""
        if wizard_page not in WIZARD_PAGES:
            logger.error(f"Unknown wizard page: {wizard_page}")
            return None
        return {"type": "wizard", "page": wizard_page}
    if sub in DIALOGS:
        return {"type": "dialog", "key": sub}
    logger.error(f"Unknown target: {target}")
    return None


def add_mock_camera(dialog):
    """Add a mock camera, select it in the list, and inject mock image."""
    machine = dialog.machine

    def _add():
        camera = Camera("Test Camera", "mock-device-0")
        camera.enabled = True
        machine.add_camera(camera)
        return camera

    camera = run_on_main_thread(_add)
    time.sleep(0.5)

    camera_page = dialog.camera_page
    list_box = camera_page.camera_list_editor.list_box

    def _select():
        row = list_box.get_row_at_index(0)
        if row:
            list_box.select_row(row)

    run_on_main_thread(_select)
    time.sleep(0.25)

    controller = get_context().camera_mgr.get_controller("mock-device-0")
    if controller:
        inject_mock_image(controller)
    return camera, controller


def setup_camera_page(dialog):
    """Ensure a mock camera exists and is selected on the camera page."""
    machine = dialog.machine

    def _has_camera():
        return len(machine.cameras) > 0

    if not run_on_main_thread(_has_camera):
        return add_mock_camera(dialog)

    controller = get_context().camera_mgr.get_controller("mock-device-0")
    if controller:
        inject_mock_image(controller)
    return None, controller


def _iter_descendants(widget):
    """Yield widget and all of its descendants, depth-first."""
    child = widget.get_first_child()
    while child:
        yield child
        yield from _iter_descendants(child)
        child = child.get_next_sibling()


def _scroll_camera_page_to_bottom(dialog):
    """Scroll the camera page so its bottommost group is visible.

    The value is pinned repeatedly rather than set once, because the
    page scrolls smoothly and a single set is only captured in-flight.
    """

    def _find_adjustment():
        for widget in _iter_descendants(dialog.camera_page):
            if isinstance(widget, Gtk.ScrolledWindow):
                adjustment = widget.get_vadjustment()
                if adjustment.get_upper() > adjustment.get_page_size():
                    return adjustment
        return None

    def _scroll():
        adjustment = _find_adjustment()
        if adjustment is None:
            logger.warning("No scroll adjustment found for the camera page")
            return
        remaining = [30]

        def _pin():
            adjustment.set_value(adjustment.get_upper())
            remaining[0] -= 1
            return remaining[0] > 0

        GLib.timeout_add(50, _pin)

    run_on_main_thread(_scroll)


def take_wizard_screenshot(parent_dialog, wizard_page: str):
    """Open the lens calibration wizard directly and take a screenshot."""

    _camera, controller = add_mock_camera(parent_dialog)
    if not controller:
        logger.error("Failed to create mock camera controller")
        return

    wizard = None

    def _open_wizard():
        nonlocal wizard
        wizard = CameraWizard(parent_dialog, controller)
        wizard.present()
        return wizard

    wizard = run_on_main_thread(_open_wizard)
    time.sleep(0.5)

    def _choose_automatic():
        # image settings -> lens choice -> (automatic) -> card -> capture
        choice = wizard._pages["lens_choice"]
        choice._on_branch_clicked(choice._automatic_btn)

    run_on_main_thread(_choose_automatic)
    time.sleep(0.5)

    if wizard_page == "capture":
        run_on_main_thread(lambda: wizard._navigate_to("capture"))
        time.sleep(0.5)

    # Activate the wizard window so it is the topmost window for
    # the capture
    try:
        subprocess.run(
            [
                "xdotool",
                "search",
                "--name",
                "Camera Wizard",
                "windowactivate",
            ],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
        time.sleep(0.25)
    except (FileNotFoundError, subprocess.TimeoutExpired):
        logger.warning("xdotool not available, relying on window focus")

    take_screenshot()

    def _cleanup():
        wizard.close()

    run_on_main_thread(_cleanup)
    time.sleep(0.25)


@restore_config
def main():
    target_info = parse_target(TARGET)
    time.sleep(0.25)
    dialog = open_machine_settings(win, PAGE)
    time.sleep(0.25)

    if target_info is None:
        camera, controller = setup_camera_page(dialog)
        camera = camera or (controller.config if controller else None)
        if camera is not None:
            width_mm, height_mm = dialog.machine.axis_extents
            align_mock_camera(camera, width_mm, height_mm)
            # Show the outside-workspace settings in their enabled state.
            camera.outside_view_enabled = True
        _scroll_camera_page_to_bottom(dialog)
        time.sleep(1.8)
        take_screenshot()
        time.sleep(0.25)
        app.quit_idle()
        return

    if target_info["type"] == "wizard":
        take_wizard_screenshot(dialog, target_info["page"])
        time.sleep(0.25)
        app.quit_idle()
        return

    config = DIALOGS.get(target_info["key"])
    if config is None:
        logger.error(f"Unknown camera dialog target: {target_info['key']}")
        app.quit_idle()
        return

    _camera, controller = add_mock_camera(dialog)
    if not controller:
        logger.error("Failed to create mock camera controller")
        app.quit_idle()
        return

    def _open():
        d = config["dialog_cls"](dialog, controller)
        d.present()
        return d

    camera_dialog = run_on_main_thread(_open)
    time.sleep(0.5)
    take_screenshot()

    def _close():
        camera_dialog.close()

    run_on_main_thread(_close)
    time.sleep(0.25)
    app.quit_idle()


main()
