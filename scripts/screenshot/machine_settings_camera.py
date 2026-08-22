"""Screenshot: Machine settings - Camera page and dialogs."""

import logging
import subprocess
import time
from pathlib import Path

import cv2
from gi.repository import GLib
from utils import (
    get_target,
    open_machine_settings,
    restore_config,
    run_on_main_thread,
    take_screenshot,
    target_to_filename,
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

MOCK_IMAGE_PATH = (
    Path(__file__).parent.parent.parent
    / "website"
    / "static"
    / "images"
    / "work-surface.png"
)

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


def load_mock_image():
    """Load the mock camera image as a BGR numpy array."""
    img = cv2.imread(str(MOCK_IMAGE_PATH), cv2.IMREAD_COLOR)
    if img is None:
        logger.error("Failed to load mock image from %s", MOCK_IMAGE_PATH)
        return None
    logger.info("Loaded mock image: %s (%s)", MOCK_IMAGE_PATH, img.shape)
    return img


def inject_mock_image(controller):
    """Inject mock image data into the controller and trigger redraw."""
    img = load_mock_image()
    if img is None:
        return False
    controller._raw_image_data = img.copy()
    controller._image_data = img.copy()
    GLib.idle_add(controller.image_captured.send, controller)
    # Prevent the real capture loop from starting (mock device can't open)
    controller._running = True
    time.sleep(0.25)
    return True


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


def take_wizard_screenshot(parent_dialog, wizard_page: str, output: str):
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

    # Activate the wizard window so gnome-screenshot -w captures it
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

    take_screenshot(output)

    def _cleanup():
        wizard.close()

    run_on_main_thread(_cleanup)
    time.sleep(0.25)


@restore_config
def main():
    target_info = parse_target(TARGET)
    output = target_to_filename(TARGET)
    time.sleep(0.25)
    dialog = open_machine_settings(win, PAGE)
    time.sleep(0.25)

    if target_info is None:
        setup_camera_page(dialog)
        time.sleep(0.25)
        take_screenshot(output)
        time.sleep(0.25)
        app.quit_idle()
        return

    if target_info["type"] == "wizard":
        take_wizard_screenshot(dialog, target_info["page"], output)
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
    take_screenshot(output)

    def _close():
        camera_dialog.close()

    run_on_main_thread(_close)
    time.sleep(0.25)
    app.quit_idle()


main()
