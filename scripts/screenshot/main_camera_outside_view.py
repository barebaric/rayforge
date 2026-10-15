"""
Screenshot: Main window with the camera view outside the workspace.

Usage: pixi run screenshot main:camera-outside-view
"""

import logging
import time

from gi.repository import GLib
from utils import (
    align_mock_camera,
    clear_window_subtitle,
    inject_mock_image,
    restore_config,
    run_on_main_thread,
    set_window_size,
    take_screenshot,
    wait_for_settled,
)

from rayforge.camera.models.camera import Camera
from rayforge.context import get_context
from rayforge.ui_gtk.canvas2d.elements.camera_image import CameraImageElement
from rayforge.uiscript import app, win

logger = logging.getLogger(__name__)


@restore_config
def main():
    set_window_size(win, 2400, 1650)

    machine = get_context().config.machine
    width_mm, height_mm = machine.axis_extents

    camera = Camera("Test Camera", "mock-device-0")
    camera.enabled = True
    camera.transparency = 0.65
    align_mock_camera(camera, width_mm, height_mm)
    camera.outside_view_margin_mm = 30
    camera.outside_view_enabled = True

    def _add_camera():
        machine.add_camera(camera)
        return camera

    run_on_main_thread(_add_camera)
    time.sleep(0.5)

    controller = get_context().camera_mgr.get_controller(camera.id)
    if not controller:
        logger.error("Failed to create mock camera controller")
        app.quit_idle()
        return

    def _sync_elements():
        # The work surface only picks up new cameras on machine changes,
        # which can race the creation of the controller. Re-sync now that
        # the controller definitely exists.
        win.surface._sync_camera_elements()

    run_on_main_thread(_sync_elements)
    inject_mock_image(controller)

    def _show_camera_overlay():
        action = win.action_manager.get_action("toggle_camera_view")
        action.change_state(GLib.Variant.new_boolean(True))
        # Zoom out so the margin around the workspace is on screen.
        win.surface.set_pan(0.0, 0.0)
        win.surface.set_zoom(0.72)

    run_on_main_thread(_show_camera_overlay)
    time.sleep(1.0)

    def _check():
        elems = list(win.surface.find_by_type(CameraImageElement))
        if not elems or not elems[0].visible:
            logger.error("Camera element is not visible on the work surface")
        return False

    GLib.timeout_add(500, _check)
    if not wait_for_settled(win, timeout=10):
        logger.error("Document did not settle in time")
        app.quit_idle()
        return

    clear_window_subtitle(win)
    logger.info("Taking screenshot: main-camera-outside-view.webp")
    take_screenshot()
    time.sleep(0.25)
    app.quit_idle()


main()
