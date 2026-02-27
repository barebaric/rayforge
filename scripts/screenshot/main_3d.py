#!/usr/bin/env python3
"""
Screenshot: Main window in 3D mode.

Usage: pixi run screenshot main:3d
"""

import time
import logging
from rayforge.uiscript import app, win
from utils import (
    load_project,
    wait_for_settled,
    show_panel,
    hide_panel,
    save_panel_states,
    restore_panel_states,
    take_screenshot,
    clear_window_subtitle,
    run_on_main_thread,
    set_window_size,
)

logger = logging.getLogger(__name__)

PANELS = ["show_3d_view", "toggle_control_panel", "toggle_gcode_preview"]


def wait_for_3d_view(timeout: float = 10.0) -> bool:
    """
    Wait for the 3D view to be visible and rendered.

    Returns:
        True if 3D view is ready within timeout.
    """
    start = time.time()
    while time.time() - start < timeout:
        is_3d = run_on_main_thread(
            lambda: win.view_stack.get_visible_child_name() == "3d"
        )
        if is_3d:
            canvas = run_on_main_thread(lambda: win.canvas3d)
            if canvas is not None:
                time.sleep(0.5)
                logger.info("3D view is ready")
                return True
        time.sleep(0.1)
    logger.warning("3D view not ready within timeout")
    return False


def main():
    set_window_size(win, 2400, 1650)

    load_project(win, "pretty.ryp")
    logger.info("Waiting for document to settle...")

    if not wait_for_settled(win, timeout=20):
        logger.error("Document did not settle in time")
        app.quit_idle()
        return

    logger.info("Setting up 3D mode")

    saved_states = save_panel_states(win, PANELS)
    show_panel(win, "show_3d_view", True)
    hide_panel(win, "toggle_control_panel")
    show_panel(win, "toggle_gcode_preview", True)

    logger.info("Waiting for 3D view to render...")
    if not wait_for_3d_view(timeout=10):
        logger.error("3D view did not initialize in time")
        app.quit_idle()
        return

    time.sleep(0.5)

    clear_window_subtitle(win)
    logger.info("Taking screenshot: main-3d.png")
    take_screenshot("main-3d.png")

    restore_panel_states(win, saved_states)

    time.sleep(0.25)
    app.quit_idle()


main()
