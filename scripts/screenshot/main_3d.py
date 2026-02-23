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
)

logger = logging.getLogger(__name__)

PANELS = ["show_3d_view", "toggle_control_panel", "toggle_gcode_preview"]


def main():
    win.set_default_size(2400, 1650)
    logger.info("Window size set to 2400x1650")

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
    if not wait_for_settled(win, timeout=20):
        logger.error("3D view did not settle in time")
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
