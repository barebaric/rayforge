#!/usr/bin/env python3
"""
Screenshot: Main window in standard mode.

Usage: pixi run screenshot main
"""

import time
import logging
from rayforge.uiscript import app, win
from utils import (
    load_project,
    wait_for_settled,
    show_panel,
    save_panel_states,
    restore_panel_states,
    take_screenshot,
)

logger = logging.getLogger(__name__)

PANELS = ["toggle_control_panel", "toggle_gcode_preview"]


def main():
    win.set_default_size(2400, 1650)
    logger.info("Window size set to 2400x1650")

    load_project(win, "contour.ryp")
    logger.info("Waiting for document to settle...")
    if not wait_for_settled(win, timeout=10):
        logger.error("Document did not settle in time")
        app.quit_idle()
        return

    logger.info("Document settled, setting up standard mode")

    saved_states = save_panel_states(win, PANELS)
    show_panel(win, "toggle_control_panel", True)
    show_panel(win, "toggle_gcode_preview", True)

    time.sleep(0.25)

    logger.info("Taking screenshot: main-standard.png")
    take_screenshot("main-standard.png")

    restore_panel_states(win, saved_states)

    time.sleep(0.25)
    app.quit_idle()


main()
