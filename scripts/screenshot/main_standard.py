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
    show_bottom_tab,
    save_panel_states,
    restore_panel_states,
    take_screenshot,
    clear_window_subtitle,
    set_window_size,
)

logger = logging.getLogger(__name__)

PANELS = ["toggle_bottom_panel"]


def main():
    set_window_size(win, 2400, 1650)

    load_project(win, "contour.ryp")
    logger.info("Waiting for document to settle...")
    if not wait_for_settled(win, timeout=10):
        logger.error("Document did not settle in time")
        app.quit_idle()
        return

    logger.info("Document settled, setting up standard mode")

    saved_states = save_panel_states(win, PANELS)
    show_panel(win, "toggle_bottom_panel", True)
    show_bottom_tab(win, "gcode")

    time.sleep(0.25)

    clear_window_subtitle(win)
    logger.info("Taking screenshot: main-standard.png")
    take_screenshot("main-standard.png")

    restore_panel_states(win, saved_states)

    time.sleep(0.25)
    app.quit_idle()


main()
