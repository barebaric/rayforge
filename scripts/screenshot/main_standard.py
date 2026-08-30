"""
Screenshot: Main window in standard mode.

Usage: pixi run screenshot main
"""

import logging
import time

from utils import (
    clear_window_subtitle,
    load_project,
    restore_config,
    restore_panel_states,
    save_panel_states,
    set_window_size,
    show_bottom_tab,
    show_panel,
    take_screenshot,
    wait_for_settled,
)

from rayforge.uiscript import app, win

logger = logging.getLogger(__name__)

PANELS = ["toggle_bottom_panel"]


@restore_config
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
    logger.info("Taking screenshot: main-standard.webp")
    take_screenshot()

    restore_panel_states(win, saved_states)

    time.sleep(0.25)
    app.quit_idle()


main()
