#!/usr/bin/env python3
"""
Screenshot: Control panel (cropped to just the panel).

Usage: pixi run screenshot control-panel
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
    take_cropped_screenshot,
    run_on_main_thread,
    set_window_size,
)

logger = logging.getLogger(__name__)

PANEL_HEIGHT = 280
MARGIN = 10
STATUS_BAR_HEIGHT = 60
PANELS = ["toggle_control_panel", "toggle_gcode_preview"]


def main():
    set_window_size(win, 1000, 900)

    load_project(win, "contour.ryp")
    logger.info("Waiting for document to settle...")
    if not wait_for_settled(win, timeout=10):
        logger.error("Document did not settle in time")
        app.quit_idle()
        return

    logger.info("Document settled, showing control panel")

    saved_states = save_panel_states(win, PANELS)
    show_panel(win, "toggle_control_panel", True)
    show_panel(win, "toggle_gcode_preview", False)

    time.sleep(0.5)

    window_height = run_on_main_thread(lambda: win.get_height())
    crop_from_top = window_height - PANEL_HEIGHT - MARGIN - STATUS_BAR_HEIGHT

    logger.info(
        f"Taking cropped screenshot (window height={window_height}, "
        f"crop_top={crop_from_top})"
    )
    take_cropped_screenshot(
        "control-panel.png",
        from_top=crop_from_top,
    )

    restore_panel_states(win, saved_states)

    time.sleep(0.25)
    app.quit_idle()


main()
