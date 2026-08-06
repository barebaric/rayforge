#!/usr/bin/env python3
"""
Screenshot: Bottom panel dock tabs (cropped to just the panel).

Usage: pixi run screenshot bottom-panel
       pixi run screenshot bottom-panel:console
       pixi run screenshot bottom-panel:layers
"""

import logging
import time

from utils import (
    get_target,
    load_project,
    restore_panel_states,
    run_on_main_thread,
    save_panel_states,
    set_window_size,
    show_bottom_tab,
    show_panel,
    take_cropped_screenshot,
    target_to_filename,
    wait_for_settled,
)

from rayforge.uiscript import app, win

logger = logging.getLogger(__name__)

PANEL_HEIGHT = 280
MARGIN = 10
STATUS_BAR_HEIGHT = 60
PANELS = ["toggle_bottom_panel"]

TAB_CONFIG = {
    "console": {"project": "contour.ryp"},
    "layers": {"project": "twolayer.ryp"},
}


def get_tab_name(target: str) -> str:
    parts = target.split(":")
    if len(parts) > 1:
        return parts[1]
    return "console"


def main():
    target = get_target("bottom-panel:console")
    tab_name = get_tab_name(target)
    config = TAB_CONFIG.get(tab_name, TAB_CONFIG["console"])

    set_window_size(win, 1400, 900)

    load_project(win, config["project"])
    logger.info("Waiting for document to settle...")
    if not wait_for_settled(win, timeout=10):
        logger.error("Document did not settle in time")
        app.quit_idle()
        return

    logger.info("Document settled, showing bottom panel")

    saved_states = save_panel_states(win, PANELS)
    show_panel(win, "toggle_bottom_panel", True)
    show_bottom_tab(win, tab_name)

    time.sleep(0.5)

    window_height = run_on_main_thread(lambda: win.get_height())
    crop_from_top = window_height - PANEL_HEIGHT - MARGIN - STATUS_BAR_HEIGHT

    logger.info(
        f"Taking cropped screenshot of '{tab_name}' tab "
        f"(window height={window_height}, crop_top={crop_from_top})"
    )
    take_cropped_screenshot(
        target_to_filename(target),
        from_top=crop_from_top,
    )

    restore_panel_states(win, saved_states)

    time.sleep(0.25)
    app.quit_idle()


main()
