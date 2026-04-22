#!/usr/bin/env python3
"""
Screenshot: Main window in 3D mode with rotary project.

Usage: pixi run screenshot main:3d-rotary
"""

import time
import logging
from rayforge.uiscript import app, win
from rayforge.ui_gtk.sim3d.camera import ViewDirection
from utils import (
    load_project,
    wait_for_settled,
    wait_for_3d_rendered,
    seek_3d_playback,
    show_panel,
    hide_panel,
    show_bottom_tab,
    save_panel_states,
    restore_panel_states,
    take_screenshot,
    clear_window_subtitle,
    run_on_main_thread,
    set_window_size,
    wcs,
)

logger = logging.getLogger(__name__)

PANELS = ["show_3d_view", "toggle_bottom_panel"]


def main():
    set_window_size(win, 2400, 1650)

    load_project(win, "rotary.ryp")
    logger.info("Waiting for document to settle...")

    if not wait_for_settled(win, timeout=20):
        logger.error("Document did not settle in time")
        app.quit_idle()
        return

    logger.info("Setting up 3D mode with rotary project")

    saved_states = save_panel_states(win, PANELS)

    with wcs(win, "G55"):
        show_panel(win, "show_3d_view", True)
        hide_panel(win, "toggle_bottom_panel")
        show_bottom_tab(win, "gcode")

        logger.info(
            "Waiting for pipeline to settle after 3D view activation..."
        )
        if not wait_for_settled(win, timeout=30):
            logger.error("Pipeline did not settle after 3D view activation")
            app.quit_idle()
            return

        logger.info("Waiting for 3D scene to render...")
        if not wait_for_3d_rendered(win, timeout=15):
            logger.error("3D scene did not render in time")
            app.quit_idle()
            return

        run_on_main_thread(
            lambda: win.view_cmd.set_view(ViewDirection.ISO, win.canvas3d)
        )

        def _set_perspective() -> None:
            if win.canvas3d and win.canvas3d.camera:
                win.canvas3d.camera.is_perspective = True
                win.canvas3d.queue_render()

        run_on_main_thread(_set_perspective)
        time.sleep(0.5)

        seek_3d_playback(win, 0.6)

        clear_window_subtitle(win)
        logger.info("Taking screenshot: main-3d-rotary.png")
        take_screenshot("main-3d-rotary.png")

    restore_panel_states(win, saved_states)

    time.sleep(0.25)
    app.quit_idle()


main()
