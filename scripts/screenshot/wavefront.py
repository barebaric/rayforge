"""
Screenshot: Wavefront operation in the main window.

Usage: pixi run screenshot operations:wavefront
"""

import logging
import time

from utils import (
    clear_window_subtitle,
    load_project,
    restore_config,
    set_window_size,
    take_cropped_screenshot,
    wait_for_settled,
)

from rayforge.uiscript import app, win

logger = logging.getLogger(__name__)


@restore_config
def main():
    set_window_size(win, 2400, 1650)

    load_project(win, "wavefront.ryp")
    logger.info("Waiting for document to settle...")
    if not wait_for_settled(win, timeout=30):
        logger.error("Document did not settle in time")
        app.quit_idle()
        return

    logger.info("Document settled")

    clear_window_subtitle(win)
    time.sleep(0.25)

    logger.info("Taking cropped screenshot: operations-wavefront.webp")
    take_cropped_screenshot(
        from_left=880,
        from_right=880,
        from_top=590,
        from_bottom=810,
    )

    time.sleep(0.25)
    app.quit_idle()


main()
