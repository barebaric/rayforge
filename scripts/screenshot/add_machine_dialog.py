#!/usr/bin/env python3
"""
Screenshot: Add Machine dialog.

Usage: pixi run screenshot app-settings:machines:add
"""

import time
import logging
from rayforge.uiscript import app, win
from utils import (
    wait_for_settled,
    take_screenshot,
    run_on_main_thread,
    set_window_size,
)

logger = logging.getLogger(__name__)


def main():
    set_window_size(win, 1400, 1000)

    logger.info("Waiting for document to settle...")
    if not wait_for_settled(win, timeout=10):
        logger.error("Document did not settle in time")
        app.quit_idle()
        return

    from rayforge.ui_gtk.machine.profile_selector import (
        MachineProfileSelectorDialog,
    )

    def open_dialog():
        dialog = MachineProfileSelectorDialog(transient_for=win)
        dialog.present()
        return dialog

    dialog = run_on_main_thread(open_dialog)

    time.sleep(1.0)

    logger.info("Taking screenshot: app-settings-machines-add.png")
    take_screenshot("app-settings-machines-add.png")

    time.sleep(0.25)

    def close_dialog():
        dialog.close()

    run_on_main_thread(close_dialog)
    app.quit_idle()


main()
