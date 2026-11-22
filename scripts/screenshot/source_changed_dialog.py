"""
Screenshot: Source changed dialog.

Usage: pixi run screenshot source-changed-dialog
"""

import logging
import time
from pathlib import Path

from utils import (
    restore_config,
    run_on_main_thread,
    set_window_size,
    take_screenshot,
    wait_for_settled,
)

from rayforge.uiscript import app, win

logger = logging.getLogger(__name__)

SOURCE_FILE = Path("~/designs/badge.svg").expanduser()


@restore_config
def main():
    set_window_size(win, 1200, 900)

    logger.info("Waiting for document to settle...")
    if not wait_for_settled(win, timeout=10):
        logger.error("Document did not settle in time")
        app.quit_idle()
        return

    from rayforge.core.source_asset import SourceAsset
    from rayforge.image import SVG_RENDERER
    from rayforge.ui_gtk.doceditor.source_changed_dialog import (
        SourceChangedDialog,
    )

    def open_dialog():
        asset = SourceAsset(
            source_file=SOURCE_FILE,
            original_data=b"",
            renderer=SVG_RENDERER,
        )
        dialog = SourceChangedDialog([asset])
        dialog.present(win)
        return dialog

    dialog = run_on_main_thread(open_dialog)

    time.sleep(1.0)

    logger.info("Taking screenshot: source-changed-dialog.webp")
    take_screenshot()

    time.sleep(0.25)

    def close_dialog():
        dialog.close()

    run_on_main_thread(close_dialog)
    app.quit_idle()


main()
