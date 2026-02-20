#!/usr/bin/env python3
"""
Screenshot: Import dialog.

Usage: pixi run screenshot import-dialog
"""

import time
import logging
from pathlib import Path
from rayforge.uiscript import app, win
from utils import (
    wait_for_settled,
    take_screenshot,
    _run_on_main_thread,
)

logger = logging.getLogger(__name__)

TEST_IMAGE = (
    Path(__file__).parent.parent.parent
    / "tests"
    / "image"
    / "svg"
    / "rayforge.svg"
)


def main():
    win.set_default_size(1400, 1000)
    logger.info("Window size set to 1400x1000")

    logger.info("Waiting for document to settle...")
    if not wait_for_settled(win, timeout=10):
        logger.error("Document did not settle in time")
        app.quit_idle()
        return

    from rayforge.ui_gtk.doceditor.import_dialog import ImportDialog

    def open_import_dialog():
        _, features = win.doc_editor.file.get_importer_info(
            TEST_IMAGE, "image/svg+xml"
        )
        dialog = ImportDialog(
            parent=win,
            editor=win.doc_editor,
            file_path=TEST_IMAGE,
            mime_type="image/svg+xml",
            features=features,
        )
        dialog.set_default_size(1100, 800)
        dialog.present()
        return dialog

    dialog = _run_on_main_thread(open_import_dialog)

    time.sleep(1.0)

    logger.info("Taking screenshot: import-dialog.png")
    take_screenshot("import-dialog.png")

    time.sleep(0.25)

    def close_dialog():
        dialog.close()

    _run_on_main_thread(close_dialog)
    app.quit_idle()


main()
