#!/usr/bin/env python3
"""
Screenshot: Point rotation array dialog with canvas showing array preview.

Usage: pixi run screenshot point-rotation-array-dialog
"""

import logging
import time

from utils import (
    clear_window_subtitle,
    load_project,
    open_array_dialog,
    run_on_main_thread,
    set_window_size,
    take_window_screenshot,
    wait_for_settled,
)

from rayforge.uiscript import app, win

logger = logging.getLogger(__name__)


def _select_all_items():
    """Select all content items on the active layer."""
    win.surface.update_from_doc()
    layer = win.doc_editor.doc.active_layer
    items = layer.get_content_items()
    if items:
        win.surface.select_items(items)
    return items


def _configure_point_rotation_dialog(dialog):
    """Set point rotation to 8 copies, 360 degrees."""
    dialog._updating = True
    try:
        dialog._pr_count_row.set_value(8)
        dialog._pr_angle_row.set_value(360.0)
    finally:
        dialog._updating = False
    dialog._update_preview()


def main():
    set_window_size(win, 2400, 1650)

    load_project(win, "pattern.ryp")
    logger.info("Waiting for document to settle...")
    if not wait_for_settled(win, timeout=10):
        logger.error("Document did not settle in time")
        app.quit_idle()
        return

    logger.info("Document settled")

    items = run_on_main_thread(_select_all_items)
    if not items:
        logger.error("No items found in document")
        app.quit_idle()
        return

    logger.info(f"Selected {len(items)} items")

    dialog = open_array_dialog(win, mode="point_rotation")
    time.sleep(0.5)

    run_on_main_thread(lambda: _configure_point_rotation_dialog(dialog))
    time.sleep(0.5)

    clear_window_subtitle(win)

    logger.info("Taking screenshot: point-rotation-array-dialog.png")
    take_window_screenshot(win, "main-array-point-rotation.png")

    time.sleep(0.25)

    def close_dialog():
        dialog.close()

    run_on_main_thread(close_dialog)
    app.quit_idle()


main()
