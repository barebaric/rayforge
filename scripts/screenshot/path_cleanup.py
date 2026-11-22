"""
Screenshot: Close Paths tolerance dialog with a vector workpiece selected.

Usage: pixi run screenshot main:path-cleanup
"""

import logging
import time
from gettext import gettext as _

from utils import (
    clear_window_subtitle,
    load_project,
    restore_config,
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


def _open_close_paths_dialog():
    from rayforge.ui_gtk.doceditor.path_tolerance_dialog import (
        PathToleranceDialog,
    )

    def _apply(tolerance: float):
        pass

    dialog = PathToleranceDialog(
        heading=_("Close Paths"),
        body=_(
            "Closes open paths whose start and end point are closer "
            "than the tolerance."
        ),
        apply_label=_("Close Paths"),
        key="close-paths",
        on_apply=_apply,
    )
    dialog.present(win)
    return dialog


@restore_config
def main():
    set_window_size(win, 2000, 1400)

    load_project(win, "contour.ryp")
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

    dialog = run_on_main_thread(_open_close_paths_dialog)
    time.sleep(0.5)

    clear_window_subtitle(win)

    logger.info("Taking screenshot: main-path-cleanup.webp")
    take_window_screenshot(win)

    time.sleep(0.25)

    def close_dialog():
        dialog.close()

    run_on_main_thread(close_dialog)
    app.quit_idle()


main()
