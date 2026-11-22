"""
Screenshot: The Adjust Image dialog.

Usage: pixi run screenshot addon:image-editing:adjust
"""

import logging
import os
import sys
import tempfile
import time
from pathlib import Path

from PIL import Image
from utils import (
    get_target,
    restore_config,
    run_on_main_thread,
    set_window_size,
    take_screenshot,
    wait_for_settled,
)

from rayforge.uiscript import app, win

logger = logging.getLogger(__name__)

TARGET = get_target("addon:image-editing:adjust")

# Addons are loaded by path at runtime, not importable by name; the
# tests' root conftest puts their directories on sys.path likewise.
ADDON_DIR = (
    Path(__file__).parent.parent.parent
    / "rayforge"
    / "builtin_addons"
    / "rayforge-addon-image-editing"
)
sys.path.insert(0, str(ADDON_DIR))

EXAMPLE_IMAGE = (
    Path(__file__).parent.parent.parent
    / "website"
    / "static"
    / "images"
    / "image-edit-invert-before.webp"
)


def _import_example_image() -> Path:
    """Import the docs' example photo as a PNG, and return its path.

    The import reads the file from a background thread, so the path
    stays alive until the document has settled; the caller removes it.
    """
    fd, png_path = tempfile.mkstemp(prefix="rayforge-shot-", suffix=".png")
    os.close(fd)
    Image.open(EXAMPLE_IMAGE).save(png_path, format="PNG")
    win.doc_editor.file.load_file_from_path(Path(png_path), "image/png", None)
    return Path(png_path)


def _select_all_items():
    """Select all content items on the active layer."""
    win.surface.update_from_doc()
    layer = win.doc_editor.doc.active_layer
    items = layer.get_content_items()
    if items:
        win.surface.select_items(items)
    return items


def _stage_values(dialog):
    """Set the sliders to the values shown in the documentation."""
    dialog.brightness_scale.set_value(10)
    dialog.contrast_scale.set_value(25)
    dialog.sharpen_scale.set_value(50)


@restore_config
def main():
    set_window_size(win, 1400, 1000)

    png_path = run_on_main_thread(_import_example_image)
    if not wait_for_settled(win, timeout=10):
        logger.error("Document did not settle in time")
        app.quit_idle()
        return
    png_path.unlink(missing_ok=True)

    items = run_on_main_thread(_select_all_items)
    if not items:
        logger.error("No items found in document")
        app.quit_idle()
        return
    logger.info(f"Selected {len(items)} items")

    from image_editing.adjust_dialog import AdjustImageDialog

    def open_dialog():
        dialog = AdjustImageDialog(
            win,
            win.doc_editor.history_manager,
            win.surface.get_selected_workpieces(),
        )
        dialog.present()
        return dialog

    dialog = run_on_main_thread(open_dialog)
    time.sleep(0.5)

    run_on_main_thread(lambda: _stage_values(dialog))
    time.sleep(1.0)

    logger.info("Taking screenshot: %s", TARGET)
    take_screenshot()

    run_on_main_thread(dialog.close)
    app.quit_idle()


main()
