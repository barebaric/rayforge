#!/usr/bin/env python3
"""Screenshot: Deepnest addon settings dialog."""

import logging
import time
from pathlib import Path

from rayforge.uiscript import app, win
from utils import (
    take_screenshot,
    run_on_main_thread,
    set_window_size,
    wait_for_settled,
)

logger = logging.getLogger(__name__)

HEARTS_PROJECT = (
    Path(__file__).parent.parent.parent / "tests" / "assets" / "pretty.ryp"
)

PRODUCT_ID = "A56heLPCXT6uPUpnmpnZYQ=="


def _ensure_addon_loaded():
    from rayforge.context import get_context

    ctx = get_context()
    if "deepnest" in ctx.addon_mgr.loaded_addons:
        return

    def _add_license():
        ctx.license_validator.add_gumroad_license(
            PRODUCT_ID, "TESTKEY-deepnest"
        )

    run_on_main_thread(_add_license)

    if "deepnest" not in ctx.addon_mgr.loaded_addons:
        raise RuntimeError("Failed to load deepnest addon")


def open_dialog():
    import importlib

    NestingSettingsDialog = importlib.import_module(
        "rayforge_addons.deepnest.deepnest.dialog"
    ).NestingSettingsDialog

    dialog = NestingSettingsDialog(win)
    dialog.present()
    return dialog


def main():
    _ensure_addon_loaded()

    set_window_size(win, 1400, 1000)

    def load():
        win.doc_editor.file.load_project_from_path(HEARTS_PROJECT)

    run_on_main_thread(load)
    if not wait_for_settled(win, timeout=15):
        logger.error("Document did not settle in time")
        app.quit_idle()
        return

    time.sleep(1.0)

    dialog = run_on_main_thread(open_dialog)
    time.sleep(0.5)

    take_screenshot("addon-deepnest.png")

    time.sleep(0.25)
    run_on_main_thread(dialog.close)
    app.quit_idle()


main()
