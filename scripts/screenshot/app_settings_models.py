#!/usr/bin/env python3
"""Screenshot: App settings - Models page (library list + detail)."""

import os
import time
import logging
from rayforge.uiscript import app, win
from utils import open_app_settings, run_on_main_thread, take_screenshot

logger = logging.getLogger(__name__)

PAGE = "models"
TARGET = os.environ.get("TARGET", "")


def _activate_first_library(settings_win):
    def _do():
        page = settings_win.content_stack.get_visible_child()
        row = page._library_list.get_row_at_index(0)
        if row:
            page._library_list.select_row(row)
            page._on_library_activated(row)

    run_on_main_thread(_do)


def main():
    time.sleep(0.25)
    settings_win = open_app_settings(win, PAGE)
    time.sleep(0.25)

    if TARGET.endswith(":detail"):
        _activate_first_library(settings_win)
        time.sleep(0.25)
        take_screenshot("application-models-detail.png")
    else:
        take_screenshot("application-models-libraries.png")

    time.sleep(0.25)
    app.quit_idle()


main()
