#!/usr/bin/env python3
"""Screenshot: App settings - Machines page."""

import logging
import time

from utils import (
    get_target,
    open_app_settings,
    take_screenshot,
    target_to_filename,
)

from rayforge.uiscript import app, win

logger = logging.getLogger(__name__)
PAGE = "machines"


def main():
    target = get_target(f"app-settings:{PAGE}")
    time.sleep(0.25)
    open_app_settings(win, PAGE)
    time.sleep(0.25)
    take_screenshot(target_to_filename(target))
    time.sleep(0.25)
    app.quit_idle()


main()
