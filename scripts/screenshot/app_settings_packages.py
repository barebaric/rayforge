#!/usr/bin/env python3
"""Screenshot: App settings - Packages page."""

import time
import logging
from rayforge.uiscript import app, win
from utils import open_app_settings, take_screenshot

logger = logging.getLogger(__name__)
PAGE = "packages"


def main():
    time.sleep(0.25)
    open_app_settings(win, PAGE)
    time.sleep(0.25)
    take_screenshot(f"application-{PAGE}.png")
    time.sleep(0.25)
    app.quit_idle()


main()
