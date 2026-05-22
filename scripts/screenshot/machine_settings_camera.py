#!/usr/bin/env python3
"""Screenshot: Machine settings - Camera page."""

import logging
import time

from utils import open_machine_settings, take_screenshot

from rayforge.uiscript import app, win

logger = logging.getLogger(__name__)
PAGE = "camera"


def main():
    time.sleep(0.25)
    open_machine_settings(win, PAGE)
    time.sleep(0.25)
    take_screenshot(f"machine-{PAGE}.png")
    time.sleep(0.25)
    app.quit_idle()


main()
