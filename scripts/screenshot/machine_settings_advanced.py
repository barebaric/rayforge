#!/usr/bin/env python3
"""Screenshot: Machine settings - Advanced page."""

import time
import logging
from rayforge.uiscript import app, win
from utils import open_machine_settings, take_screenshot

logger = logging.getLogger(__name__)
PAGE = "advanced"


def main():
    time.sleep(0.25)
    open_machine_settings(win, PAGE)
    time.sleep(0.25)
    take_screenshot(f"machine-{PAGE}.png")
    time.sleep(0.25)
    app.quit_idle()


main()
