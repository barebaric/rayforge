#!/usr/bin/env python3
"""Screenshot: Material test grid dialog."""

import logging
import time

from utils import (
    open_material_test,
    set_window_size,
    take_screenshot,
)

from rayforge.uiscript import app, win

logger = logging.getLogger(__name__)


def main():
    set_window_size(win, 2400, 1650)

    open_material_test(win)
    time.sleep(0.25)
    take_screenshot("material-test-grid.png")
    time.sleep(0.25)
    app.quit_idle()


main()
