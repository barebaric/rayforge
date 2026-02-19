#!/usr/bin/env python3
"""Screenshot: Material test grid dialog."""

import time
import logging
from rayforge.uiscript import app, win
from utils import (
    open_material_test,
    take_screenshot,
)

logger = logging.getLogger(__name__)


def main():
    win.set_default_size(2400, 1650)

    open_material_test(win)
    time.sleep(0.25)
    take_screenshot("material-test-grid.png")
    time.sleep(0.25)
    app.quit_idle()


main()
