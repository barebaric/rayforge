"""
Screenshot: a calibration test right after Tools → Create …, with its
Test Settings in the item properties panel.

Usage: pixi run screenshot material-test
       pixi run screenshot interval-test
       pixi run screenshot focus-test
"""

import logging
import time

from utils import (
    clear_window_subtitle,
    create_calibration_test,
    get_target,
    restore_config,
    set_window_size,
    take_screenshot,
)

from rayforge.uiscript import app, win

logger = logging.getLogger(__name__)

ACTIONS = {
    "material-test": "material_test",
    "interval-test": "interval_test",
    "focus-test": "focus_test",
}


@restore_config
def main():
    set_window_size(win, 2400, 1650)
    target = get_target("material-test")
    create_calibration_test(win, ACTIONS[target])
    time.sleep(1.5)
    clear_window_subtitle(win)
    take_screenshot()
    time.sleep(0.25)
    app.quit_idle()


main()
