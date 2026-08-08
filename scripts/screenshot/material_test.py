"""Screenshot: Material test grid dialog."""

import logging
import time

from utils import (
    get_target,
    open_material_test,
    set_window_size,
    take_screenshot,
    target_to_filename,
)

from rayforge.uiscript import app, win

logger = logging.getLogger(__name__)


def main():
    target = get_target("material-test")
    set_window_size(win, 2400, 1650)

    open_material_test(win)
    time.sleep(0.25)
    take_screenshot(target_to_filename(target))
    time.sleep(0.25)
    app.quit_idle()


main()
