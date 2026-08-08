"""Screenshot: Machine settings - Rotary Module page."""

import logging
import time

from utils import (
    get_target,
    open_machine_settings,
    take_screenshot,
    target_to_filename,
)

from rayforge.uiscript import app, win

logger = logging.getLogger(__name__)
PAGE = "rotary-module"


def main():
    target = get_target(f"machine-settings:{PAGE}")
    time.sleep(0.25)
    open_machine_settings(win, PAGE)
    time.sleep(0.25)
    take_screenshot(target_to_filename(target))
    time.sleep(0.25)
    app.quit_idle()


main()
