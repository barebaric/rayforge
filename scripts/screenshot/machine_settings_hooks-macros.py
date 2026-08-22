"""Screenshot: Machine settings - Hooks & Macros page."""

import logging
import time

from utils import (
    get_target,
    open_machine_settings,
    restore_config,
    take_screenshot,
    target_to_filename,
)

from rayforge.uiscript import app, win

logger = logging.getLogger(__name__)
PAGE = "hooks-macros"


@restore_config
def main():
    target = get_target(f"machine-settings:{PAGE}")
    time.sleep(0.25)
    open_machine_settings(win, PAGE)
    time.sleep(0.25)
    take_screenshot(target_to_filename(target))
    time.sleep(0.25)
    app.quit_idle()


main()
