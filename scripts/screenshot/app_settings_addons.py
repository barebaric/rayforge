"""Screenshot: App settings - Packages page."""

import logging
import time

from utils import (
    get_target,
    open_app_settings,
    restore_config,
    take_screenshot,
    target_to_filename,
)

from rayforge.uiscript import app, win

logger = logging.getLogger(__name__)
PAGE = "addons"


@restore_config
def main():
    target = get_target(f"app-settings:{PAGE}")
    time.sleep(0.25)
    open_app_settings(win, PAGE)
    time.sleep(0.25)
    take_screenshot(target_to_filename(target))
    time.sleep(0.25)
    app.quit_idle()


main()
