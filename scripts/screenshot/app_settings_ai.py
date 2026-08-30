"""Screenshot: App settings - AI page."""

import logging
import time

from utils import (
    open_app_settings,
    restore_config,
    take_screenshot,
)

from rayforge.uiscript import app, win

logger = logging.getLogger(__name__)
PAGE = "ai"


@restore_config
def main():
    time.sleep(0.25)
    open_app_settings(win, PAGE)
    time.sleep(0.25)
    take_screenshot()
    time.sleep(0.25)
    app.quit_idle()


main()
