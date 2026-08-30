"""Screenshot: Recipe editor - Applicability page."""

import logging
import time

from utils import (
    open_recipe_editor,
    restore_config,
    take_screenshot,
)

from rayforge.uiscript import app, win

logger = logging.getLogger(__name__)
PAGE = "applicability"


@restore_config
def main():
    time.sleep(0.25)
    open_recipe_editor(win, PAGE)
    time.sleep(0.25)
    take_screenshot()
    time.sleep(0.25)
    app.quit_idle()


main()
