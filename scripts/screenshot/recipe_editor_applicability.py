#!/usr/bin/env python3
"""Screenshot: Recipe editor - Applicability page."""

import time
import logging
from rayforge.uiscript import app, win
from utils import open_recipe_editor, take_screenshot

logger = logging.getLogger(__name__)
PAGE = "applicability"


def main():
    time.sleep(0.25)
    open_recipe_editor(win, PAGE)
    time.sleep(0.25)
    take_screenshot(f"recipe-editor-{PAGE}.png")
    time.sleep(0.25)
    app.quit_idle()


main()
