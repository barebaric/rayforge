"""Screenshot: Recipe editor - Step Settings page for a laser step."""

import logging
import time

from utils import (
    get_target,
    open_recipe_editor,
    take_screenshot,
    target_to_filename,
)

from rayforge.uiscript import app, win

logger = logging.getLogger(__name__)
PAGE = "step-settings"


def main():
    target = get_target(f"recipe-editor:{PAGE}")
    time.sleep(0.25)
    open_recipe_editor(
        win,
        "settings",
        step_type="ContourStep",
        settings_page=1,
    )
    time.sleep(0.25)
    take_screenshot(target_to_filename(target))
    time.sleep(0.25)
    app.quit_idle()


main()
