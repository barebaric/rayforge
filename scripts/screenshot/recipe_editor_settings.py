"""Screenshot: Recipe editor - Laser, Step Settings, Post Processing pages."""

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

CONFIGS = {
    "laser": {
        "page": "settings",
        "step_type": "ContourStep",
        "settings_page": 0,
    },
    "step-settings": {
        "page": "settings",
        "step_type": "ContourStep",
        "settings_page": 1,
    },
    "post-processing": {
        "page": "post-processing",
        "step_type": "ContourStep",
    },
}


def main():
    target = get_target("recipe-editor:step-settings")
    _, subpage = target.split(":", 1)
    config = CONFIGS[subpage]
    time.sleep(0.25)
    open_recipe_editor(
        win,
        config["page"],
        step_type=config.get("step_type"),
        settings_page=config.get("settings_page", 0),
    )
    time.sleep(0.25)
    take_screenshot(target_to_filename(target))
    time.sleep(0.25)
    app.quit_idle()


main()
