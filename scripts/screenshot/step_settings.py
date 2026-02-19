#!/usr/bin/env python3
"""Screenshot: Step settings dialog."""

import os
import time
import logging
from rayforge.uiscript import app, win
from utils import (
    load_project,
    open_step_settings,
    get_step_by_index,
    find_step_by_type,
    take_screenshot,
)

logger = logging.getLogger(__name__)

STEP_TYPE = os.environ.get("STEP_TYPE", None)


def main():
    win.set_default_size(2400, 1650)

    load_project(win, "allsteps.ryp")
    time.sleep(0.25)

    if STEP_TYPE:
        found = find_step_by_type(win, STEP_TYPE)
        if found is None:
            logger.error(f"No step found with type: {STEP_TYPE}")
            app.quit_idle()
            return
        step, step_index = found
        if step is None:
            logger.error(f"No step found with type: {STEP_TYPE}")
            app.quit_idle()
            return
        step_type = STEP_TYPE
    else:
        step_index = 0
        step = get_step_by_index(win, 0)
        if step is None:
            logger.error("No step found at index 0")
            app.quit_idle()
            return
        step_type = step.typelabel.lower().replace(" ", "-")

    logger.info(f"Opening step settings for: {step.name}")

    open_step_settings(win, step_index=step_index, page="step-settings")
    time.sleep(0.25)

    take_screenshot(f"step-{step_type}-step-settings.png")
    time.sleep(0.25)
    app.quit_idle()


main()
