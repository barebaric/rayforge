"""Screenshot: Contour post-processing page with Perforation expanded."""

import logging
import time

from utils import (
    find_step_by_type,
    get_target,
    load_project,
    open_step_settings,
    restore_config,
    run_on_main_thread,
    set_window_size,
    take_screenshot,
)

from rayforge.uiscript import app, win

logger = logging.getLogger(__name__)

TARGET = get_target("step-settings:contour:post:perforation")

TRANSFORMER_NAME = "PerforationTransformer"


def stage_perforation_group(dialog) -> None:
    """Enable the Perforation transformer and expand its settings."""
    page = dialog.post_processing_view
    group = page.get_transformer_group(TRANSFORMER_NAME)
    group.enable_switch.set_active(True)
    page.get_expander_row(TRANSFORMER_NAME).set_expanded(True)


@restore_config
def main():
    set_window_size(win, 2400, 1650)

    load_project(win, "allsteps.ryp")
    time.sleep(0.25)

    step_type, page, _mode = "contour", "post-processing", None
    logger.info(f"Target: {TARGET} -> type={step_type}, page={page}")

    found = find_step_by_type(win, step_type)
    if found is None or found[0] is None:
        raise RuntimeError(f"No step found with type: {step_type}")

    step, step_index = found
    logger.info(f"Found step: {step.name} at index {step_index}")

    dialog = open_step_settings(win, step_index=step_index, page=page)
    time.sleep(0.5)

    run_on_main_thread(lambda: stage_perforation_group(dialog))
    time.sleep(0.5)

    if not take_screenshot():
        raise RuntimeError("Failed to capture the screenshot")
    time.sleep(0.25)
    app.quit_idle()


main()
