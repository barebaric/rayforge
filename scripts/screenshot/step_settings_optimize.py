"""Screenshot: Contour post-processing page with Optimize expanded."""

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

TARGET = get_target("step-settings:contour:post:optimize")

TRANSFORMER_NAME = "Optimize"


def enable_cut_planner(step) -> None:
    """Turn on the cut planner switches of the step's Optimize."""
    dicts = [
        t_dict
        for t_dict in (
            step.per_step_transformers_dicts
            + step.per_workpiece_transformers_dicts
        )
        if t_dict.get("name") == TRANSFORMER_NAME
    ]
    if not dicts:
        raise RuntimeError(f"Step has no {TRANSFORMER_NAME} transformer")
    dicts[0]["best_start_point"] = True
    dicts[0]["prefer_corners"] = True


def expand_optimize_group(dialog) -> None:
    expander = dialog.post_processing_view.get_expander_row(TRANSFORMER_NAME)
    expander.set_expanded(True)


@restore_config
def main():
    set_window_size(win, 2400, 1650)

    load_project(win, "allsteps.ryp")
    time.sleep(0.25)

    logger.info(f"Target: {TARGET} -> contour, post-processing page")

    step, step_index = find_step_by_type(win, "contour")
    if step is None:
        raise RuntimeError("No contour step found in allsteps.ryp")
    logger.info(f"Found step: {step.name} at index {step_index}")

    enable_cut_planner(step)

    dialog = open_step_settings(
        win, step_index=step_index, page="post-processing"
    )
    time.sleep(0.5)

    run_on_main_thread(lambda: expand_optimize_group(dialog))
    time.sleep(0.5)

    if not take_screenshot():
        raise RuntimeError("Failed to capture the screenshot")
    time.sleep(0.25)
    app.quit_idle()


main()
