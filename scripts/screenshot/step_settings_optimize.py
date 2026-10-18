"""Screenshot: Contour post-processing page with Optimize expanded."""

import logging
import time

from gi.repository import Adw
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


def find_post_page(widget):
    """Find the post-processing page in the widget tree."""
    if hasattr(widget, "_group_dicts"):
        return widget
    child = widget.get_first_child()
    while child is not None:
        found = find_post_page(child)
        if found is not None:
            return found
        child = child.get_next_sibling()
    return None


def find_expander_row(widget, title: str):
    """Find the expander row with the given title in the widget tree."""
    if isinstance(widget, Adw.ExpanderRow) and widget.get_title() == title:
        return widget
    child = widget.get_first_child()
    while child is not None:
        found = find_expander_row(child, title)
        if found is not None:
            return found
        child = child.get_next_sibling()
    return None


def stage_optimize_group(dialog) -> bool:
    """Expand the Optimize group and turn on the cut planner."""

    def _stage() -> bool:
        page = find_post_page(dialog)
        if page is None:
            return False
        for group, t_dict in page._group_dicts.items():
            if t_dict.get("name") != TRANSFORMER_NAME:
                continue
            group.enable_switch.set_active(True)
            expander = find_expander_row(page, group.get_title())
            if expander is None:
                return False
            expander.set_expanded(True)
            group.best_start_row.set_active(True)
            group.prefer_corners_row.set_active(True)
            return True
        return False

    staged = run_on_main_thread(_stage)
    if not staged:
        logger.error(f"No {TRANSFORMER_NAME} found on the post page")
    return staged


@restore_config
def main():
    set_window_size(win, 2400, 1650)

    load_project(win, "allsteps.ryp")
    time.sleep(0.25)

    step_type, page, _mode = "contour", "post-processing", None
    logger.info(f"Target: {TARGET} -> type={step_type}, page={page}")

    found = find_step_by_type(win, step_type)
    if found is None or found[0] is None:
        logger.error(f"No step found with type: {step_type}")
        app.quit_idle()
        return

    step, step_index = found
    logger.info(f"Found step: {step.name} at index {step_index}")

    dialog = open_step_settings(win, step_index=step_index, page=page)
    time.sleep(0.5)

    if not stage_optimize_group(dialog):
        app.quit_idle()
        return
    time.sleep(0.5)

    take_screenshot()
    time.sleep(0.25)
    app.quit_idle()


main()
