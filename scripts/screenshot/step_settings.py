#!/usr/bin/env python3
"""Screenshot: Step settings dialog."""

import os
import time
import logging
from rayforge.uiscript import app, win
from utils import (
    load_project,
    open_step_settings,
    find_step_by_type,
    take_screenshot,
    run_on_main_thread,
    set_window_size,
)

logger = logging.getLogger(__name__)

TARGET = os.environ.get("TARGET", "step-settings:contour:general")

ENGRAVE_MODES = {
    "constant_power": "CONSTANT_POWER",
    "dither": "DITHER",
    "multi_pass": "MULTI_PASS",
    "variable": "POWER_MODULATION",
}


def parse_target(target: str) -> tuple[str, str, str | None]:
    """Parse target into (step_type, page, mode)."""
    parts = target.split(":")
    step_type = parts[1] if len(parts) > 1 else "contour"
    tab = parts[2] if len(parts) > 2 else "general"
    mode = parts[3] if len(parts) > 3 else None

    page = "post-processing" if tab == "post" else "step-settings"
    return step_type, page, mode


def set_engrave_mode(dialog, mode_name: str):
    """Set the engrave mode in the dialog's engraver widget."""
    from rayforge.pipeline.producer.raster import DepthMode

    mode_enum = DepthMode[mode_name]
    mode_index = list(DepthMode).index(mode_enum)

    general_view = dialog.general_view
    logger.info("Searching for mode_row in general_view children...")

    def find_mode_row(widget, depth=0):
        indent = "  " * depth
        logger.info(f"{indent}Checking: {type(widget).__name__}")
        if hasattr(widget, "mode_row") and widget.mode_row is not None:
            return widget.mode_row
        child = widget.get_first_child()
        while child:
            result = find_mode_row(child, depth + 1)
            if result is not None:
                return result
            child = child.get_next_sibling()
        return None

    mode_row = find_mode_row(general_view)
    if mode_row is not None:
        mode_row.set_selected(mode_index)
        logger.info(f"Set engrave mode to: {mode_name} (index {mode_index})")
        return True

    logger.warning("Could not find engraver widget with mode_row")
    return False


def main():
    set_window_size(win, 2400, 1650)

    load_project(win, "allsteps.ryp")
    time.sleep(0.25)

    step_type, page, mode = parse_target(TARGET)
    logger.info(
        f"Target: {TARGET} -> type={step_type}, page={page}, mode={mode}"
    )

    found = find_step_by_type(win, step_type)
    if found is None:
        logger.error(f"No step found with type: {step_type}")
        app.quit_idle()
        return

    step, step_index = found
    if step is None:
        logger.error(f"No step found with type: {step_type}")
        app.quit_idle()
        return

    logger.info(f"Found step: {step.name} at index {step_index}")

    dialog = open_step_settings(win, step_index=step_index, page=page)
    time.sleep(0.5)

    if mode and step_type == "engrave":
        mode_name = ENGRAVE_MODES.get(mode)
        if mode_name:
            run_on_main_thread(lambda m=mode_name: set_engrave_mode(dialog, m))
            time.sleep(0.5)

    output_name = f"{TARGET.replace(':', '-')}.png"

    take_screenshot(output_name)
    time.sleep(0.25)
    app.quit_idle()


main()
