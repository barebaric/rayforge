"""Screenshot: Workpieces tab of the step settings dialog."""

import logging
import time

from utils import (
    find_step_by_type,
    get_all_steps,
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

TARGET = get_target("step-settings:material-test-grid:workpieces")


def _create_material_test() -> None:
    """Create the grid through the editor's own command so the step is
    linked to its generated workpiece like in the real flow, and move
    another workpiece onto the grid's layer, so the tab shows one
    workpiece the grid's step leaves out."""
    doc = win.doc_editor.doc
    other = doc.layers[0].all_workpieces[0]
    win.doc_editor.calibration.create("material_test_grid")
    doc.active_layer.add_child(other)


@restore_config
def main():
    set_window_size(win, 2400, 1650)

    load_project(win, "allsteps.ryp")
    time.sleep(0.25)

    run_on_main_thread(_create_material_test)
    time.sleep(0.25)

    found = find_step_by_type(win, "material-test-grid")
    if found is None or found[0] is None:
        logger.error("Material test grid step not found")
        app.quit_idle()
        return

    step = found[0]
    step_index = get_all_steps(win).index(step)
    logger.info("Found step: %s at index %s", step.name, step_index)

    open_step_settings(win, step_index=step_index, page="workpieces")
    time.sleep(0.5)

    take_screenshot()
    time.sleep(0.25)
    app.quit_idle()


main()
