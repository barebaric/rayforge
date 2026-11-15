"""
Screenshot: CNC adaptive clearing step settings.

Usage: pixi run screenshot addon:cnc:adaptive-clearing
"""

import logging
import time

from utils import (
    load_project,
    open_step_settings,
    restore_config,
    run_on_main_thread,
    set_window_size,
    take_screenshot,
)

from rayforge.uiscript import app, win

logger = logging.getLogger(__name__)


def _setup_cnc_step() -> int:
    """Switch to a spindle machine and add an adaptive clearing step."""
    from rayforge.context import get_context
    from rayforge.core.step_registry import step_registry
    from rayforge.machine.models.machine import Machine
    from rayforge.machine.models.spindle import SpindleHead

    context = get_context()
    context.addon_mgr.enable_addon("cnc_essentials")
    machine = Machine(context)
    machine.set_axis_extents(200, 150)
    machine.max_cut_speed = 5000
    machine.max_travel_speed = 10000
    spindle = SpindleHead()
    spindle.name = "Spindle 1"
    machine.heads.clear()
    machine.add_head(spindle)
    context.machine_mgr.machines.clear()
    context.machine_mgr.add_machine(machine)
    context.config.set_machine(machine)

    step_cls = step_registry.get("AdaptiveClearStep")
    assert step_cls is not None, "AdaptiveClearStep is not registered"
    layer = win.doc_editor.doc.layers[0]
    workflow = layer.workflow
    assert workflow is not None
    step = step_cls.create(win.doc_editor.context)
    workflow.add_child(step)
    logger.info("Added adaptive clearing step to layer %s", layer.name)
    return len(workflow.steps) - 1


@restore_config
def main():
    set_window_size(win, 2400, 1650)

    load_project(win, "contour.ryp")
    time.sleep(0.25)

    step_index = run_on_main_thread(_setup_cnc_step)
    time.sleep(1.0)

    dialog = open_step_settings(win, step_index)
    run_on_main_thread(lambda: dialog.set_focus(None))
    time.sleep(1.0)

    logger.info("Taking screenshot: addon-cnc-adaptive-clearing.webp")
    take_screenshot()

    time.sleep(0.25)
    app.quit_idle()


main()
