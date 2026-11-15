"""
Screenshot: Adaptive clearing toolpath in the 3D canvas.

Usage: pixi run screenshot addon:cnc:adaptive-clearing
"""

import logging
import time
from pathlib import Path

from utils import (
    clear_window_subtitle,
    hide_panel,
    restore_config,
    restore_panel_states,
    run_on_main_thread,
    save_panel_states,
    seek_3d_playback,
    set_window_size,
    show_bottom_tab,
    show_panel,
    take_screenshot,
    wait_for_3d_rendered,
    wait_for_settled,
)

from rayforge.uiscript import app, win

logger = logging.getLogger(__name__)

PANELS = ["show_3d_view", "toggle_bottom_panel"]

POCKET_SVG = (
    Path(__file__).parent.parent.parent
    / "tests"
    / "image"
    / "svg"
    / "cnc-pocket.svg"
)


def _setup_machine():
    from rayforge.context import get_context
    from rayforge.machine.models.machine import Machine
    from rayforge.machine.models.spindle import SpindleHead

    context = get_context()
    context.addon_mgr.enable_addon("cnc_essentials")
    machine = Machine(context)
    machine.set_axis_extents(120, 90)
    machine.max_cut_speed = 5000
    machine.max_travel_speed = 10000
    spindle = SpindleHead()
    spindle.name = "Spindle 1"
    machine.heads.clear()
    machine.add_head(spindle)
    context.machine_mgr.machines.clear()
    context.machine_mgr.add_machine(machine)
    context.config.set_machine(machine)


def _add_step():
    from rayforge.core.step_registry import step_registry

    step_cls = step_registry.get("AdaptiveClearStep")
    assert step_cls is not None, "AdaptiveClearStep is not registered"
    layer = win.doc_editor.doc.layers[0]
    workflow = layer.workflow
    assert workflow is not None
    step = step_cls.create(win.doc_editor.context)
    step.tool_diameter = 6.0
    step.step_over = 2.0
    step.target_depth = -5.0
    step.safe_z = 2.0
    workflow.add_child(step)
    logger.info("Added adaptive clearing step to layer %s", layer.name)


@restore_config
def main():
    set_window_size(win, 2400, 1650)

    run_on_main_thread(_setup_machine)
    run_on_main_thread(
        lambda: win.doc_editor.file.load_file_from_path(
            POCKET_SVG, "image/svg+xml", None
        )
    )
    if not wait_for_settled(win, timeout=20):
        logger.error("Document did not settle after import")
        app.quit_idle()
        return

    run_on_main_thread(_add_step)
    if not wait_for_settled(win, timeout=30):
        logger.error("Pipeline did not settle after adding the step")
        app.quit_idle()
        return

    saved_states = save_panel_states(win, PANELS)

    show_panel(win, "show_3d_view", True)
    hide_panel(win, "toggle_bottom_panel")
    show_bottom_tab(win, "gcode")

    if not wait_for_settled(win, timeout=30):
        logger.error("Pipeline did not settle after 3D view activation")
        app.quit_idle()
        return
    if not wait_for_3d_rendered(win, timeout=15):
        logger.error("3D scene did not render in time")
        app.quit_idle()
        return

    seek_3d_playback(win, 0.8)
    clear_window_subtitle(win)

    logger.info("Taking screenshot: addon-cnc-adaptive-clearing.webp")
    take_screenshot()

    restore_panel_states(win, saved_states)
    time.sleep(0.25)
    app.quit_idle()


main()
