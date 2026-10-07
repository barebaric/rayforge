"""Screenshot: Command step settings dialog."""

import logging
import time

from utils import (
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

TARGET = get_target("step-settings:command:general")

DEFAULT_COMMAND_TEXT = """G0 Z5 ; lift for safety
M8 ; air assist on
G4 P2.5 ; let the air clear"""


def _add_command_step() -> int:
    """Add a Command step to the first layer and return its global
    step index (the index space open_step_settings() resolves)."""
    from rayforge.core.step_registry import step_registry

    step_cls = step_registry.get("CommandStep")
    assert step_cls is not None, "CommandStep is not registered"
    layer = win.doc_editor.doc.layers[0]
    workflow = layer.workflow
    assert workflow is not None
    step = step_cls.create(win.doc_editor.context)
    step.command_text = DEFAULT_COMMAND_TEXT
    workflow.add_child(step)
    logger.info("Added Command step to layer %s", layer.name)
    return len(workflow.steps) - 1


@restore_config
def main():
    set_window_size(win, 2400, 1650)

    load_project(win, "allsteps.ryp")
    time.sleep(0.25)

    step_index = run_on_main_thread(_add_command_step)
    time.sleep(0.25)

    open_step_settings(win, step_index=step_index)
    time.sleep(0.5)

    take_screenshot()
    time.sleep(0.25)
    app.quit_idle()


main()
