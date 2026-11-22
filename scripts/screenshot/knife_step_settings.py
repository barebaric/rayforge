"""
Screenshot: step settings for the drag knife and tangential knife steps.

Usage: pixi run screenshot step-settings:drag-knife:general
       pixi run screenshot step-settings:tangential-knife:general
"""

import logging
import time
from pathlib import Path

from utils import (
    clear_window_subtitle,
    get_target,
    open_step_settings,
    restore_config,
    run_on_main_thread,
    set_window_size,
    take_screenshot,
)

from rayforge.uiscript import app, win

logger = logging.getLogger(__name__)

POCKET_SVG = (
    Path(__file__).parent.parent.parent
    / "tests"
    / "image"
    / "svg"
    / "cnc-pocket.svg"
)

TARGETS = {
    "step-settings:drag-knife:general": (
        "DragKnifeCutStep",
        "DragKnifeHead",
        "Drag Knife Cut",
    ),
    "step-settings:tangential-knife:general": (
        "TangentialKnifeCutStep",
        "TangentialKnifeHead",
        "Tangential Knife Cut",
    ),
}


def _setup_machine(head_type: str):
    from rayforge.context import get_context
    from rayforge.machine.models import knife
    from rayforge.machine.models.machine import Machine

    context = get_context()
    machine = Machine(context)
    machine.set_axis_extents(120, 90)
    machine.max_cut_speed = 1000
    machine.max_travel_speed = 3000

    head_cls = getattr(knife, head_type)
    head = head_cls()
    machine.heads.clear()
    machine.add_head(head)
    context.machine_mgr.machines.clear()
    context.machine_mgr.add_machine(machine)
    context.config.set_machine(machine)
    logger.info(
        "Active machine %s with heads %s",
        context.machine.name if context.machine else None,
        [
            (head.name, head.machine_capability)
            for head in (context.machine.heads if context.machine else [])
        ],
    )


def _add_step(step_type: str, name: str) -> int:
    from rayforge.context import get_context
    from rayforge.core.step_registry import step_registry

    step_cls = step_registry.get(step_type)
    if step_cls is None:
        raise RuntimeError(f"{step_type} is not registered")
    step = step_cls.create(get_context())
    step.name = name
    logger.info(
        "Created %s with selected head %s",
        step_cls.__name__,
        getattr(step, "selected_head_uid", None),
    )

    layer = win.doc_editor.doc.layers[0]
    workflow = layer.workflow
    if workflow is None:
        raise RuntimeError("The imported document has no workflow")
    workflow.add_child(step)

    index = 0
    for doc_layer in win.doc_editor.doc.layers:
        if doc_layer.workflow is None:
            continue
        for candidate in doc_layer.workflow.steps:
            if candidate is step:
                return index
            index += 1
    raise RuntimeError("The knife step was not added to the document")


@restore_config
def main():
    set_window_size(win, 2400, 1650)

    target = get_target("step-settings:drag-knife:general")
    step_type, head_type, name = TARGETS[target]

    run_on_main_thread(lambda: _setup_machine(head_type))
    run_on_main_thread(
        lambda: win.doc_editor.file.load_file_from_path(
            POCKET_SVG, "image/svg+xml", None
        )
    )
    time.sleep(1.0)

    step_index = run_on_main_thread(lambda: _add_step(step_type, name))
    time.sleep(1.0)

    open_step_settings(win, step_index=step_index, page="step-settings")
    time.sleep(0.75)
    clear_window_subtitle(win)
    logger.info(f"Taking screenshot for target {target}")
    take_screenshot()
    time.sleep(0.25)
    app.quit_idle()


main()
