#!/usr/bin/env python3
"""Screenshot: Print & Cut addon wizard dialog."""

import importlib
import logging
import os
import time
from pathlib import Path

from rayforge.uiscript import app, win
from utils import (
    take_screenshot,
    run_on_main_thread,
    set_window_size,
    wait_for_settled,
)

logger = logging.getLogger(__name__)

TARGET = os.environ.get("TARGET", "print-and-cut:pick")

MOUSE_SVG = (
    Path(__file__).parent.parent.parent
    / "tests"
    / "image"
    / "svg"
    / "mouse.svg"
)

DESIGN_P1 = (0.145, 0.845)
DESIGN_P2 = (0.854, 0.175)
PHYSICAL_P1 = (10.0, 20.0)
PHYSICAL_P2 = (50.0, 60.0)


def setup_wizard():
    from rayforge.context import get_context

    wizard_mod = importlib.import_module(
        "rayforge_addons.print_and_cut.print_and_cut.wizard"
    )
    PrintAndCutWizard = wizard_mod.PrintAndCutWizard

    doc = win.doc_editor.doc
    wps = [wp for layer in doc.layers for wp in layer.all_workpieces]
    if not wps:
        raise RuntimeError("No workpiece found after import")
    item = wps[0]

    win.surface.select_items([item])

    ctx = get_context()
    machine = ctx.machine
    if not machine:
        raise RuntimeError("No machine configured")

    wizard = PrintAndCutWizard(
        parent=win,
        item=item,
        machine=machine,
        machine_cmd=win.machine_cmd,
        editor=win.doc_editor,
    )

    wizard._design_point1 = DESIGN_P1
    wizard._design_point2 = DESIGN_P2
    wizard._pick_surface.set_points(DESIGN_P1, DESIGN_P2)
    wizard._point1_row.set_subtitle("Point picked")
    wizard._point2_row.set_subtitle("Point picked")
    wizard._pick_status_row.set_subtitle(
        "Both points selected. Click Next to continue."
    )

    return wizard


def show_pick_page(wizard):
    wizard._next_btn.set_sensitive(True)
    wizard.present()


def show_jog_page(wizard):
    wizard._right_stack.set_visible_child_name("jog")
    wizard._back_btn.set_visible(True)
    wizard._reset_btn.set_visible(False)
    wizard._update_jog_next_btn()
    wizard.present()


def show_apply_page(wizard):
    wizard._physical_point1 = PHYSICAL_P1
    wizard._physical_point2 = PHYSICAL_P2
    wizard._pos1_row.set_subtitle(
        f"({PHYSICAL_P1[0]:.2f}, {PHYSICAL_P1[1]:.2f})"
    )
    wizard._pos2_row.set_subtitle(
        f"({PHYSICAL_P2[0]:.2f}, {PHYSICAL_P2[1]:.2f})"
    )

    wizard._right_stack.set_visible_child_name("apply")
    wizard._back_btn.set_visible(True)
    wizard._reset_btn.set_visible(False)
    wizard._next_btn.set_visible(False)
    wizard._apply_btn.set_visible(True)
    wizard._apply_btn.set_sensitive(True)
    wizard._update_apply_preview()
    wizard.present()


def main():
    set_window_size(win, 1400, 1000)

    run_on_main_thread(
        lambda: win.doc_editor.file.load_file_from_path(
            MOUSE_SVG, "image/svg+xml", None
        )
    )
    if not wait_for_settled(win, timeout=15):
        logger.error("Document did not settle in time")
        app.quit_idle()
        return

    time.sleep(2.0)

    wizard = run_on_main_thread(setup_wizard)

    page = TARGET.split(":")[-1] if ":" in TARGET else "pick"

    if page == "pick":
        run_on_main_thread(lambda: show_pick_page(wizard))
        time.sleep(1.0)
        take_screenshot("addon-print-and-cut-pick.png")
    elif page == "jog":
        run_on_main_thread(lambda: show_jog_page(wizard))
        time.sleep(0.5)
        take_screenshot("addon-print-and-cut-jog.png")
    elif page == "apply":
        run_on_main_thread(lambda: show_apply_page(wizard))
        time.sleep(0.5)
        take_screenshot("addon-print-and-cut-apply.png")
    else:
        logger.error(f"Unknown page: {page}")
        app.quit_idle()
        return

    time.sleep(0.25)
    run_on_main_thread(wizard.close)
    app.quit_idle()


main()
