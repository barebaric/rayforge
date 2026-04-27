#!/usr/bin/env python3
"""Screenshot: Projector Mode addon window."""

import logging
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

MOUSE_SVG = (
    Path(__file__).parent.parent.parent
    / "tests"
    / "image"
    / "svg"
    / "mouse.svg"
)


def show_projector():
    action = win.lookup_action("toggle_projector_mode")
    if action is None:
        raise RuntimeError("toggle_projector_mode action not found")
    from gi.repository import GLib

    action.change_state(GLib.Variant.new_boolean(True))

    from gi.repository import Gtk

    for toplevel in Gtk.Window.list_toplevels():
        if toplevel != win and toplevel.is_visible():
            return toplevel
    raise RuntimeError("Projector window not found")


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

    projector_win = run_on_main_thread(show_projector)
    time.sleep(1.0)

    take_screenshot("addon-projector-mode.png")

    time.sleep(0.25)
    run_on_main_thread(projector_win.close)
    app.quit_idle()


main()
