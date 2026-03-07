#!/usr/bin/env python3
"""
Screenshot: AI Workpiece Generator dialog.

Usage: pixi run screenshot ai-workpiece-generator
"""

import time
import logging
from rayforge.uiscript import app, win
from utils import (
    wait_for_settled,
    take_screenshot,
    run_on_main_thread,
)

logger = logging.getLogger(__name__)

EXAMPLE_PROMPT = "A gear with 12 teeth, outer diameter 50mm, center hole 10mm"


def main():
    logger.info("Waiting for document to settle...")
    if not wait_for_settled(win, timeout=10):
        logger.error("Document did not settle in time")
        app.quit_idle()
        return

    def open_dialog():
        action = win.lookup_action("ai_generate_workpiece")
        if action:
            action.activate(None)
        else:
            logger.error("Action 'ai_generate_workpiece' not found")

    run_on_main_thread(open_dialog)
    time.sleep(0.5)

    def set_example_text():
        from gi.repository import Gtk

        for toplevel in Gtk.Window.list_toplevels():
            if toplevel != win and toplevel.is_visible():
                text_view = None

                def find_text_view(widget):
                    nonlocal text_view
                    if isinstance(widget, Gtk.TextView):
                        text_view = widget
                        return
                    if hasattr(widget, "get_first_child"):
                        child = widget.get_first_child()
                        while child:
                            find_text_view(child)
                            if text_view:
                                break
                            child = child.get_next_sibling()

                find_text_view(toplevel)
                if text_view:
                    buffer = text_view.get_buffer()
                    buffer.set_text(EXAMPLE_PROMPT)
                    return

    run_on_main_thread(set_example_text)
    time.sleep(0.5)

    logger.info("Taking screenshot: ai-workpiece-generator.png")
    take_screenshot("ai-workpiece-generator.png")

    app.quit_idle()


main()
