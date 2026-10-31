"""Screenshot: App settings - General page."""

import logging
import time

from gi.repository import Gtk
from utils import (
    open_app_settings,
    restore_config,
    run_on_main_thread,
    take_screenshot,
)

from rayforge.uiscript import app, win

logger = logging.getLogger(__name__)
PAGE = "general"


def _iter_descendants(widget):
    """Yield all descendants of a widget, depth-first."""
    child = widget.get_first_child()
    while child:
        yield child
        yield from _iter_descendants(child)
        child = child.get_next_sibling()


def _scroll_general_page_to_bottom(dialog):
    """Show the behavior settings at the bottom of the general page."""

    def _scroll():
        page = dialog.content_stack.get_visible_child()
        for widget in _iter_descendants(page):
            if isinstance(widget, Gtk.ScrolledWindow):
                adjustment = widget.get_vadjustment()
                adjustment.set_value(adjustment.get_upper())
                return

    run_on_main_thread(_scroll)


@restore_config
def main():
    time.sleep(0.25)
    dialog = open_app_settings(win, PAGE)
    time.sleep(0.25)
    _scroll_general_page_to_bottom(dialog)
    time.sleep(0.75)
    take_screenshot()
    time.sleep(0.25)
    app.quit_idle()


main()
