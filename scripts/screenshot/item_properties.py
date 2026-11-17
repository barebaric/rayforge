"""
Screenshot: the item properties panel (cropped to the right-hand
sidebar with the transform rows, including the position reference
selector).

Usage: pixi run screenshot main:item-properties
"""

import logging
import time

from gi.repository import Gtk
from PIL import Image
from utils import (
    DECOR_MARGIN_PX,
    DECOR_OFFSET_Y_PX,
    OUTPUT_DIR,
    _save_webp_deterministic,
    clear_window_subtitle,
    get_target,
    load_project,
    restore_config,
    restore_panel_states,
    run_on_main_thread,
    save_panel_states,
    set_window_size,
    show_panel,
    take_window_screenshot,
    target_to_filename,
    wait_for_settled,
)

from rayforge.core.workpiece import AnchorMode, WorkPiece
from rayforge.uiscript import app, win

logger = logging.getLogger(__name__)

MARGIN = 6
PANELS = ["toggle_bottom_panel"]


def _visible_widget(widget):
    """The ancestor whose allocation matches what is on screen.

    The properties list lives inside a scrolled window, so its own
    height can exceed the visible pane; the crop must use the
    viewport, not the content.
    """
    current = widget
    while current is not None:
        if isinstance(current, (Gtk.ScrolledWindow, Gtk.Viewport)):
            return current
        current = current.get_parent()
    return widget


@restore_config
def main():
    set_window_size(win, 1600, 1000)

    load_project(win, "bezier.ryp")
    logger.info("Waiting for document to settle...")
    if not wait_for_settled(win, timeout=30):
        logger.error("Document did not settle in time")
        app.quit_idle()
        return

    saved_states = save_panel_states(win, PANELS)
    show_panel(win, "toggle_bottom_panel", False)

    def _stage():
        doc = win.doc_editor.doc
        candidates = [
            w
            for layer in doc.layers
            for w in layer.all_workpieces
            if isinstance(w, WorkPiece) and w.geometry_provider_uid
        ]
        if not candidates:
            return False
        wp = candidates[0]
        wp.anchor_mode = AnchorMode.REFERENCE
        win.surface.select_items([wp])
        return True

    if not run_on_main_thread(_stage):
        logger.error("No sketch workpiece found in document")
        restore_panel_states(win, saved_states)
        app.quit_idle()
        return
    time.sleep(0.75)
    clear_window_subtitle(win)

    target = get_target("main:item-properties")
    full_path = OUTPUT_DIR / target_to_filename(target)
    if not take_window_screenshot(win):
        restore_panel_states(win, saved_states)
        app.quit_idle()
        return

    props = win.item_props_widget
    viewport = _visible_widget(props)
    px, py = props.translate_coordinates(win, 0, 0)
    vy = viewport.translate_coordinates(win, 0, 0)[1]
    x = px
    y = max(py, vy)
    bottom = min(py + props.get_height(), vy + viewport.get_height())
    w = props.get_width()
    h = bottom - y
    logger.info(f"Properties panel at window ({x}, {y}), size {w}x{h}")
    ox = DECOR_MARGIN_PX
    oy = DECOR_MARGIN_PX + DECOR_OFFSET_Y_PX
    img = Image.open(full_path)
    _save_webp_deterministic(
        img.crop(
            (
                max(x - MARGIN + ox, 0),
                max(y - MARGIN + oy, 0),
                min(x + w + MARGIN + ox, img.width),
                min(y + h + MARGIN + oy, img.height),
            )
        ),
        full_path,
    )
    logger.info(f"Properties panel screenshot saved to {full_path}")

    restore_panel_states(win, saved_states)
    time.sleep(0.25)
    app.quit_idle()


main()
