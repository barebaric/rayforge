"""
Screenshot: Sketcher offset tool — before, dialog, and after.

Usage:
  pixi run screenshot addons:sketcher:offset:before
  pixi run screenshot addons:sketcher:offset:dialog
  pixi run screenshot addons:sketcher:offset:after
"""

import logging
import time

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
    run_on_main_thread,
    set_window_size,
    take_screenshot,
    take_window_screenshot,
    target_to_filename,
    wait_for_settled,
)

from rayforge.uiscript import app, win

logger = logging.getLogger(__name__)

TARGET = get_target("addons:sketcher:offset:before")
OFFSET_DISTANCE = 8.0


def _enter_sketch_mode():
    """Opens the sketch editor for the first sketch-backed workpiece."""
    from rayforge_addons.sketcher.sketcher.ui_gtk import get_sketch_mode_cmd

    from rayforge.core.workpiece import WorkPiece

    doc = win.doc_editor.doc
    for layer in doc.layers:
        for wp in layer.all_workpieces:
            if isinstance(wp, WorkPiece) and wp.geometry_provider_uid:
                get_sketch_mode_cmd().enter_sketch_mode(wp)
                return True
    return False


def _get_sketch_element():
    from rayforge_addons.sketcher.sketcher.ui_gtk import get_sketch_studio

    return get_sketch_studio().canvas.sketch_element


def _get_canvas():
    from rayforge_addons.sketcher.sketcher.ui_gtk import get_sketch_studio

    return get_sketch_studio().canvas


def _select_all_entities():
    """Selects every visible entity of the sketch."""
    element = _get_sketch_element()
    sketch = element.sketch
    element.selection.clear()
    for entity in sketch.registry.entities:
        if entity.construction or entity.invisible:
            continue
        element.selection.select_entity(entity, is_multi=True)
    element.mark_dirty()
    return len(element.selection.entity_ids)


def _get_geometry_screen_rect():
    """Returns the on-screen (in-window) bounding rect of the visible
    sketch geometry, via the canvas's full model-to-screen chain."""
    canvas = _get_canvas()
    element = canvas.sketch_element
    sketch = element.sketch
    to_screen = (
        canvas.view_transform
        @ element.get_world_transform()
        @ element.content_transform
    )
    xs: list[float] = []
    ys: list[float] = []
    for entity in sketch.registry.entities:
        if entity.construction or entity.invisible:
            continue
        for x, y in entity.to_polyline(sketch.registry, 0.5):
            sx, sy = to_screen.transform_point(x, y)
            xs.append(sx)
            ys.append(sy)
    if not xs:
        return None
    ox, oy = canvas.translate_coordinates(win, 0, 0)
    return (
        min(xs) + ox,
        min(ys) + oy,
        max(xs) + ox,
        max(ys) + oy,
    )


def _capture_geometry(target: str) -> bool:
    """Screenshots the window and crops it to the sketch geometry with
    a comfortable margin, keeping the original zoom level."""
    rect = run_on_main_thread(_get_geometry_screen_rect)
    if rect is None:
        logger.error("No geometry to crop to")
        return False
    full_path = OUTPUT_DIR / target_to_filename(target)
    if not take_window_screenshot(win):
        return False

    margin = 100
    x0, y0, x1, y1 = rect
    x0 += DECOR_MARGIN_PX
    y0 += DECOR_MARGIN_PX + DECOR_OFFSET_Y_PX
    x1 += DECOR_MARGIN_PX
    y1 += DECOR_MARGIN_PX + DECOR_OFFSET_Y_PX
    left = max(x0 - margin, 0)
    top = max(y0 - margin, 0)

    img = Image.open(full_path)
    _save_webp_deterministic(
        img.crop((left, top, x1 + margin, y1 + margin)), full_path
    )
    logger.info(f"Geometry screenshot saved to {full_path}")
    return True


def _activate_offset_tool():
    """Opens the offset dialog with a live preview."""
    _get_sketch_element().set_tool("offset")


def _capture_dialog() -> bool:
    """Screenshots the offset MessageDialog window itself.

    The dialog is modal, so it is the active window in every
    display mode; take_screenshot captures it framed correctly on
    the desktop, on the composited Xvfb session and — via the
    content-crop fallback — on bare Xvfb without a WM.
    """
    return take_screenshot()


def _apply_offset():
    """Applies the offset command and selects the result."""
    from rayforge_addons.sketcher.sketcher.core.commands import OffsetCommand

    element = _get_sketch_element()
    sketch = element.sketch
    cmd = OffsetCommand(
        sketch, list(element.selection.entity_ids), OFFSET_DISTANCE
    )
    element.execute_command(cmd)
    element.selection.clear()
    element.set_tool("select")
    for entity in sketch.registry.entities:
        if entity.type == "polygon":
            element.selection.select_entity(entity, is_multi=True)
    element.mark_dirty()


@restore_config
def main():
    target = get_target("addons:sketcher:offset:before")
    set_window_size(win, 2400, 1650)

    load_project(win, "bezier.ryp")
    logger.info("Waiting for document to settle...")
    if not wait_for_settled(win, timeout=30):
        logger.error("Document did not settle in time")
        app.quit_idle()
        return

    if not run_on_main_thread(_enter_sketch_mode):
        logger.error("No sketch workpiece found in document")
        app.quit_idle()
        return
    time.sleep(0.75)

    count = run_on_main_thread(_select_all_entities)
    if not count:
        logger.error("No entities found in sketch")
        app.quit_idle()
        return
    logger.info(f"Selected {count} sketch entities")
    time.sleep(0.5)

    if target.endswith(":dialog"):
        run_on_main_thread(_activate_offset_tool)
        time.sleep(0.75)
        clear_window_subtitle(win)
        _capture_dialog()
    else:
        if target.endswith(":after"):
            run_on_main_thread(_apply_offset)
            time.sleep(0.75)
        clear_window_subtitle(win)
        _capture_geometry(target)

    time.sleep(0.25)
    app.quit_idle()


main()
