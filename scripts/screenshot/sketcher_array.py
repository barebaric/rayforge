"""
Screenshot: Sketcher array tools — circular array and array along
curve, each with its dialog and live preview.

Usage:
  pixi run screenshot addons:sketcher:array:circular
  pixi run screenshot addons:sketcher:array:curve-along
"""

import logging
import time

from rayforge_addons.sketcher.sketcher.core.commands import (
    AddItemsCommand,
    RemoveItemsCommand,
)
from rayforge_addons.sketcher.sketcher.core.entities import Line, Point
from rayforge_addons.sketcher.sketcher.ui_gtk import (
    get_sketch_mode_cmd,
    get_sketch_studio,
)
from utils import (
    clear_window_subtitle,
    get_target,
    load_project,
    restore_config,
    run_on_main_thread,
    set_window_size,
    take_region_screenshot,
    wait_for_settled,
)

from rayforge.core.workpiece import WorkPiece
from rayforge.uiscript import app, win

logger = logging.getLogger(__name__)

TARGET = get_target("addons:sketcher:array:circular")

CROP_MARGIN_PX = 60
CROP_REGIONS = {
    "circular": (930, 310, 1830, 1230),
    "curve-along": (960, 330, 2085, 1120),
}


def _enter_sketch_mode():
    """Opens the sketch editor for the first sketch-backed workpiece."""
    doc = win.doc_editor.doc
    for layer in doc.layers:
        for wp in layer.all_workpieces:
            if isinstance(wp, WorkPiece) and wp.geometry_provider_uid:
                get_sketch_mode_cmd().enter_sketch_mode(wp)
                return True
    return False


def _get_element():
    return get_sketch_studio().canvas.sketch_element


def _model_to_widget(element, x, y):
    canvas = element.canvas
    to_screen = (
        canvas.view_transform
        @ element.get_world_transform()
        @ element.content_transform
    )
    return to_screen.transform_point((x, y))


def _frame_view(
    bbox, fraction: float, pan_dx_mm: float = 0.0, pan_dy_mm: float = 0.0
):
    """Zooms and pans so the given model-space bounding box fills the
    canvas, then shifts the content to keep it clear of overlaid
    dialogs."""
    element = _get_element()
    canvas = element.canvas
    x0, y0, x1, y1 = bbox
    model_w = max(x1 - x0, 1e-6)
    model_h = max(y1 - y0, 1e-6)

    wx0, _ = _model_to_widget(element, x0, y0)
    wx1, _ = _model_to_widget(element, x1, y1)
    px_per_mm = max(abs(wx1 - wx0) / model_w, 1e-9)
    widget_w = canvas.get_width()
    widget_h = canvas.get_height()
    target_px_per_mm = min(
        fraction * widget_w / model_w, fraction * widget_h / model_h
    )
    zoom = canvas.zoom_level * target_px_per_mm / px_per_mm
    canvas.set_zoom(max(zoom, 0.01))

    cx = (x0 + x1) / 2
    cy = (y0 + y1) / 2
    px, py = _model_to_widget(element, cx, cy)
    scale_x, _ = canvas.view_transform.get_scale()
    new_pan_x = canvas.pan_x_mm - (widget_w / 2 - px) / scale_x
    new_pan_y = canvas.pan_y_mm + (widget_h / 2 - py) / scale_x
    canvas.set_pan(new_pan_x + pan_dx_mm, new_pan_y + pan_dy_mm)


def _clear_sketch(element):
    sketch = element.sketch
    registry = sketch.registry
    entities = list(registry.entities)
    points = [p for p in registry.points if p.id != sketch.origin_id]
    element.execute_command(
        RemoveItemsCommand(
            sketch, "Clear Sketch", points=points, entities=entities
        )
    )
    element.mark_dirty()


def _select_entities(element, entities):
    element.selection.clear()
    for entity in entities:
        element.selection.select_entity(entity, is_multi=True)
    element.mark_dirty()


def _stage_seed_rect(element, x, y):
    """Adds a small rectangle used as the array template."""
    points = [
        Point(-1, x - 8.0, y - 5.0),
        Point(-2, x + 8.0, y - 5.0),
        Point(-3, x + 8.0, y + 5.0),
        Point(-4, x - 8.0, y + 5.0),
    ]
    lines = [
        Line(-11, -1, -2),
        Line(-12, -2, -3),
        Line(-13, -3, -4),
        Line(-14, -4, -1),
    ]
    element.execute_command(
        AddItemsCommand(
            element.sketch, "Add Seed", points=points, entities=lines
        )
    )
    element.mark_dirty()
    return lines


def _stage_circular(element):
    seed_lines = _stage_seed_rect(element, 55.0, 0.0)
    _select_entities(element, seed_lines)
    element.set_tool("circular_array")


def _stage_curve_along(element):
    guide_start = Point(-1, -70.0, -20.0)
    guide_end = Point(-2, 70.0, 20.0)
    guide = Line(-10, -1, -2)
    element.execute_command(
        AddItemsCommand(
            element.sketch,
            "Add Guide",
            points=[guide_start, guide_end],
            entities=[guide],
        )
    )
    element.mark_dirty()
    seed_lines = _stage_seed_rect(element, 0.0, 0.0)
    _select_entities(element, [guide] + seed_lines)
    element.set_tool("curve_along_array")


def _region_key(target: str) -> str:
    if target.endswith(":circular"):
        return "circular"
    return "curve-along"


def _padded_region(target: str) -> tuple[int, int, int, int]:
    x0, y0, x1, y1 = CROP_REGIONS[_region_key(target)]
    return (
        x0 - CROP_MARGIN_PX,
        y0 - CROP_MARGIN_PX,
        x1 + CROP_MARGIN_PX,
        y1 + CROP_MARGIN_PX,
    )


@restore_config
def main():
    target = get_target("addons:sketcher:array:circular")
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

    run_on_main_thread(lambda: _stage_target_scene(target))
    time.sleep(1.0)
    if target.endswith(":circular"):
        run_on_main_thread(
            lambda: _frame_view(
                (-70.0, -70.0, 70.0, 70.0), 0.42, pan_dy_mm=22.0
            )
        )
    else:
        run_on_main_thread(
            lambda: _frame_view(
                (-85.0, -35.0, 85.0, 35.0),
                0.45,
                pan_dx_mm=-14.0,
                pan_dy_mm=14.0,
            )
        )
    time.sleep(0.75)
    clear_window_subtitle(win)
    take_region_screenshot(win, _padded_region(target))
    time.sleep(0.25)
    app.quit_idle()


def _stage_target_scene(target: str):
    element = _get_element()
    _clear_sketch(element)
    if target.endswith(":circular"):
        _stage_circular(element)
    elif target.endswith(":curve-along"):
        _stage_curve_along(element)
    else:
        logger.error(f"Unknown target: {target}")


main()
