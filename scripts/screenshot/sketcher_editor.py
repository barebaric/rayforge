"""
Screenshot: Sketcher editor visuals — the editor window, a constrained
sketch, the pie menu, and magnetic snapping.

Usage:
  pixi run screenshot addons:sketcher:editor
  pixi run screenshot addons:sketcher:constraints
  pixi run screenshot addons:sketcher:pie-menu
  pixi run screenshot addons:sketcher:snap
"""

import logging
import time

from PIL import Image
from rayforge_addons.sketcher.sketcher.core.commands import (
    AddItemsCommand,
    CreateOrEditConstraintCommand,
    EllipseCommand,
    RectangleCommand,
    RemoveItemsCommand,
)
from rayforge_addons.sketcher.sketcher.core.entities import Line, Point
from rayforge_addons.sketcher.sketcher.ui_gtk import (
    get_sketch_mode_cmd,
    get_sketch_studio,
)
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
    take_window_screenshot,
    target_to_filename,
    wait_for_settled,
)

from rayforge.core.workpiece import WorkPiece
from rayforge.uiscript import app, win

logger = logging.getLogger(__name__)

TARGET = get_target("addons:sketcher:editor")

RECT_START = (-50.0, -30.0)
RECT_END = (50.0, 30.0)
CIRCLE_CENTER = (85.0, 0.0)
CIRCLE_RADIUS = 14.0
SNAP_GUIDE_X = 100.0
SNAP_LINE_YS = (60.0, 85.0, 110.0)
PATH_START = (40.0, 135.0)
SNAP_TARGET = (SNAP_GUIDE_X, 135.0)


def _enter_sketch_mode():
    """Opens the sketch editor for the first sketch-backed workpiece."""
    doc = win.doc_editor.doc
    for layer in doc.layers:
        for wp in layer.all_workpieces:
            if isinstance(wp, WorkPiece) and wp.geometry_provider_uid:
                get_sketch_mode_cmd().enter_sketch_mode(wp)
                return True
    return False


def _get_canvas():
    return get_sketch_studio().canvas


def _get_element():
    return get_sketch_studio().canvas.sketch_element


def _model_to_world(element, x, y):
    to_world = element.get_world_transform() @ element.content_transform
    return to_world.transform_point((x, y))


def _model_to_widget(element, x, y):
    canvas = element.canvas
    to_screen = (
        canvas.view_transform
        @ element.get_world_transform()
        @ element.content_transform
    )
    return to_screen.transform_point((x, y))


def _get_geometry_bbox_model(element):
    sketch = element.sketch
    xs: list[float] = []
    ys: list[float] = []
    for entity in sketch.registry.entities:
        if entity.construction or entity.invisible:
            continue
        for x, y in entity.to_polyline(sketch.registry, 0.5):
            xs.append(x)
            ys.append(y)
    if not xs:
        return None
    return min(xs), min(ys), max(xs), max(ys)


def _frame_view(fraction: float = 0.5):
    """Zooms and pans so the visible geometry fills the canvas."""
    canvas = _get_canvas()
    element = _get_element()
    bbox = _get_geometry_bbox_model(element)
    if bbox is None:
        return
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
    canvas.set_pan(new_pan_x, new_pan_y)


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


def _draw_rectangle(element):
    registry = element.sketch.registry
    preview = RectangleCommand.start_preview(
        registry, RECT_START[0], RECT_START[1]
    )
    start_id = preview.start_id
    start_temp = preview.start_temp
    RectangleCommand.cleanup_preview(registry, preview)
    element.execute_command(
        RectangleCommand(
            element.sketch, start_id, RECT_END, is_start_temp=start_temp
        )
    )
    element.mark_dirty()


def _draw_circle(element):
    registry = element.sketch.registry
    dx = CIRCLE_RADIUS
    start = (CIRCLE_CENTER[0] - dx, CIRCLE_CENTER[1] - dx)
    end = (CIRCLE_CENTER[0] + dx, CIRCLE_CENTER[1] + dx)
    preview = EllipseCommand.start_preview(registry, start[0], start[1])
    start_id = preview.start_id
    start_temp = preview.start_temp
    EllipseCommand.cleanup_preview(registry, preview)
    element.execute_command(
        EllipseCommand(
            element.sketch,
            start_id,
            end,
            is_start_temp=start_temp,
            constrain_circle=True,
        )
    )
    element.mark_dirty()


def _get_horizontal_line_at(element, y):
    registry = element.sketch.registry
    for entity in registry.entities:
        if not isinstance(entity, Line):
            continue
        p1 = registry.get_point(entity.p1_idx)
        p2 = registry.get_point(entity.p2_idx)
        if abs(p1.y - p2.y) < 1e-6 and abs(p1.y - y) < 1e-6:
            return entity
    return None


def _get_vertical_line_at(element, x):
    registry = element.sketch.registry
    for entity in registry.entities:
        if not isinstance(entity, Line):
            continue
        p1 = registry.get_point(entity.p1_idx)
        p2 = registry.get_point(entity.p2_idx)
        if abs(p1.x - p2.x) < 1e-6 and abs(p1.x - x) < 1e-6:
            return entity
    return None


def _add_dimensions(element):
    for entity in (
        _get_horizontal_line_at(element, RECT_END[1]),
        _get_vertical_line_at(element, RECT_START[0]),
    ):
        if entity is not None:
            element.execute_command(
                CreateOrEditConstraintCommand(element.sketch, entity)
            )
    element.mark_dirty()


def _stage_scene():
    element = _get_element()
    _clear_sketch(element)
    _draw_rectangle(element)
    _draw_circle(element)
    _add_dimensions(element)
    element.selection.clear()
    element.set_tool("select")


def _capture_geometry(target: str, margin: int = 100) -> bool:
    """Screenshots the window and crops it to the sketch geometry with
    a comfortable margin, keeping the original zoom level."""
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
        logger.error("No geometry to crop to")
        return False
    full_path = OUTPUT_DIR / target_to_filename(target)
    if not take_window_screenshot(win):
        return False

    ox, oy = canvas.translate_coordinates(win, 0, 0)
    x0, y0, x1, y1 = min(xs) + ox, min(ys) + oy, max(xs) + ox, max(ys) + oy
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


def _capture_pie_menu() -> bool:
    """Screenshots the window and crops the fixed pie menu region."""
    full_path = OUTPUT_DIR / target_to_filename(TARGET)
    if not take_window_screenshot(win):
        return False
    img = Image.open(full_path)
    _save_webp_deterministic(img.crop((1550, 575, 1775, 800)), full_path)
    logger.info(f"Region screenshot saved to {full_path}")
    return True


def _open_pie_menu_at_bottom_line():
    """Selects the rectangle's dimensioned line and opens the pie menu
    on it."""
    element = _get_element()
    line = _get_horizontal_line_at(element, RECT_END[1])
    if line is None:
        logger.error("Rectangle line not found")
        return False
    element.selection.select_entity(line, is_multi=False)
    element.mark_dirty()

    pie_menu = element.editor.pie_menu
    pie_menu.set_context(element, line, "entity")
    wx, wy = _model_to_widget(element, 48.0, RECT_END[1])
    pie_menu.popup_at_location(wx, wy)
    return True


def _stage_snap_scene():
    """Replaces the scene with three parallel lines whose endpoints form
    an equidistant column, plus a fourth line collinear with the cursor,
    so the alignment guides and the equidistant snap indicator show when
    the path tool continues the pattern."""
    element = _get_element()
    _clear_sketch(element)
    length = 40.0
    points = []
    entities = []
    for i, y in enumerate(SNAP_LINE_YS):
        base = -(i * 2 + 1)
        points.append(Point(base, SNAP_GUIDE_X, y))
        points.append(Point(base - 1, SNAP_GUIDE_X + length, y))
        entities.append(Line(-(i * 2 + 11), base, base - 1))
    points.append(Point(-50, SNAP_GUIDE_X + 60.0, SNAP_TARGET[1]))
    points.append(Point(-51, SNAP_GUIDE_X + 100.0, SNAP_TARGET[1]))
    entities.append(Line(-60, -50, -51))
    element.execute_command(
        AddItemsCommand(
            element.sketch,
            "Stage Snap Scene",
            points=points,
            entities=entities,
        )
    )
    element.selection.clear()
    element.set_tool("select")
    element.mark_dirty()


def _stage_path_snap():
    """Starts a path collinear with the fourth line and hovers at the
    extension of the equidistant endpoint column, where both alignment
    guides and the equidistant snap indicator appear."""
    element = _get_element()
    canvas = element.canvas
    element.set_tool("path")
    tool = element.current_tool
    wx, wy = _model_to_world(element, PATH_START[0], PATH_START[1])
    tool.on_press(wx, wy, 1)
    tool.on_release(wx, wy)
    time.sleep(0.5)
    wx, wy = _model_to_world(element, SNAP_TARGET[0], SNAP_TARGET[1])
    tool.on_hover_motion(wx, wy)
    time.sleep(0.25)
    preview_ids = tool.get_preview_state().get_preview_point_ids()
    tool.query_snap_for_creation(
        element, SNAP_TARGET[0], SNAP_TARGET[1], exclude_points=preview_ids
    )
    canvas.queue_draw()
    return True


@restore_config
def main():
    target = get_target("addons:sketcher:editor")
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

    if target.endswith(":snap"):
        run_on_main_thread(_stage_snap_scene)
        time.sleep(0.5)
        run_on_main_thread(lambda: _frame_view(0.16))
        time.sleep(0.75)
        run_on_main_thread(_stage_path_snap)
        time.sleep(0.75)
        clear_window_subtitle(win)
        _capture_geometry(target, margin=60)
        time.sleep(0.25)
        app.quit_idle()
        return

    run_on_main_thread(_stage_scene)
    time.sleep(0.5)
    run_on_main_thread(_frame_view)
    time.sleep(0.75)
    clear_window_subtitle(win)

    if target.endswith(":pie-menu"):
        run_on_main_thread(_open_pie_menu_at_bottom_line)
        time.sleep(0.75)
        _capture_pie_menu()
    elif target.endswith(":constraints"):
        _capture_geometry(target)
    else:
        take_window_screenshot(win)

    time.sleep(0.25)
    app.quit_idle()


main()
