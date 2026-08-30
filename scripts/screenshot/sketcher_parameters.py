"""
Screenshot: Sketch parameterization — the in-editor parameter panel, an
expression-driven dimension constraint, and the main-window property panel
for assigning per-instance parameter values.

Usage:
  pixi run screenshot addons:sketcher:parameters:panel
  pixi run screenshot addons:sketcher:parameters:expression
  pixi run screenshot addons:sketcher:parameters
"""

import logging
import time

from PIL import Image
from rayforge_addons.sketcher.sketcher.core.commands import (
    AddItemsCommand,
    CreateOrEditConstraintCommand,
    RectangleCommand,
    RemoveItemsCommand,
)
from rayforge_addons.sketcher.sketcher.core.constraints import (
    DistanceConstraint,
)
from rayforge_addons.sketcher.sketcher.core.entities import Line
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

from rayforge.core.varset import FloatVar
from rayforge.core.workpiece import WorkPiece
from rayforge.uiscript import app, win

logger = logging.getLogger(__name__)

TARGET = get_target("addons:sketcher:parameters:panel")

RECT_START = (-50.0, -30.0)
RECT_END = (50.0, 30.0)


def _enter_sketch_mode():
    """Opens the sketch editor for the first sketch-backed workpiece."""
    doc = win.doc_editor.doc
    for layer in doc.layers:
        for wp in layer.all_workpieces:
            if isinstance(wp, WorkPiece) and wp.geometry_provider_uid:
                get_sketch_mode_cmd().enter_sketch_mode(wp)
                return wp
    return None


def _get_element():
    return get_sketch_studio().canvas.sketch_element


def _get_canvas():
    return get_sketch_studio().canvas


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


def _add_parameters(element):
    """Adds the width/thickness parameters to the sketch."""
    sketch = element.sketch
    sketch.input_parameters.add(
        FloatVar(
            key="width",
            label="Width",
            description="Overall width of the part",
            default=100.0,
        )
    )
    sketch.input_parameters.add(
        FloatVar(
            key="thickness",
            label="Wall thickness",
            description="Material wall thickness",
            default=5.0,
            min_val=0.1,
        )
    )
    get_sketch_studio().varset_editor.populate(sketch.input_parameters)
    element.mark_dirty()


def _expand_first_parameter_row():
    """Expands the first parameter row so its editor fields are visible."""
    editor = get_sketch_studio().varset_editor
    i = 0
    while row := editor.list_box.get_row_at_index(i):
        widget = row.get_child()
        if hasattr(widget, "set_expanded"):
            widget.set_expanded(True)
            break
        i += 1


def _capture_panel_region(target: str) -> bool:
    """Screenshots the window and crops it to the in-editor Sketch
    Parameters panel (the VarSet editor) in the sidebar."""
    widget = get_sketch_studio().varset_editor
    full_path = OUTPUT_DIR / target_to_filename(target)
    if not take_window_screenshot(win):
        return False
    pos = widget.translate_coordinates(win, 0, 0)
    if pos is None:
        logger.error("Parameter panel is not visible")
        return False
    x, y = pos
    w, h = widget.get_width(), widget.get_height()
    logger.info(f"Parameter panel at window ({x}, {y}), size {w}x{h}")
    margin = 6
    ox = DECOR_MARGIN_PX
    oy = DECOR_MARGIN_PX + DECOR_OFFSET_Y_PX
    img = Image.open(full_path)
    _save_webp_deterministic(
        img.crop(
            (
                max(x - margin + ox, 0),
                max(y - margin + oy, 0),
                min(x + w + margin + ox, img.width),
                min(y + h + margin + oy, img.height),
            )
        ),
        full_path,
    )
    logger.info(f"Parameter panel screenshot saved to {full_path}")
    return True


def _stage_expression_scene(element):
    """A rectangle whose width is driven by the 'width' parameter through
    an expression-driven dimension constraint (drawn in orange), plus a
    plain numeric dimension for contrast."""
    _clear_sketch(element)
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

    element.sketch.input_parameters.add(
        FloatVar(key="width", label="Width", default=100.0)
    )
    get_sketch_studio().varset_editor.populate(element.sketch.input_parameters)

    # Find the top edge (horizontal) of the rectangle for the plain
    # dimension, and the left edge (vertical) for the expression one.
    top = None
    left = None
    for entity in registry.entities:
        if not isinstance(entity, Line):
            continue
        p1 = registry.get_point(entity.p1_idx)
        p2 = registry.get_point(entity.p2_idx)
        if abs(p1.y - p2.y) < 1e-6 and abs(p1.y - RECT_END[1]) < 1e-6:
            top = entity
        if abs(p1.x - p2.x) < 1e-6 and abs(p1.x - RECT_START[0]) < 1e-6:
            left = entity
    if top is not None:
        element.execute_command(
            CreateOrEditConstraintCommand(element.sketch, top)
        )
    if left is not None:
        element.execute_command(
            AddItemsCommand(
                element.sketch,
                "Stage Expression Dimension",
                constraints=[
                    DistanceConstraint(
                        left.p1_idx,
                        left.p2_idx,
                        expression="width / 2",
                        value=25.0,
                    )
                ],
            )
        )
    element.selection.clear()
    element.set_tool("select")
    element.mark_dirty()


def _model_to_widget(element, x, y):
    canvas = element.canvas
    to_screen = (
        canvas.view_transform
        @ element.get_world_transform()
        @ element.content_transform
    )
    return to_screen.transform_point((x, y))


def _frame_view(fraction: float = 0.5):
    canvas = _get_canvas()
    element = _get_element()
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
        return
    x0, y0 = min(xs), min(ys)
    x1, y1 = max(xs), max(ys)
    model_w = max(x1 - x0, 1e-6)
    model_h = max(y1 - y0, 1e-6)
    widget_w = canvas.get_width()
    widget_h = canvas.get_height()
    target_px_per_mm = min(
        fraction * widget_w / model_w, fraction * widget_h / model_h
    )
    px_per_mm = 1.0
    wx0, _ = _model_to_widget(element, x0, y0)
    wx1, _ = _model_to_widget(element, x1, y1)
    px_per_mm = max(abs(wx1 - wx0) / model_w, 1e-9)
    zoom = canvas.zoom_level * target_px_per_mm / px_per_mm
    canvas.set_zoom(max(zoom, 0.01))
    cx = (x0 + x1) / 2
    cy = (y0 + y1) / 2
    px, py = _model_to_widget(element, cx, cy)
    scale_x, _ = canvas.view_transform.get_scale()
    new_pan_x = canvas.pan_x_mm - (widget_w / 2 - px) / scale_x
    new_pan_y = canvas.pan_y_mm + (widget_h / 2 - py) / scale_x
    canvas.set_pan(new_pan_x, new_pan_y)


def _capture_geometry(target: str, margin: int = 100) -> bool:
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
        img.crop(
            (
                left,
                top,
                min(x1 + margin, img.width),
                min(y1 + margin, img.height),
            )
        ),
        full_path,
    )
    logger.info(f"Geometry screenshot saved to {full_path}")
    return True


def _is_panel() -> bool:
    return TARGET.endswith(":panel")


def _is_expression() -> bool:
    return TARGET.endswith(":expression")


def _find_sketch_parameters_expander():
    """Returns the 'Sketch Parameters' Expander in the main window's
    property panel, or None."""
    widget = getattr(win, "item_props_widget", None)
    if widget is None:
        return None
    for provider, widgets, expander in getattr(widget, "_separate_groups", []):
        if getattr(provider, "group_title", "") == "Sketch Parameters":
            return expander
    return None


def _capture_mainwindow_property_region(target: str) -> bool:
    """Screenshots the window and crops it to the 'Sketch Parameters'
    group in the main window's right-hand property panel."""
    expander = _find_sketch_parameters_expander()
    full_path = OUTPUT_DIR / target_to_filename(target)
    if expander is None or not expander.get_visible():
        logger.error("Sketch Parameters panel is not visible")
        return False
    if not take_window_screenshot(win):
        return False
    pos = expander.translate_coordinates(win, 0, 0)
    if pos is None:
        logger.error("Could not locate Sketch Parameters panel")
        return False
    x, y = pos
    w, h = expander.get_width(), expander.get_height()
    logger.info(f"Sketch Parameters at window ({x}, {y}), size {w}x{h}")
    margin = 8
    ox = DECOR_MARGIN_PX
    oy = DECOR_MARGIN_PX + DECOR_OFFSET_Y_PX
    img = Image.open(full_path)
    _save_webp_deterministic(
        img.crop(
            (
                max(x - margin + ox, 0),
                max(y - margin + oy, 0),
                min(x + w + margin + ox, img.width),
                min(y + h + margin + oy, img.height),
            )
        ),
        full_path,
    )
    logger.info(f"Main window property screenshot saved to {full_path}")
    return True


@restore_config
def main():
    set_window_size(win, 2400, 1650)

    load_project(win, "bezier.ryp")
    logger.info("Waiting for document to settle...")
    if not wait_for_settled(win, timeout=30):
        logger.error("Document did not settle in time")
        app.quit_idle()
        return

    if not (_is_panel() or _is_expression()):
        # Main window property panel: add parameters to the sketch, select
        # the sketch workpiece so the right-hand panel offers the Sketch
        # Parameters group, and give the workpiece a non-default width.
        def _stage_mainwindow():
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
            sketch = wp.get_geometry_provider()
            if sketch is None or sketch.input_parameters is None:
                return False
            sketch.input_parameters.add(
                FloatVar(
                    key="width",
                    label="Width",
                    description="Overall width of the part",
                    default=100.0,
                )
            )
            sketch.input_parameters.add(
                FloatVar(
                    key="thickness",
                    label="Wall thickness",
                    description="Material wall thickness",
                    default=5.0,
                    min_val=0.1,
                )
            )
            params = dict(wp.geometry_provider_params)
            params["width"] = 120.0
            params["thickness"] = 6.0
            wp.geometry_provider_params = params
            win.surface.select_items([wp])
            return True

        if not run_on_main_thread(_stage_mainwindow):
            logger.error("Could not stage main window parameter scene")
            app.quit_idle()
            return
        time.sleep(0.75)
        clear_window_subtitle(win)
        _capture_mainwindow_property_region(TARGET)
        time.sleep(0.25)
        app.quit_idle()
        return

    if not run_on_main_thread(_enter_sketch_mode):
        logger.error("No sketch workpiece found in document")
        app.quit_idle()
        return
    time.sleep(0.75)

    if _is_expression():
        run_on_main_thread(lambda: _stage_expression_scene(_get_element()))
        time.sleep(0.5)
        run_on_main_thread(lambda: _frame_view(0.35))
        time.sleep(0.75)
        clear_window_subtitle(win)
        _capture_geometry(TARGET, margin=90)
        time.sleep(0.25)
        app.quit_idle()
        return

    if _is_panel():
        run_on_main_thread(lambda: _draw_rectangle(_get_element()))
        run_on_main_thread(lambda: _add_parameters(_get_element()))
        time.sleep(0.5)
        run_on_main_thread(_expand_first_parameter_row)
        time.sleep(0.5)
        clear_window_subtitle(win)
        _capture_panel_region(TARGET)
        time.sleep(0.25)
        app.quit_idle()
        return


main()
