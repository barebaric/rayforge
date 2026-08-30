"""
Screenshot: Sketcher drawing and modification tools — path, arc and
ellipse, rectangle, chamfer and fillet, fill, construction grid, and
text boxes (including the font properties panel).

Usage:
  pixi run screenshot addons:sketcher:tool:path
  pixi run screenshot addons:sketcher:tool:arc-ellipse
  pixi run screenshot addons:sketcher:tool:rectangle
  pixi run screenshot addons:sketcher:tool:chamfer-fillet
  pixi run screenshot addons:sketcher:tool:fill
  pixi run screenshot addons:sketcher:tool:text-box
  pixi run screenshot addons:sketcher:tool:text-box:font-properties
  pixi run screenshot addons:sketcher:tool:grid
"""

import logging
import math
import time

from PIL import Image
from rayforge_addons.sketcher.sketcher.core.commands import (
    AddFillCommand,
    AddItemsCommand,
    ArcCommand,
    BezierCommand,
    ChamferCommand,
    EllipseCommand,
    FilletCommand,
    GridCommand,
    ModifyTextPropertyCommand,
    RectangleCommand,
    RemoveItemsCommand,
    RoundedRectCommand,
    TextBoxCommand,
)
from rayforge_addons.sketcher.sketcher.core.entities import (
    Bezier,
    Line,
    Point,
)
from rayforge_addons.sketcher.sketcher.ui_gtk import (
    get_sketch_mode_cmd,
    get_sketch_studio,
)
from raygeo.geo.shape.text import FontConfig
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

TARGET = get_target("addons:sketcher:tool:path")

MARGIN_PX = 100


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


def _model_to_widget(element, x, y):
    canvas = element.canvas
    to_screen = (
        canvas.view_transform
        @ element.get_world_transform()
        @ element.content_transform
    )
    return to_screen.transform_point((x, y))


def _frame_view(fraction: float = 0.5):
    """Zooms and pans so the visible geometry fills the canvas."""
    element = _get_element()
    bbox = _get_bbox_model(element)
    if bbox is None:
        return
    x0, y0, x1, y1 = bbox
    model_w = max(x1 - x0, 1e-6)
    model_h = max(y1 - y0, 1e-6)

    canvas = element.canvas
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


def _get_bbox_model(element, include_construction: bool = False):
    sketch = element.sketch
    xs: list[float] = []
    ys: list[float] = []
    for entity in sketch.registry.entities:
        if entity.invisible or (
            entity.construction and not include_construction
        ):
            continue
        # to_polyline() only returns the first polygon, which misses
        # every glyph of multi-glyph text boxes, so iterate all of them.
        polygons = entity.to_geometry(sketch.registry).to_polygons(0.5)
        for polygon in polygons:
            for x, y in polygon:
                xs.append(x)
                ys.append(y)
    if not xs:
        return None
    return min(xs), min(ys), max(xs), max(ys)


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


def _capture_geometry(
    target: str, margin: int = MARGIN_PX, include_construction: bool = False
) -> bool:
    """Screenshots the window and crops it to the sketch geometry with
    a comfortable margin, keeping the original zoom level."""
    element = _get_element()
    bbox = _get_bbox_model(element, include_construction)
    if bbox is None:
        logger.error("No geometry to crop to")
        return False
    full_path = OUTPUT_DIR / target_to_filename(target)
    if not take_window_screenshot(win):
        return False

    canvas = element.canvas
    ox, oy = canvas.translate_coordinates(win, 0, 0)
    wx0, wy0 = _model_to_widget(element, *bbox[:2])
    wx1, wy1 = _model_to_widget(element, *bbox[2:])
    x0 = min(wx0, wx1) + ox + DECOR_MARGIN_PX
    y0 = min(wy0, wy1) + oy + DECOR_MARGIN_PX + DECOR_OFFSET_Y_PX
    x1 = max(wx0, wx1) + ox + DECOR_MARGIN_PX
    y1 = max(wy0, wy1) + oy + DECOR_MARGIN_PX + DECOR_OFFSET_Y_PX
    left = max(x0 - margin, 0)
    top = max(y0 - margin, 0)

    img = Image.open(full_path)
    _save_webp_deterministic(
        img.crop((left, top, x1 + margin, y1 + margin)), full_path
    )
    logger.info(f"Geometry screenshot saved to {full_path}")
    return True


def _preview_start(command_cls, registry, x, y):
    preview = command_cls.start_preview(registry, x, y)
    state = (preview.start_id, preview.start_temp)
    command_cls.cleanup_preview(registry, preview)
    return state


def _build_path_geometry(element):
    """A path of two lines joined by a bezier segment. Returns
    (waypoint_point, bezier_entity)."""
    p1 = Point(-1, 0.0, 0.0)
    p2 = Point(-2, 60.0, 0.0)
    p3 = Point(-3, 110.0, 40.0)
    p4 = Point(-4, 170.0, 40.0)
    entities = [Line(-11, -1, -2), Line(-12, -3, -4)]
    element.execute_command(
        AddItemsCommand(
            element.sketch,
            "Stage Path",
            points=[p1, p2, p3, p4],
            entities=entities,
        )
    )
    element.execute_command(
        BezierCommand(
            element.sketch,
            p2.id,
            (110.0, 40.0),
            end_pid=p3.id,
            is_line=False,
            cp1=(25.0, 0.0),
            cp2=(-25.0, 0.0),
        )
    )
    element.mark_dirty()
    registry = element.sketch.registry
    bezier = next(
        entity for entity in registry.entities if isinstance(entity, Bezier)
    )
    return p3, bezier


def _stage_path_scene(element):
    """The path scene with every entity selected, so the waypoint and
    handle editing affordances are visible."""
    registry = element.sketch.registry
    _build_path_geometry(element)
    for entity in registry.entities:
        element.selection.select_entity(entity, is_multi=True)
    element.set_tool("select")


def _stage_path_waypoint_scene(element):
    """The path scene, unselected, prepared for the pie menu shot."""
    _build_path_geometry(element)
    element.selection.clear()
    element.set_tool("select")


def _find_point_at(element, x, y):
    for point in element.sketch.registry.points:
        if abs(point.x - x) < 1e-6 and abs(point.y - y) < 1e-6:
            return point
    return None


def _select_path_waypoint(element):
    """Selects the bezier and its shared waypoint, the combination for
    which the waypoint and straighten tools are offered. Returns the
    widget-space position of the waypoint."""
    waypoint = _find_point_at(element, 110.0, 40.0)
    if waypoint is None:
        logger.error("Path waypoint not found")
        return None
    element.selection.select_entity(_get_path_bezier(element), is_multi=False)
    element.selection.select_point(waypoint.id, is_multi=True)
    element.set_tool("select")
    return _model_to_widget(element, waypoint.x, waypoint.y)


def _open_path_pie_menu(element, pos):
    """Opens the pie menu on the selected bezier at the waypoint."""
    if pos is None:
        return None
    pie_menu = element.editor.pie_menu
    pie_menu.set_context(element, _get_path_bezier(element), "entity")
    pie_menu.popup_at_location(*pos)
    return pos


def _get_path_bezier(element):
    return next(
        entity
        for entity in element.sketch.registry.entities
        if isinstance(entity, Bezier)
    )


def _capture_pie_menu_region(pos) -> bool:
    """Screenshots the window and crops a square region around the
    pie menu popup location."""
    full_path = OUTPUT_DIR / target_to_filename(TARGET)
    if not take_window_screenshot(win):
        return False
    ox, oy = _get_canvas().translate_coordinates(win, 0, 0)
    margin = 210
    cx = pos[0] + ox + DECOR_MARGIN_PX
    cy = pos[1] + oy + DECOR_MARGIN_PX + DECOR_OFFSET_Y_PX
    img = Image.open(full_path)
    _save_webp_deterministic(
        img.crop((cx - margin, cy - margin, cx + margin, cy + margin)),
        full_path,
    )
    logger.info(f"Pie menu screenshot saved to {full_path}")
    return True


def _stage_arc_ellipse_scene(element):
    registry = element.sketch.registry
    center_id, _ = _add_free_point(element, -70.0, 0.0, -100)
    start_id, _ = _add_free_point(element, -30.0, 0.0, -101)
    angle = math.radians(120.0)
    end_x = -70.0 + 40.0 * math.cos(angle)
    end_y = 40.0 * math.sin(angle)
    element.execute_command(
        ArcCommand(element.sketch, center_id, start_id, (end_x, end_y))
    )
    start_id, start_temp = _preview_start(
        EllipseCommand, registry, 30.0, -20.0
    )
    element.execute_command(
        EllipseCommand(
            element.sketch,
            start_id,
            (100.0, 20.0),
            is_start_temp=start_temp,
        )
    )
    element.mark_dirty()
    element.selection.clear()
    element.set_tool("select")


def _add_free_point(element, x, y, point_id):
    """Adds a single point with the given temp id and returns
    (real_point_id, is_temp)."""
    point = Point(point_id, x, y)
    element.execute_command(
        AddItemsCommand(
            element.sketch, "Add Point", points=[point], entities=[]
        )
    )
    return point.id, False


def _stage_rectangle_scene(element):
    registry = element.sketch.registry
    start_id, start_temp = _preview_start(
        RectangleCommand, registry, -110.0, -35.0
    )
    element.execute_command(
        RectangleCommand(
            element.sketch, start_id, (-30.0, 35.0), is_start_temp=start_temp
        )
    )
    start_id, start_temp = _preview_start(
        RoundedRectCommand, registry, 30.0, -35.0
    )
    element.execute_command(
        RoundedRectCommand(
            element.sketch,
            start_id,
            (110.0, 35.0),
            radius=14.0,
            is_start_temp=start_temp,
        )
    )
    element.mark_dirty()
    element.selection.clear()
    element.set_tool("select")


def _find_corner(element, x, y):
    """Finds the junction point nearest to (x, y) together with the two
    lines meeting there. Returns (corner_pid, line1_id, line2_id)."""
    registry = element.sketch.registry
    connection_count: dict[int, int] = {}
    for entity in registry.entities:
        if not isinstance(entity, Line):
            continue
        for pid in (entity.p1_idx, entity.p2_idx):
            connection_count[pid] = connection_count.get(pid, 0) + 1

    best = None
    for pid, count in connection_count.items():
        if count != 2:
            continue
        point = registry.get_point(pid)
        dist = math.hypot(point.x - x, point.y - y)
        if best is None or dist < best[0]:
            best = (dist, pid)
    if best is None:
        raise RuntimeError(f"No junction point near ({x}, {y})")
    corner_pid = best[1]
    lines = [
        entity
        for entity in registry.entities
        if isinstance(entity, Line)
        and corner_pid in (entity.p1_idx, entity.p2_idx)
    ]
    return corner_pid, lines[0].id, lines[1].id


def _stage_chamfer_fillet_scene(element):
    registry = element.sketch.registry
    start_id, start_temp = _preview_start(
        RectangleCommand, registry, -130.0, -30.0
    )
    element.execute_command(
        RectangleCommand(
            element.sketch, start_id, (-50.0, 30.0), is_start_temp=start_temp
        )
    )
    start_id, start_temp = _preview_start(
        RectangleCommand, registry, 50.0, -30.0
    )
    element.execute_command(
        RectangleCommand(
            element.sketch, start_id, (130.0, 30.0), is_start_temp=start_temp
        )
    )
    element.mark_dirty()

    corner_pid, line1_id, line2_id = _find_corner(element, -50.0, 30.0)
    element.execute_command(
        ChamferCommand(element.sketch, corner_pid, line1_id, line2_id, 12.0)
    )
    corner_pid, line1_id, line2_id = _find_corner(element, 50.0, 30.0)
    element.execute_command(
        FilletCommand(element.sketch, corner_pid, line1_id, line2_id, 13.0)
    )
    element.mark_dirty()
    element.selection.clear()
    element.set_tool("select")


def _stage_fill_scene(element):
    registry = element.sketch.registry
    start_id, start_temp = _preview_start(
        RectangleCommand, registry, -45.0, -30.0
    )
    rect_cmd = RectangleCommand(
        element.sketch, start_id, (45.0, 30.0), is_start_temp=start_temp
    )
    element.execute_command(rect_cmd)
    element.mark_dirty()
    loop = element.sketch.get_loop_at_point(0.0, 0.0)
    if not loop:
        logger.error("No closed loop found for fill")
        return False
    element.execute_command(AddFillCommand(element.sketch, loop))
    element.mark_dirty()
    element.selection.clear()
    element.set_tool("select")
    return True


def _stage_grid_scene(element):
    element.execute_command(GridCommand(element.sketch, rows=4, cols=6))
    element.mark_dirty()
    element.selection.clear()
    element.set_tool("select")


def _make_text_box(element, origin, content, font_config=None):
    """Creates a text box at *origin* with the given *content*, resizing
    the box to fit the text via ModifyTextPropertyCommand."""
    registry = element.sketch.registry
    cmd = TextBoxCommand(element.sketch, origin=origin)
    element.execute_command(cmd)
    if cmd.text_box_id is None:
        raise RuntimeError(f"Text box not created at {origin}")
    entity = registry.get_entity(cmd.text_box_id)
    if font_config is None:
        font_config = entity.font_config
    element.execute_command(
        ModifyTextPropertyCommand(
            element.sketch, cmd.text_box_id, content, font_config
        )
    )
    return cmd.text_box_id


def _stage_text_box_scene(element):
    """A large wordmark above a smaller part label, the typical use for
    engraved text."""
    _make_text_box(element, (-32.2, -25.0), "Rayforge", FontConfig(size=40.0))
    _make_text_box(element, (-14.7, -6.0), "Part # 001", FontConfig(size=16.0))
    element.mark_dirty()
    element.selection.clear()
    element.set_tool("select")
    return True


def _stage_text_box_properties_scene(element):
    """A single text box, selected so the Font Properties panel appears
    in the sidebar. The neighboring sidebar groups are hidden so the
    panel is the sole focus of the capture."""
    text_box_id = _make_text_box(
        element, (0.0, 0.0), "Rayforge", FontConfig(size=40.0)
    )
    element.mark_dirty()
    entity = element.sketch.registry.get_entity(text_box_id)
    element.selection.select_entity(entity, is_multi=False)
    element.set_tool("select")

    box = get_sketch_studio().font_properties.get_parent()
    for child in box:
        if child is not get_sketch_studio().font_properties:
            child.set_visible(False)
    return True


def _capture_font_properties_region(target: str) -> bool:
    """Screenshots the window and crops it to the Font Properties panel
    in the sidebar. The margins are per-side: tight left/top so the card
    keeps its rounded corners without grabbing the toolbar, a little room
    on the right and below."""
    studio = get_sketch_studio()
    widget = studio.font_properties
    full_path = OUTPUT_DIR / target_to_filename(target)
    if not take_window_screenshot(win):
        return False
    pos = widget.translate_coordinates(win, 0, 0)
    if pos is None:
        logger.error("Font properties panel is not visible")
        return False
    x, y = pos
    w, h = widget.get_width(), widget.get_height()
    logger.info(f"Font properties panel at window ({x}, {y}), size {w}x{h}")
    margin_left = 8
    margin_top = 8
    margin_right = 8
    margin_bottom = 16
    ox = DECOR_MARGIN_PX
    oy = DECOR_MARGIN_PX + DECOR_OFFSET_Y_PX
    img = Image.open(full_path)
    _save_webp_deterministic(
        img.crop(
            (
                max(x - margin_left + ox, 0),
                max(y - margin_top + oy, 0),
                x + w + margin_right + ox,
                y + h + margin_bottom + oy,
            )
        ),
        full_path,
    )
    logger.info(f"Font properties screenshot saved to {full_path}")
    return True


@restore_config
def main():
    target = get_target("addons:sketcher:tool:path")
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

    if target.endswith(":path-pie-menu"):
        run_on_main_thread(lambda: _stage_target_scene(target))
        time.sleep(0.5)
        run_on_main_thread(lambda: _frame_view(0.4))
        time.sleep(0.75)
        pos = run_on_main_thread(lambda: _select_path_waypoint(_get_element()))
        time.sleep(0.25)
        pos = run_on_main_thread(
            lambda: _open_path_pie_menu(_get_element(), pos)
        )
        if pos is None:
            logger.error("Failed to open pie menu")
            app.quit_idle()
            return
        time.sleep(0.75)
        clear_window_subtitle(win)
        _capture_pie_menu_region(pos)
        time.sleep(0.25)
        app.quit_idle()
        return

    result = run_on_main_thread(lambda: _stage_target_scene(target))
    if result is False:
        app.quit_idle()
        return
    time.sleep(0.5)
    run_on_main_thread(lambda: _frame_view(0.4))
    time.sleep(0.75)
    clear_window_subtitle(win)
    if target.endswith(":font-properties"):
        _capture_font_properties_region(target)
    else:
        _capture_geometry(
            target, include_construction=target.endswith(":grid")
        )
    time.sleep(0.25)
    app.quit_idle()


def _stage_target_scene(target: str):
    element = _get_element()
    _clear_sketch(element)
    if target.endswith(":path-pie-menu"):
        _stage_path_waypoint_scene(element)
    elif target.endswith(":path"):
        _stage_path_scene(element)
    elif target.endswith(":arc-ellipse"):
        _stage_arc_ellipse_scene(element)
    elif target.endswith(":rectangle"):
        _stage_rectangle_scene(element)
    elif target.endswith(":chamfer-fillet"):
        _stage_chamfer_fillet_scene(element)
    elif target.endswith(":fill"):
        return _stage_fill_scene(element)
    elif target.endswith(":text-box"):
        return _stage_text_box_scene(element)
    elif target.endswith(":font-properties"):
        return _stage_text_box_properties_scene(element)
    elif target.endswith(":grid"):
        _stage_grid_scene(element)
    else:
        logger.error(f"Unknown target: {target}")
        return False
    return True


main()
