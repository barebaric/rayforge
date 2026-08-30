"""
Screenshot: one image per sketcher constraint, applied to a small
scene, plus a conflicting-constraints shot for the documentation.

Usage:
  pixi run screenshot "addons:sketcher:constraint:*"
  pixi run screenshot addons:sketcher:conflicts
"""

import logging
import time

from PIL import Image
from rayforge_addons.sketcher.sketcher.core.commands import (
    AddItemsCommand,
    AngleConstraintCommand,
    RemoveItemsCommand,
)
from rayforge_addons.sketcher.sketcher.core.constraints import (
    AngleConstraint,
    AspectRatioConstraint,
    CoincidentConstraint,
    DiameterConstraint,
    DistanceConstraint,
    EqualLengthConstraint,
    HorizontalConstraint,
    PerpendicularConstraint,
    PointOnLineConstraint,
    RadiusConstraint,
    SymmetryConstraint,
    TangentConstraint,
    VerticalConstraint,
)
from rayforge_addons.sketcher.sketcher.core.entities import (
    Circle,
    Line,
    Point,
)
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

TARGET = get_target("addons:sketcher:constraint:coincident")

CROP_MARGIN_PX = 110
FRAME_FRACTION = 0.16


def _pt(id_, x, y):
    return Point(id_, x, y)


def _scene_coincident():
    points = [
        _pt(-1, 20.0, 105.0),
        _pt(-2, 60.0, 90.0),
        _pt(-3, 60.0, 90.0),
        _pt(-4, 100.0, 105.0),
    ]
    entities = [Line(-11, -1, -2), Line(-12, -3, -4)]
    constraints = [CoincidentConstraint(-2, -3)]
    return points, entities, constraints


def _scene_horizontal():
    points = [_pt(-1, 60.0, 80.0), _pt(-2, 140.0, 80.0)]
    entities = [Line(-11, -1, -2)]
    constraints = [HorizontalConstraint(-1, -2)]
    return points, entities, constraints


def _scene_vertical():
    points = [_pt(-1, 100.0, 30.0), _pt(-2, 100.0, 110.0)]
    entities = [Line(-11, -1, -2)]
    constraints = [VerticalConstraint(-1, -2)]
    return points, entities, constraints


def _scene_perpendicular():
    points = [
        _pt(-1, 30.0, 90.0),
        _pt(-2, 140.0, 90.0),
        _pt(-3, 75.0, 45.0),
        _pt(-4, 75.0, 90.0),
    ]
    entities = [Line(-11, -1, -2), Line(-12, -3, -4)]
    constraints = [PerpendicularConstraint(-11, -12)]
    return points, entities, constraints


def _scene_tangent():
    points = [
        _pt(-1, 45.0, 45.0),
        _pt(-2, 155.0, 45.0),
        _pt(-3, 100.0, 25.0),
        _pt(-4, 120.0, 25.0),
    ]
    entities = [Line(-11, -1, -2), Circle(-12, -3, -4)]
    constraints = [TangentConstraint(-11, -12)]
    return points, entities, constraints


def _scene_point_on_line():
    points = [
        _pt(-1, 45.0, 60.0),
        _pt(-2, 155.0, 90.0),
        _pt(-3, 75.0, 20.0),
        _pt(-4, 111.0, 78.0),
    ]
    entities = [Line(-11, -1, -2), Line(-12, -3, -4)]
    constraints = [PointOnLineConstraint(-4, -11)]
    return points, entities, constraints


def _scene_symmetry():
    points = [
        _pt(-1, 100.0, 25.0),
        _pt(-2, 100.0, 125.0),
        _pt(-3, 65.0, 75.0),
        _pt(-4, 135.0, 75.0),
    ]
    entities = [Line(-11, -1, -2), Line(-12, -3, -4)]
    constraints = [SymmetryConstraint(-3, -4, axis=-11)]
    return points, entities, constraints


def _scene_distance():
    points = [_pt(-1, 60.0, 80.0), _pt(-2, 140.0, 80.0)]
    entities = [Line(-11, -1, -2)]
    constraints = [DistanceConstraint(-1, -2, 80.0)]
    return points, entities, constraints


def _scene_diameter():
    points = [_pt(-1, 0.0, 0.0), _pt(-2, 22.0, 0.0)]
    entities = [Circle(-11, -1, -2)]
    constraints = [DiameterConstraint(-11, 44.0)]
    return points, entities, constraints


def _scene_radius():
    points = [_pt(-1, 0.0, 0.0), _pt(-2, 0.0, 22.0)]
    entities = [Circle(-11, -1, -2)]
    constraints = [RadiusConstraint(-11, 22.0)]
    return points, entities, constraints


def _scene_angle():
    points = [_pt(-1, 0.0, 0.0), _pt(-2, 45.0, 0.0), _pt(-3, 32.0, 32.0)]
    entities = [Line(-11, -1, -2), Line(-12, -1, -3)]
    constraints = None
    return points, entities, constraints


def _scene_aspect_ratio():
    points = [_pt(-1, -45.0, 0.0), _pt(-2, 0.0, 0.0), _pt(-3, 0.0, -22.0)]
    entities = [Line(-11, -1, -2), Line(-12, -2, -3)]
    ratio = 45.0 / 22.0
    constraints = [AspectRatioConstraint(-1, -2, -2, -3, ratio)]
    return points, entities, constraints


def _scene_equal_length():
    points = [
        _pt(-1, 60.0, 60.0),
        _pt(-2, 100.0, 60.0),
        _pt(-3, 60.0, 110.0),
        _pt(-4, 100.0, 110.0),
    ]
    entities = [Line(-11, -1, -2), Line(-12, -3, -4)]
    constraints = [EqualLengthConstraint([-11, -12])]
    return points, entities, constraints


def _scene_conflicts():
    points = [
        _pt(-1, 60.0, 60.0),
        _pt(-2, 100.0, 60.0),
        _pt(-3, 75.0, 85.0),
    ]
    entities = [Line(-11, -1, -2), Line(-12, -2, -3), Line(-13, -3, -1)]
    constraints = [
        DistanceConstraint(-1, -2, 40.0),
        DistanceConstraint(-2, -3, 20.0),
        DistanceConstraint(-3, -1, 70.0),
    ]
    return points, entities, constraints


SCENES = {
    "coincident": _scene_coincident,
    "horizontal": _scene_horizontal,
    "vertical": _scene_vertical,
    "perpendicular": _scene_perpendicular,
    "tangent": _scene_tangent,
    "point-on-line": _scene_point_on_line,
    "symmetry": _scene_symmetry,
    "distance": _scene_distance,
    "diameter": _scene_diameter,
    "radius": _scene_radius,
    "angle": _scene_angle,
    "aspect-ratio": _scene_aspect_ratio,
    "equal-length": _scene_equal_length,
    "conflicts": _scene_conflicts,
}


def _enter_sketch_mode():
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


def _get_scene_name():
    for name in SCENES:
        if TARGET.endswith(":" + name):
            return name
    raise ValueError(f"Unknown constraint target: {TARGET}")


def _stage_scene():
    element = _get_element()
    _clear_sketch(element)
    points, entities, constraints = SCENES[_get_scene_name()]()
    element.execute_command(
        AddItemsCommand(
            element.sketch,
            "Stage Scene",
            points=points,
            entities=entities,
            constraints=constraints,
        )
    )
    if TARGET.endswith(":angle"):
        registry = element.sketch.registry
        lines = [e for e in registry.entities if isinstance(e, Line)]
        params = AngleConstraintCommand.calculate_constraint_params(
            registry, lines[0].id, lines[1].id
        )
        element.execute_command(
            AddItemsCommand(
                element.sketch,
                "Stage Angle",
                constraints=[
                    AngleConstraint(
                        params.anchor_id,
                        params.other_id,
                        params.value_deg,
                        e1_far_idx=params.anchor_far_idx,
                        e2_far_idx=params.other_far_idx,
                    )
                ],
            )
        )
    element.selection.clear()
    element.set_tool("select")
    element.mark_dirty()


def _frame_view(fraction: float = FRAME_FRACTION):
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

    to_screen = (
        canvas.view_transform
        @ element.get_world_transform()
        @ element.content_transform
    )
    wx0, _ = to_screen.transform_point((x0, y0))
    wx1, _ = to_screen.transform_point((x1, y1))
    wy0 = to_screen.transform_point((x0, y0))[1]
    wy1 = to_screen.transform_point((x1, y1))[1]
    scales = [
        abs(wx1 - wx0) / model_w,
        abs(wy1 - wy0) / model_h,
    ]
    px_per_mm = max([s for s in scales if s > 1e-9] or [1e-9])
    widget_w = canvas.get_width()
    widget_h = canvas.get_height()
    target_px_per_mm = min(
        fraction * widget_w / model_w, fraction * widget_h / model_h
    )
    zoom = max(canvas.zoom_level * target_px_per_mm / px_per_mm, 0.01)
    canvas.set_zoom(zoom)

    to_screen = (
        canvas.view_transform
        @ element.get_world_transform()
        @ element.content_transform
    )
    px, py = to_screen.transform_point(((x0 + x1) / 2, (y0 + y1) / 2))
    scale_x, _ = canvas.view_transform.get_scale()
    new_pan_x = canvas.pan_x_mm - (widget_w / 2 - px) / scale_x
    new_pan_y = canvas.pan_y_mm + (widget_h / 2 - py) / scale_x
    canvas.set_pan(new_pan_x, new_pan_y)


def _capture_geometry():
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
    full_path = OUTPUT_DIR / target_to_filename(TARGET)
    if not take_window_screenshot(win):
        return False

    margin = CROP_MARGIN_PX
    ox, oy = canvas.translate_coordinates(win, 0, 0)
    x0, y0, x1, y1 = min(xs) + ox, min(ys) + oy, max(xs) + ox, max(ys) + oy
    x0 += DECOR_MARGIN_PX
    y0 += DECOR_MARGIN_PX + DECOR_OFFSET_Y_PX
    x1 += DECOR_MARGIN_PX
    y1 += DECOR_MARGIN_PX + DECOR_OFFSET_Y_PX

    img = Image.open(full_path)
    _save_webp_deterministic(
        img.crop(
            (
                max(x0 - margin, 0),
                max(y0 - margin, 0),
                min(x1 + margin, img.width),
                min(y1 + margin, img.height),
            )
        ),
        full_path,
    )
    logger.info(f"Geometry screenshot saved to {full_path}")
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

    if not run_on_main_thread(_enter_sketch_mode):
        logger.error("No sketch workpiece found in document")
        app.quit_idle()
        return
    time.sleep(0.75)

    run_on_main_thread(_stage_scene)
    time.sleep(0.5)
    run_on_main_thread(_frame_view)
    time.sleep(0.75)
    clear_window_subtitle(win)

    if TARGET.endswith(":conflicts"):
        take_window_screenshot(win)
    else:
        _capture_geometry()

    time.sleep(0.25)
    app.quit_idle()


main()
