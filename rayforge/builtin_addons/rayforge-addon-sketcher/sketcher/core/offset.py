"""Offsetting of sketch geometry.

The selection is preprocessed into offsettable items (see
``contour.build_offset_items``): lone circles, arcs and ellipses
offset in place (their defining points move onto the offset ring),
everything else is replaced by the offset result — one polygon per
result outline, or one polygon per result solid for multi-ring
shapes. This module carries the offset plans and the outline frame
machinery that used to live on the entity classes.
"""

from __future__ import annotations

import itertools
import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

from raygeo.geo import Geometry
from raygeo.geo.algo.polylabel import find_largest_circle
from raygeo.geo.shape.arc import get_arc_samples, get_arc_sweep
from raygeo.geo.shape.ellipse import get_ellipse_samples
from raygeo.geo.shape.line import (
    does_line_segment_intersect_rect,
    get_line_segment_closest_point,
)
from raygeo.geo.shape.polygon import (
    EndStyle,
    JoinStyle,
    get_circle_polygon,
    get_polygon_area,
    get_polygon_closest_point,
    get_polygon_convex_hull,
    is_point_inside_polygon,
    offset_polyline,
)
from raygeo.geo.types import Rect
from raygeo.sketch.entities.polygon import Polygon

from .types import EntityID

_MIN_OFFSET_RADIUS = 1e-6


@dataclass
class OffsetPlan:
    """
    Pure description of the changes one offset operation makes.

    ``point_moves`` repositions existing defining points (in-place
    update, e.g. a circle's radius point). ``outlines`` are new
    world-coordinate outlines to add as polygons (each with optional
    hole rings); ``removed_entity_ids`` are source entities to remove
    (replacement). The offset command turns a plan into snapshot-backed
    point mutations plus add/remove commands.
    """

    point_moves: dict[EntityID, tuple[float, float]] = field(
        default_factory=dict
    )
    outlines: list[tuple[list[tuple[float, float]], bool]] = field(
        default_factory=list
    )
    holes: dict[int, list[list[tuple[float, float]]]] = field(
        default_factory=dict
    )
    removed_entity_ids: list[EntityID] = field(default_factory=list)


@dataclass
class PolygonOutline:
    """
    A single connected outline (open polyline or closed ring), ready
    for offsetting. Carries the IDs of the source entities the outline
    was sampled from; the offset command removes them (replacement).
    """

    vertices: list[tuple[float, float]] = field(default_factory=list)
    closed: bool = False
    source_ids: list[EntityID] = field(default_factory=list)

    def offset(
        self, distance: float, tolerance: float = 0.02
    ) -> list[tuple[list[tuple[float, float]], bool]]:
        return offset_outline(self.vertices, self.closed, distance, tolerance)


def rings_to_geometry(
    rings: Sequence[Sequence[tuple[float, float]]],
) -> Geometry:
    """Builds a multi-ring closed geometry from a flat ring list.
    Ring winding is preserved: outer rings CCW, hole rings CW."""
    geo = Geometry()
    for ring in rings:
        if len(ring) < 3:
            continue
        geo.move_to(ring[0][0], ring[0][1])
        for x, y in ring[1:]:
            geo.line_to(x, y)
        geo.close_path()
    return geo


def group_rings_into_solids(
    rings: Sequence[Sequence[tuple[float, float]]],
) -> list[tuple[list[tuple[float, float]], list[list[tuple[float, float]]]]]:
    """
    Groups a flat ring list into (outer, holes) solids. Role
    classification and winding canonicalization (outer CCW, holes CW)
    are delegated to raygeo's topology; each hole is then assigned to
    the smallest outer ring that contains it, and unassignable holes
    are dropped.
    """
    geo = rings_to_geometry(rings)
    if geo.is_empty():
        return []
    normalized = geo.normalize_winding_orders()
    outers, holes = normalized.split_inner_and_outer_polygons()
    solids: list[tuple[list[tuple[float, float]], list]] = [
        (outer, []) for outer in outers
    ]
    solids.sort(key=lambda solid: get_polygon_area(solid[0]))
    for hole in holes:
        for outer, assigned in solids:
            if is_point_inside_polygon(hole[0], outer):
                assigned.append(hole)
                break
    return solids


def outline_item(
    vertices: list[tuple[float, float]],
    closed: bool,
    add_point: Callable[[float, float], EntityID],
    holes: Sequence[Sequence[tuple[float, float]]] | None = None,
) -> Polygon:
    """Builds the frame points and Polygon for a world-coordinate
    outline. For closed outlines the frame center is the pole of
    inaccessibility (guaranteed inside the contour) and the handle is
    snapped onto the contour; open outlines use the bounding-box center
    with the handle snapped to the nearest polyline point. Hole rings
    share the outer ring's frame. The frame points are allocated
    through ``add_point``."""
    center, handle = _frame_for_outline(vertices, closed)
    center_pid = add_point(*center)
    handle_pid = add_point(*handle)
    local = _normalize_vertices(center, handle, vertices)
    rings: list[list[tuple[float, float]]] | None = None
    if holes:
        rings = [_normalize_vertices(center, handle, hole) for hole in holes]
    return Polygon(
        center_pid,
        handle_pid,
        local,
        closed=closed,
        rings=rings,
    )


def _normalize_vertices(
    center: tuple[float, float],
    handle: tuple[float, float],
    world_vertices: Sequence[Sequence[float]],
) -> list[tuple[float, float]]:
    """Converts world coordinates to frame-local (u, v) pairs."""
    cx, cy = center
    dx, dy = handle[0] - cx, handle[1] - cy
    scale = math.hypot(dx, dy)
    if scale < 1e-9:
        raise ValueError("Polygon frame handle coincides with center")
    ux, uy = dx / scale, dy / scale
    return [
        (
            ((x - cx) * ux + (y - cy) * uy) / scale,
            (-(x - cx) * uy + (y - cy) * ux) / scale,
        )
        for x, y in world_vertices
    ]


def _frame_for_outline(
    vertices: list[tuple[float, float]],
    closed: bool,
) -> tuple[tuple[float, float], tuple[float, float]]:
    """Chooses the (center, handle) frame for an outline. The handle is
    snapped onto the contour from the short-side midpoints of the
    oriented bounding box, so it sits at the far end of the shape's
    long axis and stays well clear of the center. The center keeps the
    pole of inaccessibility (closed) / bbox center (open)."""
    xs = [v[0] for v in vertices]
    ys = [v[1] for v in vertices]
    cx = (min(xs) + max(xs)) / 2
    cy = (min(ys) + max(ys)) / 2
    if closed:
        inner = find_largest_circle(vertices)
        if inner is not None:
            cx, cy = inner[0]

    handle: tuple[float, float] | None = None
    best_dist_sq = 1e-12
    for reference in _short_edge_midpoints(vertices):
        closest = _closest_outline_point(vertices, closed, *reference)
        if closest is None:
            continue
        point, _dist_sq = closest
        dx, dy = point[0] - cx, point[1] - cy
        frame_dist_sq = dx * dx + dy * dy
        if frame_dist_sq > best_dist_sq:
            best_dist_sq = frame_dist_sq
            handle = point
    if handle is not None:
        return (cx, cy), handle

    scale = (max(xs) - min(xs)) / 2
    if scale <= 1e-9:
        scale = max((max(ys) - min(ys)) / 2, 1.0)
    return (cx, cy), (cx + scale, cy)


def _short_edge_midpoints(
    vertices: list[tuple[float, float]],
) -> list[tuple[float, float]]:
    """Returns the midpoints of the two short sides of the outline's
    minimum-area oriented bounding box."""
    hull = get_polygon_convex_hull(vertices)
    if len(hull) < 2:
        return []
    best: tuple[float, tuple[float, float], tuple[float, float]] | None = None
    for i in range(len(hull)):
        p1 = hull[i]
        p2 = hull[(i + 1) % len(hull)]
        ex, ey = p2[0] - p1[0], p2[1] - p1[1]
        edge_len = math.hypot(ex, ey)
        if edge_len < 1e-12:
            continue
        ux, uy = ex / edge_len, ey / edge_len
        us = []
        vs = []
        for px, py in hull:
            dx, dy = px - p1[0], py - p1[1]
            us.append(dx * ux + dy * uy)
            vs.append(-dx * uy + dy * ux)
        w = max(us) - min(us)
        h = max(vs) - min(vs)
        area = w * h
        if best is not None and area >= best[0]:
            continue
        u_mid = (max(us) + min(us)) / 2
        v_mid = (max(vs) + min(vs)) / 2
        center = (
            p1[0] + u_mid * ux - v_mid * uy,
            p1[1] + u_mid * uy + v_mid * ux,
        )
        if w >= h:
            long_dir = (ux, uy)
            long_half = w / 2
        else:
            long_dir = (-uy, ux)
            long_half = h / 2
        ends = (
            (
                center[0] + long_dir[0] * long_half,
                center[1] + long_dir[1] * long_half,
            ),
            (
                center[0] - long_dir[0] * long_half,
                center[1] - long_dir[1] * long_half,
            ),
        )
        best = (area, ends[0], ends[1])
    if best is None:
        return []
    return [best[1], best[2]]


def _closest_outline_point(
    vertices: list[tuple[float, float]],
    closed: bool,
    x: float,
    y: float,
) -> tuple[tuple[float, float], float] | None:
    """Finds the closest point on the outline to (x, y). Returns
    ((px, py), distance_squared) or None for degenerate outlines."""
    if closed:
        result = get_polygon_closest_point(vertices, x, y)
        if result is None:
            return None
        _, point, dist_sq = result
        return point, dist_sq
    best: tuple[tuple[float, float], float] | None = None
    for p1, p2 in itertools.pairwise(vertices):
        _, point, dist = get_line_segment_closest_point(p1, p2, x, y)
        if best is None or dist * dist < best[1]:
            best = (point, dist * dist)
    return best


def offset_outline(
    vertices: Sequence[tuple[float, float]],
    closed: bool,
    distance: float,
    tolerance: float = 0.02,
) -> list[tuple[list[tuple[float, float]], bool]]:
    """
    Offsets a single connected outline by the given distance.

    Closed outlines are grown (positive distance) or shrunk (negative)
    via raygeo's Clipper2 offsetting; open outlines become a closed
    slot outline thickened on both sides with round end caps. The
    result may contain multiple outlines when the offset of a
    self-intersecting input splits into disjoint pieces, and may be
    empty when the offset collapses.

    Returns a list of (vertices, closed) pairs.
    """
    if len(vertices) < 2 or abs(distance) < 1e-9:
        return []
    if closed:
        geo = Geometry.from_points(vertices, True)
        grown = geo.grow(distance)
        return [(list(poly), True) for poly in grown.to_polygons(tolerance)]
    grown = offset_polyline(
        vertices, distance, JoinStyle.MITER, EndStyle.ROUND
    )
    return [(list(poly), True) for poly in grown]


def plan_offset_for_item(
    registry,
    item,
    offset: float,
) -> OffsetPlan | None:
    """Plans the offset of one offsettable item: in-place point moves
    for circles, arcs and ellipses; replacement outlines for open
    polygons, sampled chains and multi-ring shapes."""
    if isinstance(item, PolygonOutline):
        return _plan_outline_offset(item, offset)
    kind = item.type_name()
    if kind == "circle":
        return _plan_circle_offset(registry, item, offset)
    if kind == "arc":
        return _plan_arc_offset(registry, item, offset)
    if kind == "ellipse":
        return _plan_ellipse_offset(registry, item, offset)
    if kind == "polygon":
        return _plan_polygon_offset(registry, item, offset)
    return None


def preview_polylines_for_item(
    registry,
    item,
    offset: float,
) -> list[list[tuple[float, float]]]:
    """Coarse polylines of one item's offset for live preview, without
    touching the sketch."""
    if isinstance(item, PolygonOutline):
        return [
            list(vertices) + [vertices[0]] if closed else list(vertices)
            for vertices, closed in offset_outline(
                item.vertices, item.closed, offset
            )
        ]
    kind = item.type_name()
    if kind == "circle":
        center = registry.get_point(item.center_idx)
        radius_pt = registry.get_point(item.radius_pt_idx)
        radius = math.hypot(radius_pt[0] - center[0], radius_pt[1] - center[1])
        if radius + offset <= _MIN_OFFSET_RADIUS:
            return []
        ring = get_circle_polygon(center, radius + offset)
        return [list(ring) + [ring[0]]]
    if kind == "arc":
        start = registry.get_point(item.start_idx)
        end = registry.get_point(item.end_idx)
        center = registry.get_point(item.center_idx)
        radius = (
            math.hypot(start[0] - center[0], start[1] - center[1]) + offset
        )
        if radius <= _MIN_OFFSET_RADIUS:
            return []
        start_angle = math.atan2(start[1] - center[1], start[0] - center[0])
        end_angle = math.atan2(end[1] - center[1], end[0] - center[0])
        sweep = get_arc_sweep(start_angle, end_angle, item.clockwise)
        segments = max(8, int(abs(sweep) / math.pi * 32))
        return [get_arc_samples(center, radius, start_angle, sweep, segments)]
    if kind == "ellipse":
        center = registry.get_point(item.center_idx)
        rx = math.hypot(
            registry.get_point(item.radius_x_point_idx)[0] - center[0],
            registry.get_point(item.radius_x_point_idx)[1] - center[1],
        )
        ry = math.hypot(
            registry.get_point(item.radius_y_point_idx)[0] - center[0],
            registry.get_point(item.radius_y_point_idx)[1] - center[1],
        )
        if rx + offset <= 1e-9 or ry + offset <= 1e-9:
            return []
        rotation = math.atan2(
            registry.get_point(item.radius_x_point_idx)[1] - center[1],
            registry.get_point(item.radius_x_point_idx)[0] - center[0],
        )
        return [
            get_ellipse_samples(center, rx + offset, ry + offset, rotation, 64)
        ]
    if kind == "polygon":
        from .entity_ops import polygon_world_vertices

        if not item.closed:
            outline = PolygonOutline(
                polygon_world_vertices(registry, item), False
            )
            return preview_polylines_for_item(registry, outline, offset)
        polylines: list[list[tuple[float, float]]] = []
        for outer, holes in _polygon_offset_solids(registry, item, offset):
            for ring in [outer, *holes]:
                polylines.append(list(ring) + [ring[0]])
        return polylines
    return []


def _plan_outline_offset(
    outline: PolygonOutline, offset: float
) -> OffsetPlan | None:
    plan = OffsetPlan(removed_entity_ids=list(outline.source_ids))
    for vertices, closed in offset_outline(
        outline.vertices, outline.closed, offset
    ):
        plan.outlines.append((vertices, closed, None))
    if not plan.outlines:
        return None
    return plan


def _plan_circle_offset(registry, circle, offset: float) -> OffsetPlan | None:
    """Updates the circle in place: the radius point moves onto the
    concentric ring at the offset radius."""
    center = registry.get_point(circle.center_idx)
    radius_pt = registry.get_point(circle.radius_pt_idx)
    radius = math.hypot(radius_pt[0] - center[0], radius_pt[1] - center[1])
    new_radius = radius + offset
    if new_radius <= _MIN_OFFSET_RADIUS:
        return None

    dx, dy = radius_pt[0] - center[0], radius_pt[1] - center[1]
    length = math.hypot(dx, dy)
    if length < 1e-9:
        ux, uy = 1.0, 0.0
    else:
        ux, uy = dx / length, dy / length
    plan = OffsetPlan()
    plan.point_moves[circle.radius_pt_idx] = (
        center[0] + ux * new_radius,
        center[1] + uy * new_radius,
    )
    return plan


def _plan_arc_offset(registry, arc, offset: float) -> OffsetPlan | None:
    """Updates the arc in place: same center and angular extent,
    start and end points move onto the ring at the offset radius."""
    start = registry.get_point(arc.start_idx)
    end = registry.get_point(arc.end_idx)
    center = registry.get_point(arc.center_idx)
    radius = math.hypot(start[0] - center[0], start[1] - center[1])
    new_radius = radius + offset
    if new_radius <= _MIN_OFFSET_RADIUS:
        return None

    plan = OffsetPlan()
    for pid, pt in (
        (arc.start_idx, start),
        (arc.end_idx, end),
    ):
        dx, dy = pt[0] - center[0], pt[1] - center[1]
        length = math.hypot(dx, dy)
        if length < 1e-9:
            ux, uy = 1.0, 0.0
        else:
            ux, uy = dx / length, dy / length
        plan.point_moves[pid] = (
            center[0] + ux * new_radius,
            center[1] + uy * new_radius,
        )
    return plan


def _plan_ellipse_offset(
    registry, ellipse, offset: float
) -> OffsetPlan | None:
    """Updates the ellipse in place: both radius points move radially
    so each semi-axis grows by the offset. The result stays an ellipse
    (an approximation of the true offset)."""
    center = registry.get_point(ellipse.center_idx)
    rx_pt = registry.get_point(ellipse.radius_x_point_idx)
    ry_pt = registry.get_point(ellipse.radius_y_point_idx)
    rx = math.hypot(rx_pt[0] - center[0], rx_pt[1] - center[1])
    ry = math.hypot(ry_pt[0] - center[0], ry_pt[1] - center[1])
    if rx + offset <= 1e-9 or ry + offset <= 1e-9:
        return None

    plan = OffsetPlan()
    for pid, pt, radius in (
        (ellipse.radius_x_point_idx, rx_pt, rx),
        (ellipse.radius_y_point_idx, ry_pt, ry),
    ):
        scale = (radius + offset) / radius
        plan.point_moves[pid] = (
            center[0] + (pt[0] - center[0]) * scale,
            center[1] + (pt[1] - center[1]) * scale,
        )
    return plan


def _polygon_offset_solids(
    registry, polygon, offset: float
) -> list[tuple[list, list]]:
    """Grows the multi-ring shape by the offset and regroups the
    result into (outer, holes) solids."""
    from .entity_ops import polygon_world_rings

    grown = rings_to_geometry(polygon_world_rings(registry, polygon)).grow(
        offset
    )
    return group_rings_into_solids(grown.to_polygons(0.02))


def _plan_polygon_offset(
    registry, polygon, offset: float
) -> OffsetPlan | None:
    """Offsets a polygon. Closed shapes keep their entity type: the
    multi-ring geometry is grown via raygeo (holes shrink and may
    vanish) and regrouped into one polygon per result solid. Open
    polygons fall back to outline sampling."""
    from .entity_ops import polygon_world_vertices

    if not polygon.closed:
        outline = PolygonOutline(
            polygon_world_vertices(registry, polygon),
            False,
            source_ids=[polygon.id],
        )
        return _plan_outline_offset(outline, offset)
    solids = _polygon_offset_solids(registry, polygon, offset)
    if not solids:
        return None
    plan = OffsetPlan(removed_entity_ids=[polygon.id])
    for outer, holes in solids:
        plan.outlines.append((outer, True, holes or None))
    return plan


def rectangle_contains_rect(
    rings: list[list[tuple[float, float]]],
    rect: Rect,
) -> bool:
    """Whether every vertex of every ring lies inside the rect."""
    if not rings or not rings[0]:
        return False
    x_min, y_min, x_max, y_max = rect
    return all(
        x_min <= x <= x_max and y_min <= y <= y_max
        for ring in rings
        for x, y in ring
    )


def rectangle_intersects_rings(
    rings: list[list[tuple[float, float]]],
    closed: bool,
    rect: Rect,
) -> bool:
    """Whether any ring segment intersects the rect, the rect center
    lies inside the shape, or the shape is contained by the rect."""
    count = len(rings[0]) if rings else 0
    if count < 2:
        return False
    for vertices in rings:
        ring_count = len(vertices)
        for i in range(ring_count - 1 if not closed else ring_count):
            p1 = vertices[i]
            p2 = vertices[(i + 1) % ring_count]
            if does_line_segment_intersect_rect(p1, p2, rect):
                return True
    center = ((rect[0] + rect[2]) / 2, (rect[1] + rect[3]) / 2)
    outer = rings[0]
    if len(outer) >= 3 and is_point_inside_polygon(center, outer):
        return True
    return rectangle_contains_rect(rings, rect)
