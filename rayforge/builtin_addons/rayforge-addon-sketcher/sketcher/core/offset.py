"""Offset-command helpers over the compiled raygeo model."""

from __future__ import annotations

import itertools
import math
from collections.abc import Callable, Sequence

from raygeo.geo import Geometry
from raygeo.geo.algo.polylabel import find_largest_circle
from raygeo.geo.shape.line import get_line_segment_closest_point
from raygeo.geo.shape.polygon import (
    get_polygon_area,
    get_polygon_closest_point,
    get_polygon_convex_hull,
    is_point_inside_polygon,
)
from raygeo.sketch.entities import Polygon as PolygonEntity
from raygeo.sketch.entities import PolygonOutline

from .types import EntityID


def rings_to_geometry(
    rings: Sequence[Sequence[tuple[float, float]]],
) -> Geometry:
    """Builds a multi-ring closed geometry from a flat ring list."""
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
    """Groups a flat ring list into (outer, holes) solids."""
    geo = rings_to_geometry(rings)
    if geo.is_empty():
        return []
    normalized = geo.normalize_winding_orders()
    outers, holes = normalized.split_inner_and_outer_polygons()
    solids: list[tuple[list[tuple[float, float]], list]] = [
        (list(outer), []) for outer in outers
    ]
    solids.sort(key=lambda solid: get_polygon_area(solid[0]))
    for hole in holes:
        for outer, assigned in solids:
            if is_point_inside_polygon(hole[0], outer):
                assigned.append(list(hole))
                break
    return solids


def offset_outline(
    vertices: Sequence[tuple[float, float]],
    closed: bool,
    distance: float,
    tolerance: float = 0.02,
) -> list[tuple[list[tuple[float, float]], bool]]:
    """Offsets a single connected outline."""
    return PolygonOutline(vertices, closed).offset(distance, tolerance)


def outline_item(
    vertices: list[tuple[float, float]],
    closed: bool,
    allocate_id: Callable[[], EntityID],
    holes: Sequence[Sequence[tuple[float, float]]] | None = None,
) -> tuple:
    """Builds the frame points and polygon for a world outline."""
    center, handle = _frame_for_outline(vertices, closed)
    return PolygonEntity.build(
        allocate_id(),
        allocate_id(),
        allocate_id(),
        center,
        handle,
        vertices,
        closed=closed,
        world_holes=list(holes) if holes else None,
    )


def _frame_for_outline(
    vertices: list[tuple[float, float]],
    closed: bool,
) -> tuple[tuple[float, float], tuple[float, float]]:
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
