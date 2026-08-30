import itertools
import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from raygeo.geo import Geometry
from raygeo.geo.algo.polylabel import find_largest_circle
from raygeo.geo.shape.line import (
    does_line_segment_intersect_rect,
    get_line_segment_closest_point,
)
from raygeo.geo.shape.polygon import (
    EndStyle,
    JoinStyle,
    get_polygon_closest_point,
    get_polygon_convex_hull,
    is_point_inside_polygon,
    offset_polyline,
)
from raygeo.geo.types import Rect

from ..types import EntityID
from .entity import Entity, OffsetPlan, _quantize

if TYPE_CHECKING:
    from ..commands.mirror import MirrorAxis
    from ..constraints import Constraint
    from ..registry import EntityRegistry
    from ..sketch import Sketch
    from .point import Point


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

    def plan_offset(
        self,
        registry: "EntityRegistry",
        offset: float,
        allocate_id: Callable[[], EntityID],
    ) -> OffsetPlan | None:
        """
        Offsets this outline via raygeo and plans one PolygonEntity per
        result outline, with fresh frame points. The sampled source
        entities are removed (replacement). Returns None when the
        offset produced no geometry. The registry is unused (an outline
        carries its world geometry) and exists only to keep the offset
        protocol uniform with entities.
        """
        plan = OffsetPlan(removed_entity_ids=list(self.source_ids))
        for vertices, closed in offset_outline(
            self.vertices, self.closed, offset
        ):
            center_pt, handle_pt, entity = _outline_item(
                vertices, closed, allocate_id
            )
            plan.points.extend((center_pt, handle_pt))
            plan.entities.append(entity)
        if not plan.entities:
            return None
        return plan

    def preview_polylines(
        self, registry: "EntityRegistry", offset: float
    ) -> list[list[tuple[float, float]]]:
        """Offsets this outline and flattens every result outline into
        a closed polyline, for live preview."""
        return [
            list(vertices) + [vertices[0]] if closed else list(vertices)
            for vertices, closed in offset_outline(
                self.vertices, self.closed, offset
            )
        ]


def _outline_item(
    vertices: list[tuple[float, float]],
    closed: bool,
    allocate_id: Callable[[], EntityID],
) -> tuple:
    """Builds the frame points and PolygonEntity for a world-coordinate
    outline. For closed outlines the frame center is the pole of
    inaccessibility (guaranteed inside the contour) and the handle is
    snapped onto the contour; open outlines use the bounding-box center
    with the handle snapped to the nearest polyline point."""
    center, handle = _frame_for_outline(vertices, closed)
    return PolygonEntity.build(
        allocate_id(),
        allocate_id(),
        allocate_id(),
        center,
        handle,
        vertices,
        closed=closed,
    )


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


class PolygonEntity(Entity):
    """
    A closed or open outline stored as one atomic shape.

    The outline has no individually editable edges; it is transformed
    as a whole. Vertices are kept normalized in a local frame defined
    by two registry points (the Ellipse pattern):

        world = center + s * (u * x_hat + v * y_hat)

    with x_hat the unit vector from center to handle,
    y_hat = (-x_hat.y, x_hat.x) and s = |handle - center|.
    Translating, rotating and uniformly scaling the outline therefore
    works through the regular point-dragging machinery.
    """

    def __init__(
        self,
        id: EntityID,
        center_idx: EntityID,
        handle_idx: EntityID,
        vertices: Sequence[Sequence[float]],
        closed: bool = False,
        construction: bool = False,
    ):
        super().__init__(id, construction)
        self.center_idx: EntityID = center_idx
        self.handle_idx: EntityID = handle_idx
        self.vertices: list[tuple[float, float]] = [
            (float(u), float(v)) for u, v in vertices
        ]
        self.closed: bool = closed
        self.type = "polygon"

    @staticmethod
    def normalize_vertices(
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

    @classmethod
    def build(
        cls,
        entity_id: EntityID,
        center_id: EntityID,
        handle_id: EntityID,
        center: tuple[float, float],
        handle: tuple[float, float],
        world_vertices: Sequence[Sequence[float]],
        closed: bool = False,
        construction: bool = False,
    ) -> tuple["Point", "Point", "PolygonEntity"]:
        """
        Creates the frame points and the entity for a world-coordinate
        outline. Returns (center_point, handle_point, entity).
        """
        from .point import Point

        center_pt = Point(center_id, center[0], center[1])
        handle_pt = Point(handle_id, handle[0], handle[1])
        vertices = cls.normalize_vertices(center, handle, world_vertices)
        entity = cls(
            entity_id,
            center_id,
            handle_id,
            vertices,
            closed=closed,
            construction=construction,
        )
        return center_pt, handle_pt, entity

    def _frame(
        self, registry: "EntityRegistry"
    ) -> tuple[float, float, float, float, float]:
        """Returns (cx, cy, x_hat_x, x_hat_y, scale)."""
        center = registry.get_point(self.center_idx)
        handle = registry.get_point(self.handle_idx)
        dx = handle.x - center.x
        dy = handle.y - center.y
        scale = math.hypot(dx, dy)
        if scale < 1e-9:
            return center.x, center.y, 1.0, 0.0, 0.0
        return center.x, center.y, dx / scale, dy / scale, scale

    def get_world_vertices(
        self, registry: "EntityRegistry"
    ) -> list[tuple[float, float]]:
        cx, cy, ux, uy, scale = self._frame(registry)
        return [
            (
                cx + scale * (u * ux - v * uy),
                cy + scale * (u * uy + v * ux),
            )
            for u, v in self.vertices
        ]

    def to_outline(
        self, registry: "EntityRegistry"
    ) -> tuple[list[tuple[float, float]], bool]:
        """Returns (world_vertices, closed) for offset processing."""
        return self.get_world_vertices(registry), self.closed

    def as_offset_item(self, sketch: "Sketch") -> "PolygonOutline":
        """A polygon offsets on its own; the offset result replaces it."""
        vertices, closed = self.to_outline(sketch.registry)
        return PolygonOutline(vertices, closed, source_ids=[self.id])

    def get_point_ids(self) -> list[EntityID]:
        return [self.center_idx, self.handle_idx]

    def get_endpoint_ids(self) -> list[EntityID]:
        return []

    def get_junction_point_ids(self) -> list[EntityID]:
        return [self.center_idx, self.handle_idx]

    def get_state(self) -> dict[str, Any] | None:
        state = super().get_state() or {}
        state["vertices"] = list(self.vertices)
        state["closed"] = self.closed
        return state

    def set_state(self, state: dict[str, Any]) -> None:
        super().set_state(state)
        if "vertices" in state:
            self.vertices = [
                (float(u), float(v)) for u, v in state["vertices"]
            ]
        if "closed" in state:
            self.closed = state["closed"]

    def geometry_signature(self, registry: "EntityRegistry") -> tuple:
        return (
            *super().geometry_signature(registry),
            tuple((_quantize(u), _quantize(v)) for u, v in self.vertices),
            self.closed,
        )

    def update_constrained_status(
        self, registry: "EntityRegistry", constraints: Sequence["Constraint"]
    ) -> None:
        center = registry.get_point(self.center_idx)
        handle = registry.get_point(self.handle_idx)
        self.constrained = center.constrained and handle.constrained

    def get_rigidly_connected_points(
        self, point_id: EntityID
    ) -> list[EntityID]:
        if point_id == self.center_idx:
            return [self.center_idx, self.handle_idx]
        return []

    def get_drag_anchor_points(self, point_id: EntityID) -> list[EntityID]:
        if point_id == self.handle_idx:
            return [self.center_idx]
        return []

    def mirror(self, axis: "MirrorAxis") -> None:
        # The frame points are mirrored centrally by the command, which
        # flips the frame's chirality; negating the local v component
        # mirrors the outline with it (true for both mirror axes).
        self.vertices = [(u, -v) for u, v in self.vertices]

    def hit_test(
        self,
        mx: float,
        my: float,
        threshold: float,
        registry: "EntityRegistry",
    ) -> bool:
        vertices = self.get_world_vertices(registry)
        if len(vertices) < 2:
            return False
        count = len(vertices)
        for i in range(count - 1 if not self.closed else count):
            p1 = vertices[i]
            p2 = vertices[(i + 1) % count]
            _, _, dist_sq = get_line_segment_closest_point(p1, p2, mx, my)
            if dist_sq < threshold**2:
                return True
        return False

    def is_contained_by(
        self,
        rect: Rect,
        registry: "EntityRegistry",
    ) -> bool:
        vertices = self.get_world_vertices(registry)
        if not vertices:
            return False
        x_min, y_min, x_max, y_max = rect
        return all(
            x_min <= x <= x_max and y_min <= y <= y_max for x, y in vertices
        )

    def intersects_rect(
        self,
        rect: Rect,
        registry: "EntityRegistry",
    ) -> bool:
        vertices = self.get_world_vertices(registry)
        count = len(vertices)
        if count < 2:
            return False
        for i in range(count - 1 if not self.closed else count):
            p1 = vertices[i]
            p2 = vertices[(i + 1) % count]
            if does_line_segment_intersect_rect(p1, p2, rect):
                return True
        center = ((rect[0] + rect[2]) / 2, (rect[1] + rect[3]) / 2)
        if is_point_inside_polygon(center, vertices):
            return True
        return self.is_contained_by(rect, registry)

    def to_geometry(self, registry: "EntityRegistry") -> Geometry:
        vertices = self.get_world_vertices(registry)
        geo = Geometry()
        if not vertices:
            return geo
        geo.move_to(vertices[0][0], vertices[0][1])
        for x, y in vertices[1:]:
            geo.line_to(x, y)
        if self.closed:
            geo.close_path()
        return geo

    def to_polygon_vertices(
        self,
        registry: "EntityRegistry",
        forward: bool,
    ) -> list[tuple[float, float]]:
        vertices = self.get_world_vertices(registry)
        return list(vertices) if forward else list(reversed(vertices))

    def to_polyline(
        self,
        registry: "EntityRegistry",
        tolerance: float = 0.1,
    ) -> list[tuple[float, float]]:
        vertices = self.get_world_vertices(registry)
        if self.closed and vertices:
            return vertices + [vertices[0]]
        return vertices

    def create_fill_geometry(
        self, registry: "EntityRegistry"
    ) -> Geometry | None:
        if not self.closed:
            return None
        return self.to_geometry(registry)

    def to_dict(self) -> dict[str, Any]:
        data = super().to_dict()
        data.update(
            {
                "center_idx": self.center_idx,
                "handle_idx": self.handle_idx,
                "vertices": [list(v) for v in self.vertices],
                "closed": self.closed,
            }
        )
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "PolygonEntity":
        return cls(
            id=data["id"],
            center_idx=data["center_idx"],
            handle_idx=data["handle_idx"],
            vertices=data["vertices"],
            closed=data.get("closed", False),
            construction=data.get("construction", False),
        )

    def __repr__(self) -> str:
        return (
            f"PolygonEntity(id={self.id}, center={self.center_idx}, "
            f"handle={self.handle_idx}, closed={self.closed}, "
            f"vertices={len(self.vertices)})"
        )
