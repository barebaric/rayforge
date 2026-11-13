import math
from collections.abc import Iterator
from typing import TYPE_CHECKING

from raygeo.geo.shape.circle import (
    get_circle_circle_intersections,
    get_line_circle_intersections,
)
from raygeo.geo.shape.line import get_line_segment_intersection
from raygeo.geo.types import Point as GeoPoint

from raygeo.geo.shape.arc import get_arc_sweep
from raygeo.sketch.entities.arc import Arc
from raygeo.sketch.entities.circle import Circle
from raygeo.sketch.entities.line import Line

from ..engine import SnapLineProducer
from ..types import DragContext, SnapLine, SnapLineType, SnapPoint

if TYPE_CHECKING:
    from raygeo.sketch.registry import EntityRegistry


def arc_is_angle_within_sweep(
    arc: Arc, angle: float, registry: EntityRegistry
) -> bool:
    """Whether the angle lies within the arc's sweep."""
    start = registry.get_point(arc.start_idx)
    end = registry.get_point(arc.end_idx)
    center = registry.get_point(arc.center_idx)
    if start is None or end is None or center is None:
        return False
    start_angle = math.atan2(
        start[1] - center[1], start[0] - center[0]
    )
    end_angle = math.atan2(end[1] - center[1], end[0] - center[0])
    sweep = get_arc_sweep(start_angle, end_angle, arc.clockwise)
    delta = (angle - start_angle) % (2.0 * math.pi)
    sweep = sweep % (2.0 * math.pi)
    if sweep < 1e-12:
        return True
    return delta <= sweep + 1e-9


class IntersectionsProducer(SnapLineProducer):
    def __init__(self, include_construction: bool = True) -> None:
        self._include_construction: bool = include_construction
        self._cache_key: tuple | None = None
        self._cached_intersections: list[GeoPoint] = []

    def produce(
        self,
        registry: "EntityRegistry",
        drag_position: GeoPoint,
        drag_context: DragContext,
        threshold: float,
    ) -> Iterator[SnapLine]:
        for ix, iy in self._get_intersections_cached(registry, drag_context):
            yield SnapLine(
                is_horizontal=False,
                coordinate=ix,
                line_type=SnapLineType.INTERSECTION,
            )
            yield SnapLine(
                is_horizontal=True,
                coordinate=iy,
                line_type=SnapLineType.INTERSECTION,
            )

    def produce_points(
        self,
        registry: "EntityRegistry",
        drag_position: GeoPoint,
        drag_context: DragContext,
        threshold: float,
    ) -> Iterator[SnapPoint]:
        x, y = drag_position
        for ix, iy in self._get_intersections_cached(registry, drag_context):
            dist = ((ix - x) ** 2 + (iy - y) ** 2) ** 0.5
            if dist <= threshold:
                yield SnapPoint(
                    x=ix,
                    y=iy,
                    line_type=SnapLineType.INTERSECTION,
                )

    def _get_intersections_cached(
        self, registry: "EntityRegistry", drag_context: DragContext
    ) -> list[GeoPoint]:
        key = self._geometry_key(registry, drag_context)
        if key != self._cache_key:
            self._cached_intersections = list(
                self._get_all_intersections(registry, drag_context)
            )
            self._cache_key = key
        return self._cached_intersections

    def _geometry_key(
        self, registry: "EntityRegistry", drag_context: DragContext
    ) -> tuple:
        """
        Identity of the geometry the intersections depend on: the
        positions of all points of non-dragged entities, plus the
        dragged sets that define that exclusion. Positions are
        quantized like the engine fingerprint so solver residual
        noise does not invalidate the cache; dragged geometry moves
        on every drag solve and is excluded from the computation
        anyway.
        """
        dragged_points = drag_context.dragged_point_ids
        dragged_entities = drag_context.dragged_entity_ids
        positions: list[tuple[float, float]] = []
        for entity in registry.entities():
            if entity.id in dragged_entities or any(
                pid in dragged_points for pid in entity.get_point_ids()
            ):
                continue
            if not self._include_construction and entity.construction:
                continue
            for pid in entity.get_point_ids():
                point = registry.get_point(pid)
                if point is not None:
                    positions.append((round(point[0], 6), round(point[1], 6)))
        return (
            registry.entity_version(),
            tuple(positions),
            frozenset(dragged_points),
            frozenset(dragged_entities),
        )

    def _get_all_intersections(
        self, registry: "EntityRegistry", drag_context: DragContext
    ) -> Iterator[GeoPoint]:
        entities = [
            e
            for e in registry.entities()
            if not drag_context.is_entity_dragged(e.id)
            and not any(
                drag_context.is_point_dragged(pid) for pid in e.get_point_ids()
            )
            and (self._include_construction or not e.construction)
        ]

        for i, e1 in enumerate(entities):
            for e2 in entities[i + 1 :]:
                yield from self._get_intersections(e1, e2, registry)

    def _get_intersections(
        self, e1: object, e2: object, registry: "EntityRegistry"
    ) -> Iterator[GeoPoint]:
        if isinstance(e1, Line) and isinstance(e2, Line):
            yield from self._line_line_intersections(e1, e2, registry)
        elif isinstance(e1, Line) and isinstance(e2, Arc):
            yield from self._line_arc_intersections(e1, e2, registry)
        elif isinstance(e1, Arc) and isinstance(e2, Line):
            yield from self._line_arc_intersections(e2, e1, registry)
        elif isinstance(e1, Line) and isinstance(e2, Circle):
            yield from self._line_circle_intersections(e1, e2, registry)
        elif isinstance(e1, Circle) and isinstance(e2, Line):
            yield from self._line_circle_intersections(e2, e1, registry)
        elif isinstance(e1, Arc) and isinstance(e2, Arc):
            yield from self._arc_arc_intersections(e1, e2, registry)
        elif isinstance(e1, Circle) and isinstance(e2, Circle):
            yield from self._circle_circle_intersections(e1, e2, registry)

    def _line_line_intersections(
        self,
        line1: Line,
        line2: Line,
        registry: "EntityRegistry",
    ) -> Iterator[GeoPoint]:
        p1 = registry.get_point(line1.p1_idx)
        p2 = registry.get_point(line1.p2_idx)
        p3 = registry.get_point(line2.p1_idx)
        p4 = registry.get_point(line2.p2_idx)

        if not all([p1, p2, p3, p4]):
            return

        result = get_line_segment_intersection(
            p1,
            p2,
            p3,
            p4,
        )
        if result is not None:
            yield result

    def _line_arc_intersections(
        self,
        line: Line,
        arc: Arc,
        registry: "EntityRegistry",
    ) -> Iterator[GeoPoint]:
        p1 = registry.get_point(line.p1_idx)
        p2 = registry.get_point(line.p2_idx)
        center = registry.get_point(arc.center_idx)
        start = registry.get_point(arc.start_idx)

        if not all([p1, p2, center, start]):
            return

        radius = math.hypot(start[0] - center[0], start[1] - center[1])
        if radius < 1e-10:
            return

        for ix, iy in get_line_circle_intersections(
            p1,
            p2,
            center,
            radius,
        ):
            angle = math.atan2(iy - center[1], ix - center[0])
            if arc_is_angle_within_sweep(arc, angle, registry):
                yield (ix, iy)

    def _line_circle_intersections(
        self,
        line: Line,
        circle: Circle,
        registry: "EntityRegistry",
    ) -> Iterator[GeoPoint]:
        p1 = registry.get_point(line.p1_idx)
        p2 = registry.get_point(line.p2_idx)
        center = registry.get_point(circle.center_idx)
        radius_pt = registry.get_point(circle.radius_pt_idx)

        if not all([p1, p2, center, radius_pt]):
            return

        radius = math.hypot(
            radius_pt[0] - center[0], radius_pt[1] - center[1]
        )
        if radius < 1e-10:
            return

        yield from get_line_circle_intersections(
            p1,
            p2,
            center,
            radius,
        )

    def _arc_arc_intersections(
        self,
        arc1: Arc,
        arc2: Arc,
        registry: "EntityRegistry",
    ) -> Iterator[GeoPoint]:
        c1 = registry.get_point(arc1.center_idx)
        s1 = registry.get_point(arc1.start_idx)
        c2 = registry.get_point(arc2.center_idx)
        s2 = registry.get_point(arc2.start_idx)

        if not all([c1, s1, c2, s2]):
            return

        r1 = math.hypot(s1[0] - c1[0], s1[1] - c1[1])
        r2 = math.hypot(s2[0] - c2[0], s2[1] - c2[1])

        for ix, iy in get_circle_circle_intersections(
            c1, r1, c2, r2
        ):
            angle1 = math.atan2(iy - c1[1], ix - c1[0])
            angle2 = math.atan2(iy - c2[1], ix - c2[0])
            if arc_is_angle_within_sweep(
                arc1, angle1, registry
            ) and arc_is_angle_within_sweep(arc2, angle2, registry):
                yield (ix, iy)

    def _circle_circle_intersections(
        self,
        circle1: Circle,
        circle2: Circle,
        registry: "EntityRegistry",
    ) -> Iterator[GeoPoint]:
        c1 = registry.get_point(circle1.center_idx)
        r1_pt = registry.get_point(circle1.radius_pt_idx)
        c2 = registry.get_point(circle2.center_idx)
        r2_pt = registry.get_point(circle2.radius_pt_idx)

        if not all([c1, r1_pt, c2, r2_pt]):
            return

        r1 = math.hypot(r1_pt[0] - c1[0], r1_pt[1] - c1[1])
        r2 = math.hypot(r2_pt[0] - c2[0], r2_pt[1] - c2[1])

        yield from get_circle_circle_intersections(
            c1, r1, c2, r2
        )
