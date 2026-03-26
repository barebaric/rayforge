import math
from typing import Iterator, TYPE_CHECKING

from rayforge.core.geo.types import Point as GeoPoint
from ..types import SnapLine, SnapPoint, SnapLineType, DragContext
from ..engine import SnapLineProducer

if TYPE_CHECKING:
    from ...registry import EntityRegistry
    from ...entities import Line, Arc, Circle


class IntersectionsProducer(SnapLineProducer):
    def __init__(self, include_construction: bool = True):
        self._include_construction = include_construction

    def produce(
        self,
        registry: "EntityRegistry",
        drag_position: GeoPoint,
        drag_context: DragContext,
        threshold: float,
    ) -> Iterator[SnapLine]:
        x, y = drag_position
        for ix, iy in self._get_all_intersections(registry, drag_context):
            if abs(ix - x) <= threshold:
                yield SnapLine(
                    is_horizontal=False,
                    coordinate=ix,
                    line_type=SnapLineType.INTERSECTION,
                )
            if abs(iy - y) <= threshold:
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
        for ix, iy in self._get_all_intersections(registry, drag_context):
            dist = ((ix - x) ** 2 + (iy - y) ** 2) ** 0.5
            if dist <= threshold:
                yield SnapPoint(
                    x=ix,
                    y=iy,
                    line_type=SnapLineType.INTERSECTION,
                )

    def _get_all_intersections(
        self, registry: "EntityRegistry", drag_context: DragContext
    ) -> Iterator[GeoPoint]:
        from ...entities import Line, Arc, Circle

        entities = [
            e
            for e in registry.entities
            if not drag_context.is_entity_dragged(e.id)
            and (self._include_construction or not e.construction)
        ]

        for i, e1 in enumerate(entities):
            for e2 in entities[i + 1 :]:
                for intersection in self._get_intersections(e1, e2, registry):
                    yield intersection

    def _get_intersections(
        self, e1, e2, registry: "EntityRegistry"
    ) -> Iterator[GeoPoint]:
        from ...entities import Line, Arc, Circle

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
        self, line1: "Line", line2: "Line", registry: "EntityRegistry"
    ) -> Iterator[GeoPoint]:
        p1 = registry.get_point(line1.p1_idx)
        p2 = registry.get_point(line1.p2_idx)
        p3 = registry.get_point(line2.p1_idx)
        p4 = registry.get_point(line2.p2_idx)

        if not all([p1, p2, p3, p4]):
            return

        x1, y1 = p1.x, p1.y
        x2, y2 = p2.x, p2.y
        x3, y3 = p3.x, p3.y
        x4, y4 = p4.x, p4.y

        denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
        if abs(denom) < 1e-10:
            return

        t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / denom
        u = -((x1 - x2) * (y1 - y3) - (y1 - y2) * (x1 - x3)) / denom

        if 0 <= t <= 1 and 0 <= u <= 1:
            ix = x1 + t * (x2 - x1)
            iy = y1 + t * (y2 - y1)
            yield (ix, iy)

    def _line_arc_intersections(
        self, line: "Line", arc: "Arc", registry: "EntityRegistry"
    ) -> Iterator[GeoPoint]:
        p1 = registry.get_point(line.p1_idx)
        p2 = registry.get_point(line.p2_idx)
        center = registry.get_point(arc.center_idx)
        start = registry.get_point(arc.start_idx)

        if not all([p1, p2, center, start]):
            return

        radius = math.hypot(start.x - center.x, start.y - center.y)
        if radius < 1e-10:
            return

        for ix, iy in self._line_circle_intersections_raw(
            p1.x, p1.y, p2.x, p2.y, center.x, center.y, radius
        ):
            angle = math.atan2(iy - center.y, ix - center.x)
            if arc.is_angle_within_sweep(angle, registry):
                yield (ix, iy)

    def _line_circle_intersections(
        self, line: "Line", circle: "Circle", registry: "EntityRegistry"
    ) -> Iterator[GeoPoint]:
        p1 = registry.get_point(line.p1_idx)
        p2 = registry.get_point(line.p2_idx)
        center = registry.get_point(circle.center_idx)
        radius_pt = registry.get_point(circle.radius_pt_idx)

        if not all([p1, p2, center, radius_pt]):
            return

        radius = math.hypot(radius_pt.x - center.x, radius_pt.y - center.y)
        if radius < 1e-10:
            return

        yield from self._line_circle_intersections_raw(
            p1.x, p1.y, p2.x, p2.y, center.x, center.y, radius
        )

    def _line_circle_intersections_raw(
        self,
        x1: float,
        y1: float,
        x2: float,
        y2: float,
        cx: float,
        cy: float,
        r: float,
    ) -> Iterator[GeoPoint]:
        dx = x2 - x1
        dy = y2 - y1
        fx = x1 - cx
        fy = y1 - cy

        a = dx * dx + dy * dy
        b = 2 * (fx * dx + fy * dy)
        c = fx * fx + fy * fy - r * r

        discriminant = b * b - 4 * a * c
        if discriminant < 0:
            return

        sqrt_disc = math.sqrt(discriminant)
        for t in [(-b - sqrt_disc) / (2 * a), (-b + sqrt_disc) / (2 * a)]:
            if 0 <= t <= 1:
                ix = x1 + t * dx
                iy = y1 + t * dy
                yield (ix, iy)

    def _arc_arc_intersections(
        self, arc1: "Arc", arc2: "Arc", registry: "EntityRegistry"
    ) -> Iterator[GeoPoint]:
        c1 = registry.get_point(arc1.center_idx)
        s1 = registry.get_point(arc1.start_idx)
        c2 = registry.get_point(arc2.center_idx)
        s2 = registry.get_point(arc2.start_idx)

        if not all([c1, s1, c2, s2]):
            return

        r1 = math.hypot(s1.x - c1.x, s1.y - c1.y)
        r2 = math.hypot(s2.x - c2.x, s2.y - c2.y)

        for ix, iy in self._circle_circle_intersections_raw(
            c1.x, c1.y, r1, c2.x, c2.y, r2
        ):
            angle1 = math.atan2(iy - c1.y, ix - c1.x)
            angle2 = math.atan2(iy - c2.y, ix - c2.x)
            if arc1.is_angle_within_sweep(
                angle1, registry
            ) and arc2.is_angle_within_sweep(angle2, registry):
                yield (ix, iy)

    def _circle_circle_intersections(
        self, circle1: "Circle", circle2: "Circle", registry: "EntityRegistry"
    ) -> Iterator[GeoPoint]:
        c1 = registry.get_point(circle1.center_idx)
        r1_pt = registry.get_point(circle1.radius_pt_idx)
        c2 = registry.get_point(circle2.center_idx)
        r2_pt = registry.get_point(circle2.radius_pt_idx)

        if not all([c1, r1_pt, c2, r2_pt]):
            return

        r1 = math.hypot(r1_pt.x - c1.x, r1_pt.y - c1.y)
        r2 = math.hypot(r2_pt.x - c2.x, r2_pt.y - c2.y)

        yield from self._circle_circle_intersections_raw(
            c1.x, c1.y, r1, c2.x, c2.y, r2
        )

    def _circle_circle_intersections_raw(
        self,
        x1: float,
        y1: float,
        r1: float,
        x2: float,
        y2: float,
        r2: float,
    ) -> Iterator[GeoPoint]:
        d = math.hypot(x2 - x1, y2 - y1)
        if d < 1e-10 or d > r1 + r2 or d < abs(r1 - r2):
            return

        a = (r1 * r1 - r2 * r2 + d * d) / (2 * d)
        h_sq = r1 * r1 - a * a
        if h_sq < 0:
            return
        h = math.sqrt(h_sq)

        px = x1 + a * (x2 - x1) / d
        py = y1 + a * (y2 - y1) / d

        if h < 1e-10:
            yield (px, py)
        else:
            dx = h * (y2 - y1) / d
            dy = h * (x1 - x2) / d
            yield (px + dx, py + dy)
            yield (px - dx, py - dy)
