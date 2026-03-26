import math
from typing import Iterator, Optional, TYPE_CHECKING

from rayforge.core.geo.types import Point as GeoPoint
from ...entities import Line, Arc, Circle
from ..types import SnapLine, SnapPoint, SnapLineType, DragContext
from ..engine import SnapLineProducer

if TYPE_CHECKING:
    from ...registry import EntityRegistry


class OnEntityProducer(SnapLineProducer):
    def produce(
        self,
        registry: "EntityRegistry",
        drag_position: GeoPoint,
        drag_context: DragContext,
        threshold: float,
    ) -> Iterator[SnapLine]:
        return iter(())

    def produce_points(
        self,
        registry: "EntityRegistry",
        drag_position: GeoPoint,
        drag_context: DragContext,
        threshold: float,
    ) -> Iterator[SnapPoint]:
        x, y = drag_position

        for entity in registry.entities:
            if drag_context.is_entity_dragged(entity.id):
                continue
            entity_points = entity.get_point_ids()
            if any(
                drag_context.is_point_dragged(pid) for pid in entity_points
            ):
                continue

            if isinstance(entity, Line):
                snap_point = self._nearest_point_on_line(
                    entity, registry, x, y, threshold
                )
            elif isinstance(entity, Arc):
                snap_point = self._nearest_point_on_arc(
                    entity, registry, x, y, threshold
                )
            elif isinstance(entity, Circle):
                snap_point = self._nearest_point_on_circle(
                    entity, registry, x, y, threshold
                )
            else:
                continue

            if snap_point is not None:
                yield SnapPoint(
                    x=snap_point[0],
                    y=snap_point[1],
                    line_type=SnapLineType.ON_ENTITY,
                    source=entity,
                )

    def _nearest_point_on_line(
        self,
        line: Line,
        registry: "EntityRegistry",
        x: float,
        y: float,
        threshold: float,
    ) -> Optional[GeoPoint]:
        p1 = registry.get_point(line.p1_idx)
        p2 = registry.get_point(line.p2_idx)
        if not p1 or not p2:
            return None

        dx = p2.x - p1.x
        dy = p2.y - p1.y
        len_sq = dx * dx + dy * dy

        if len_sq < 1e-10:
            dist = math.hypot(x - p1.x, y - p1.y)
            if dist <= threshold:
                return (p1.x, p1.y)
            return None

        t = ((x - p1.x) * dx + (y - p1.y) * dy) / len_sq
        t = max(0.0, min(1.0, t))

        nearest_x = p1.x + t * dx
        nearest_y = p1.y + t * dy

        dist = math.hypot(x - nearest_x, y - nearest_y)
        if dist <= threshold:
            return (nearest_x, nearest_y)
        return None

    def _nearest_point_on_arc(
        self,
        arc: Arc,
        registry: "EntityRegistry",
        x: float,
        y: float,
        threshold: float,
    ) -> Optional[GeoPoint]:
        center = registry.get_point(arc.center_idx)
        start = registry.get_point(arc.start_idx)
        if not center or not start:
            return None

        radius = math.hypot(start.x - center.x, start.y - center.y)
        if radius < 1e-10:
            return None

        dist_to_center = math.hypot(x - center.x, y - center.y)
        if dist_to_center < 1e-10:
            angle = 0.0
        else:
            angle = math.atan2(y - center.y, x - center.x)

        if not arc.is_angle_within_sweep(angle, registry):
            return None

        nearest_x = center.x + radius * math.cos(angle)
        nearest_y = center.y + radius * math.sin(angle)

        dist = math.hypot(x - nearest_x, y - nearest_y)
        if dist <= threshold:
            return (nearest_x, nearest_y)
        return None

    def _nearest_point_on_circle(
        self,
        circle: Circle,
        registry: "EntityRegistry",
        x: float,
        y: float,
        threshold: float,
    ) -> Optional[GeoPoint]:
        center = registry.get_point(circle.center_idx)
        radius_pt = registry.get_point(circle.radius_pt_idx)
        if not center or not radius_pt:
            return None

        radius = math.hypot(radius_pt.x - center.x, radius_pt.y - center.y)
        if radius < 1e-10:
            return None

        dist_to_center = math.hypot(x - center.x, y - center.y)
        if dist_to_center < 1e-10:
            angle = 0.0
        else:
            angle = math.atan2(y - center.y, x - center.x)

        nearest_x = center.x + radius * math.cos(angle)
        nearest_y = center.y + radius * math.sin(angle)

        dist = math.hypot(x - nearest_x, y - nearest_y)
        if dist <= threshold:
            return (nearest_x, nearest_y)
        return None
