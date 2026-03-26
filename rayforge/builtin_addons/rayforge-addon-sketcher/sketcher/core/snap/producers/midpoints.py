from typing import Iterator, Optional, TYPE_CHECKING

from rayforge.core.geo.types import Point as GeoPoint
from ...entities import Line, Arc
from ..types import SnapLine, SnapPoint, SnapLineType, DragContext
from ..engine import SnapLineProducer

if TYPE_CHECKING:
    from ...registry import EntityRegistry


class MidpointsProducer(SnapLineProducer):
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

            mid = self._get_midpoint(entity, registry)
            if mid is None:
                continue

            mx, my = mid
            dist = ((mx - x) ** 2 + (my - y) ** 2) ** 0.5
            if dist <= threshold:
                yield SnapPoint(
                    x=mx,
                    y=my,
                    line_type=SnapLineType.MIDPOINT,
                    source=entity,
                )

    def _get_midpoint(
        self, entity: object, registry: "EntityRegistry"
    ) -> Optional[GeoPoint]:
        if isinstance(entity, Line):
            return self._line_midpoint(entity, registry)
        elif isinstance(entity, Arc):
            return self._arc_midpoint(entity, registry)
        return None

    def _line_midpoint(
        self, line: Line, registry: "EntityRegistry"
    ) -> Optional[GeoPoint]:
        p1 = registry.get_point(line.p1_idx)
        p2 = registry.get_point(line.p2_idx)
        if p1 and p2:
            return ((p1.x + p2.x) / 2, (p1.y + p2.y) / 2)
        return None

    def _arc_midpoint(
        self, arc: Arc, registry: "EntityRegistry"
    ) -> Optional[GeoPoint]:
        midpoint = arc.get_midpoint(registry)
        if midpoint:
            return (midpoint[0], midpoint[1])
        return None
