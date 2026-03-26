import math
from typing import Iterator, TYPE_CHECKING

from rayforge.core.geo.types import Point as GeoPoint
from ..types import SnapLine, SnapPoint, SnapLineType, DragContext
from ..engine import SnapLineProducer

if TYPE_CHECKING:
    from ...registry import EntityRegistry
    from ...entities import Line, Arc


class MidpointsProducer(SnapLineProducer):
    def produce(
        self,
        registry: "EntityRegistry",
        drag_position: GeoPoint,
        drag_context: DragContext,
        threshold: float,
    ) -> Iterator[SnapLine]:
        x, y = drag_position
        for entity in registry.entities:
            if drag_context.is_entity_dragged(entity.id):
                continue

            mid = self._get_midpoint(entity, registry)
            if mid is None:
                continue

            mx, my = mid
            if abs(mx - x) <= threshold:
                yield SnapLine(
                    is_horizontal=False,
                    coordinate=mx,
                    line_type=SnapLineType.MIDPOINT,
                    source=entity,
                )
            if abs(my - y) <= threshold:
                yield SnapLine(
                    is_horizontal=True,
                    coordinate=my,
                    line_type=SnapLineType.MIDPOINT,
                    source=entity,
                )

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
        self, entity, registry: "EntityRegistry"
    ) -> GeoPoint | None:
        from ...entities import Line, Arc

        if isinstance(entity, Line):
            return self._line_midpoint(entity, registry)
        elif isinstance(entity, Arc):
            return self._arc_midpoint(entity, registry)
        return None

    def _line_midpoint(
        self, line: "Line", registry: "EntityRegistry"
    ) -> GeoPoint | None:
        p1 = registry.get_point(line.p1_idx)
        p2 = registry.get_point(line.p2_idx)
        if p1 and p2:
            return ((p1.x + p2.x) / 2, (p1.y + p2.y) / 2)
        return None

    def _arc_midpoint(
        self, arc: "Arc", registry: "EntityRegistry"
    ) -> GeoPoint | None:
        midpoint = arc.get_midpoint(registry)
        if midpoint:
            return midpoint
        return None
