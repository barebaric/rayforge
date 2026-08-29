from collections.abc import Iterator
from typing import TYPE_CHECKING

from raygeo.geo.types import Point as GeoPoint

from ..engine import SnapLineProducer
from ..types import DragContext, SnapLine, SnapLineType, SnapPoint

if TYPE_CHECKING:
    from ...registry import EntityRegistry


class EntityPointsProducer(SnapLineProducer):
    def produce(
        self,
        registry: "EntityRegistry",
        drag_position: GeoPoint,
        drag_context: DragContext,
        threshold: float,
    ) -> Iterator[SnapLine]:
        for point in registry.points:
            if drag_context.is_point_dragged(point.id):
                continue
            if drag_context.coincides_with_dragged(point.x, point.y, registry):
                continue

            px, py = point.x, point.y
            yield SnapLine(
                is_horizontal=False,
                coordinate=px,
                line_type=SnapLineType.ENTITY_POINT,
                source=point,
            )
            yield SnapLine(
                is_horizontal=True,
                coordinate=py,
                line_type=SnapLineType.ENTITY_POINT,
                source=point,
            )

    def produce_points(
        self,
        registry: "EntityRegistry",
        drag_position: GeoPoint,
        drag_context: DragContext,
        threshold: float,
    ) -> Iterator[SnapPoint]:
        x, y = drag_position
        for point in registry.points:
            if drag_context.is_point_dragged(point.id):
                continue
            if drag_context.coincides_with_dragged(point.x, point.y, registry):
                continue

            px, py = point.x, point.y
            dist = ((px - x) ** 2 + (py - y) ** 2) ** 0.5
            if dist <= threshold:
                yield SnapPoint(
                    x=px,
                    y=py,
                    line_type=SnapLineType.ENTITY_POINT,
                    source=point,
                )
