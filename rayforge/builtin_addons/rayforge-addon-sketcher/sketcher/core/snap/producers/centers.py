from collections.abc import Iterator
from typing import TYPE_CHECKING

from raygeo.geo.types import Point as GeoPoint

from ...entities import Arc, Circle, Ellipse
from ..engine import SnapLineProducer
from ..types import DragContext, SnapLine, SnapLineType, SnapPoint

if TYPE_CHECKING:
    from ...registry import EntityRegistry


class CentersProducer(SnapLineProducer):
    def __init__(self, include_construction: bool = True) -> None:
        self._include_construction: bool = include_construction

    def produce(
        self,
        registry: "EntityRegistry",
        drag_position: GeoPoint,
        drag_context: DragContext,
        threshold: float,
    ) -> Iterator[SnapLine]:
        for entity in registry.entities:
            if drag_context.is_entity_dragged(entity.id):
                continue
            if not self._include_construction and entity.construction:
                continue

            if not isinstance(entity, (Arc, Circle, Ellipse)):
                continue

            if drag_context.is_point_dragged(entity.center_idx):
                continue

            center = self._get_center(entity, registry)
            if center is None:
                continue
            if drag_context.coincides_with_dragged(
                center[0], center[1], registry
            ):
                continue

            cx, cy = center
            yield SnapLine(
                is_horizontal=False,
                coordinate=cx,
                line_type=SnapLineType.CENTER,
                source=entity,
            )
            yield SnapLine(
                is_horizontal=True,
                coordinate=cy,
                line_type=SnapLineType.CENTER,
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
            if not self._include_construction and entity.construction:
                continue

            if not isinstance(entity, (Arc, Circle, Ellipse)):
                continue

            if drag_context.is_point_dragged(entity.center_idx):
                continue

            center = self._get_center(entity, registry)
            if center is None:
                continue
            if drag_context.coincides_with_dragged(
                center[0], center[1], registry
            ):
                continue

            cx, cy = center
            dist = ((cx - x) ** 2 + (cy - y) ** 2) ** 0.5
            if dist <= threshold:
                yield SnapPoint(
                    x=cx,
                    y=cy,
                    line_type=SnapLineType.CENTER,
                    source=entity,
                )

    def _get_center(
        self, entity: object, registry: "EntityRegistry"
    ) -> GeoPoint | None:
        if isinstance(entity, (Arc, Circle, Ellipse)):
            center = registry.get_point(entity.center_idx)
            if center:
                return (center.x, center.y)
        return None
