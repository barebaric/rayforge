from typing import Iterator, Optional, TYPE_CHECKING

from rayforge.core.geo.types import Point as GeoPoint
from ...entities import Arc, Circle, Ellipse
from ..types import SnapLine, SnapPoint, SnapLineType, DragContext
from ..engine import SnapLineProducer

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
        x, y = drag_position
        for entity in registry.entities:
            if drag_context.is_entity_dragged(entity.id):
                continue
            if not self._include_construction and entity.construction:
                continue

            center = self._get_center(entity, registry)
            if center is None:
                continue

            cx, cy = center
            if abs(cx - x) <= threshold:
                yield SnapLine(
                    is_horizontal=False,
                    coordinate=cx,
                    line_type=SnapLineType.CENTER,
                    source=entity,
                )
            if abs(cy - y) <= threshold:
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

            center = self._get_center(entity, registry)
            if center is None:
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
    ) -> Optional[GeoPoint]:
        if isinstance(entity, (Arc, Circle, Ellipse)):
            center = registry.get_point(entity.center_idx)
            if center:
                return (center.x, center.y)
        return None
