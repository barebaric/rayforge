from collections.abc import Iterator

from raygeo.geo.types import Point as GeoPoint
from raygeo.sketch.registry import EntityRegistry
from raygeo.sketch.snap.engine import SnapEngine, SnapLineProducer
from raygeo.sketch.snap.producers import EntityPointsProducer
from raygeo.sketch.snap.types import DragContext, SnapLine, SnapPoint


class _CountingProducer(SnapLineProducer):
    produce_calls: int

    def __init__(self) -> None:
        self.produce_calls = 0

    def produce(
        self,
        registry: EntityRegistry,
        drag_position: GeoPoint,
        drag_context: DragContext,
        threshold: float,
    ) -> Iterator[SnapLine]:
        self.produce_calls += 1
        return iter(())

    def produce_points(
        self,
        registry: EntityRegistry,
        drag_position: GeoPoint,
        drag_context: DragContext,
        threshold: float,
    ) -> Iterator[SnapPoint]:
        return iter(())


def test_preview_movement_does_not_rebuild_static_index():
    registry = EntityRegistry()
    p1 = registry.add_point(0.0, 0.0)
    p2 = registry.add_point(10.0, 0.0)
    registry.add_line(p1, p2)

    p3 = registry.add_point(0.0, 10.0)
    p4 = registry.add_point(10.0, 10.0)
    preview_entity = registry.add_line(p3, p4)

    engine = SnapEngine()
    counting = _CountingProducer()
    engine.register_producer(counting)
    context = DragContext(preview_entity_ids={preview_entity})

    engine.query(registry, (0.0, 0.0), context)
    assert counting.produce_calls == 1  # type: ignore[attr-defined]

    registry.get_point(p3).x = 5.0
    engine.query(registry, (0.0, 0.0), context)

    assert counting.produce_calls == 1  # type: ignore[attr-defined]


def test_preview_points_still_produce_live_axis_lines():
    registry = EntityRegistry()
    p1 = registry.add_point(0.0, 0.0)
    p2 = registry.add_point(10.0, 0.0)
    preview_entity = registry.add_line(p1, p2)

    engine = SnapEngine()
    context = DragContext(
        dragged_point_ids={p2},
        preview_entity_ids={preview_entity},
    )

    lines = engine.get_visible_snap_lines(registry, (0.0, 0.0), context)

    assert any(
        not line.is_horizontal and line.coordinate == 0.0 for line in lines
    )
    assert any(line.is_horizontal and line.coordinate == 0.0 for line in lines)
    assert not any(line.coordinate == 10.0 for line in lines), (
        "dragged preview point must not produce guides"
    )


def test_static_snap_points_survive_preview_exclusion():
    registry = EntityRegistry()
    registry.add_point(100.0, 100.0)
    p1 = registry.add_point(0.0, 0.0)
    p2 = registry.add_point(10.0, 0.0)
    preview_entity = registry.add_line(p1, p2)

    engine = SnapEngine()
    engine.register_producer(EntityPointsProducer())
    context = DragContext(preview_entity_ids={preview_entity})

    lines = engine.get_visible_snap_lines(registry, (0.0, 0.0), context)
    coordinates = {line.coordinate for line in lines}

    assert 100.0 in coordinates
