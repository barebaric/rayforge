import pytest
from sketcher.core.registry import EntityRegistry
from sketcher.core.snap.engine import SnapEngine, SnapLineProducer
from sketcher.core.snap.producers.intersections import IntersectionsProducer
from sketcher.core.snap.types import DragContext


class CountingProducer(SnapLineProducer):
    """Records how often the index was rebuilt from it."""

    def __init__(self):
        self.produce_calls = 0

    def produce(
        self,
        registry: EntityRegistry,
        drag_position,
        drag_context: DragContext,
        threshold: float,
    ):
        self.produce_calls += 1
        return iter(())


def build_crossing_lines():
    """Two non-dragged lines crossing at (5, 5) plus a dragged line."""
    registry = EntityRegistry()
    a1 = registry.add_point(100, 100)
    a2 = registry.add_point(110, 110)
    b1 = registry.add_point(0, 10)
    b2 = registry.add_point(10, 0)
    c1 = registry.add_point(0, 0)
    c2 = registry.add_point(10, 10)
    line_a = registry.add_line(a1, a2)
    registry.add_line(b1, b2)
    registry.add_line(c1, c2)
    drag_context = DragContext(
        dragged_point_ids={a1, a2},
        dragged_entity_ids={line_a},
    )
    return registry, drag_context


def test_index_not_rebuilt_while_only_dragged_points_move():
    registry = EntityRegistry()
    dragged = registry.add_point(0, 0)
    static = registry.add_point(50, 50)
    engine = SnapEngine()
    producer = CountingProducer()
    engine.register_producer(producer)
    drag_context = DragContext(dragged_point_ids={dragged})

    engine.query(registry, (0.0, 0.0), drag_context)
    assert producer.produce_calls == 1

    point = registry.get_point(dragged)
    point.x = 1.0
    point.y = 1.0
    engine.query(registry, (1.0, 1.0), drag_context)
    assert producer.produce_calls == 1

    registry.get_point(static).x = 51.0
    engine.query(registry, (1.0, 1.0), drag_context)
    assert producer.produce_calls == 2


def test_intersections_producer_caches_across_queries():
    registry, drag_context = build_crossing_lines()
    producer = IntersectionsProducer()

    computations = []
    original = producer._get_all_intersections

    def counting(registry, drag_context):
        computations.append(1)
        return original(registry, drag_context)

    producer._get_all_intersections = counting

    def snap_points_at(x, y):
        return [
            (p.x, p.y)
            for p in producer.produce_points(
                registry, (x, y), drag_context, 1.0
            )
        ]

    first = snap_points_at(5.0, 5.0)
    assert first == [(5.0, 5.0)]
    assert len(computations) == 1

    point = registry.get_point(0)
    point.x = -5.0
    point.y = -5.0
    second = snap_points_at(5.0, 5.0)
    assert second == [(5.0, 5.0)]
    assert len(computations) == 1

    registry.get_point(2).y = 20.0
    third = [
        (p.x, p.y)
        for p in producer.produce_points(
            registry, (20.0 / 3.0, 20.0 / 3.0), drag_context, 0.5
        )
    ]
    assert len(computations) == 2
    assert third and third != first
    assert third[0][0] == pytest.approx(20.0 / 3.0, abs=1e-6)


def test_is_snap_result_current_tracks_dragged_movement():
    registry, drag_context = build_crossing_lines()
    engine = SnapEngine()
    engine.register_producer(IntersectionsProducer())

    engine.query(registry, (5.0, 5.0), drag_context)
    assert engine.is_snap_result_current(registry) is True

    registry.get_point(0).x = 2.0
    assert engine.is_snap_result_current(registry) is False

    engine.query(registry, (5.0, 5.0), drag_context)
    assert engine.is_snap_result_current(registry) is True
