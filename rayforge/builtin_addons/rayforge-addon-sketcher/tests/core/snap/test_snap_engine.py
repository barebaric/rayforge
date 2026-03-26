import pytest

from sketcher.core.snap.types import (
    SnapLine,
    SnapPoint,
    SnapLineType,
    DragContext,
)
from sketcher.core.snap.engine import SnapLineProducer, SnapEngine
from sketcher.core.registry import EntityRegistry


class MockSnapLineProducer(SnapLineProducer):
    """Mock producer for testing SnapEngine."""

    def __init__(self, snap_lines=None, snap_points=None):
        self.snap_lines = snap_lines or []
        self.snap_points = snap_points or []
        self.produce_called = False
        self.produce_points_called = False

    def produce(
        self,
        registry: EntityRegistry,
        drag_position,
        drag_context: DragContext,
        threshold: float,
    ):
        self.produce_called = True
        return iter(self.snap_lines)

    def produce_points(
        self,
        registry: EntityRegistry,
        drag_position,
        drag_context: DragContext,
        threshold: float,
    ):
        self.produce_points_called = True
        return iter(self.snap_points)


@pytest.fixture
def registry():
    """Create a basic entity registry for testing."""
    return EntityRegistry()


@pytest.fixture
def engine():
    """Create a SnapEngine with default threshold."""
    return SnapEngine()


@pytest.fixture
def drag_context():
    """Create an empty drag context."""
    return DragContext()


def test_snap_engine_initialization_defaults():
    """Tests SnapEngine initialization with defaults."""
    engine = SnapEngine()
    assert engine.threshold == SnapEngine.DEFAULT_THRESHOLD
    assert engine.enabled is True
    assert len(engine._producers) == 0
    assert len(engine._cached_points) == 0
    assert engine._last_query_pos is None


def test_snap_engine_initialization_custom_threshold():
    """Tests SnapEngine initialization with custom threshold."""
    engine = SnapEngine(threshold=10.0)
    assert engine.threshold == 10.0


def test_snap_engine_register_producer(engine):
    """Tests registering a snap line producer."""
    producer = MockSnapLineProducer()
    engine.register_producer(producer)
    assert len(engine._producers) == 1
    assert producer in engine._producers


def test_snap_engine_unregister_producer(engine):
    """Tests unregistering a snap line producer."""
    producer = MockSnapLineProducer()
    engine.register_producer(producer)
    assert len(engine._producers) == 1

    engine.unregister_producer(producer)
    assert len(engine._producers) == 0


def test_snap_engine_unregister_nonexistent_producer(engine):
    """Tests unregistering a producer that doesn't exist."""
    producer = MockSnapLineProducer()
    engine.unregister_producer(producer)
    assert len(engine._producers) == 0


def test_snap_engine_clear_producers(engine):
    """Tests clearing all producers."""
    producer1 = MockSnapLineProducer()
    producer2 = MockSnapLineProducer()
    engine.register_producer(producer1)
    engine.register_producer(producer2)
    assert len(engine._producers) == 2

    engine.clear_producers()
    assert len(engine._producers) == 0


def test_snap_engine_enabled_property(engine):
    """Tests the enabled property."""
    assert engine.enabled is True
    engine.enabled = False
    assert engine.enabled is False
    engine.enabled = True
    assert engine.enabled is True


def test_snap_engine_threshold_property(engine):
    """Tests the threshold property."""
    assert engine.threshold == SnapEngine.DEFAULT_THRESHOLD
    engine.threshold = 15.0
    assert engine.threshold == 15.0


def test_snap_engine_query_disabled(engine, registry):
    """Tests that query returns no_snap when engine is disabled."""
    engine.enabled = False
    result = engine.query(registry, (10.0, 20.0))
    assert result.snapped is False
    assert result.position == (10.0, 20.0)


def test_snap_engine_query_with_producer(engine, registry, drag_context):
    """Tests query with a registered producer."""
    snap_line = SnapLine(
        is_horizontal=True,
        coordinate=10.0,
        line_type=SnapLineType.ENTITY_POINT,
    )
    producer = MockSnapLineProducer(snap_lines=[snap_line])
    engine.register_producer(producer)

    result = engine.query(registry, (5.0, 12.0), drag_context)
    assert result.snapped is True
    assert result.position == (5.0, 10.0)
    assert producer.produce_called is True


def test_snap_engine_query_with_snap_point(engine, registry, drag_context):
    """Tests query with a snap point from producer."""
    snap_point = SnapPoint(x=10.0, y=20.0, line_type=SnapLineType.ENTITY_POINT)
    producer = MockSnapLineProducer(snap_points=[snap_point])
    engine.register_producer(producer)

    result = engine.query(registry, (12.0, 22.0), drag_context)
    assert result.snapped is True
    assert result.position == (10.0, 20.0)
    assert result.primary_snap_point == snap_point


def test_snap_engine_query_no_snap(engine, registry, drag_context):
    """Tests query when no snap is found."""
    producer = MockSnapLineProducer()
    engine.register_producer(producer)

    result = engine.query(registry, (100.0, 200.0), drag_context)
    assert result.snapped is False
    assert result.position == (100.0, 200.0)


def test_snap_engine_query_default_drag_context(engine, registry):
    """Tests query with default drag context."""
    snap_point = SnapPoint(x=10.0, y=20.0, line_type=SnapLineType.ENTITY_POINT)
    producer = MockSnapLineProducer(snap_points=[snap_point])
    engine.register_producer(producer)

    result = engine.query(registry, (12.0, 22.0))
    assert result.snapped is True


def test_snap_engine_rebuild_index(engine, registry, drag_context):
    """Tests rebuild_index clears and rebuilds the snap line index."""
    snap_line = SnapLine(
        is_horizontal=True,
        coordinate=10.0,
        line_type=SnapLineType.ENTITY_POINT,
    )
    snap_point = SnapPoint(x=5.0, y=15.0, line_type=SnapLineType.CENTER)
    producer = MockSnapLineProducer(
        snap_lines=[snap_line], snap_points=[snap_point]
    )
    engine.register_producer(producer)

    engine.rebuild_index(registry, (5.0, 12.0), drag_context)

    assert len(engine._index._horizontal) == 1
    assert len(engine._cached_points) == 1
    assert engine._last_query_pos == (5.0, 12.0)


def test_snap_engine_multiple_producers(engine, registry, drag_context):
    """Tests that multiple producers are all used."""
    snap_line1 = SnapLine(
        is_horizontal=True,
        coordinate=10.0,
        line_type=SnapLineType.ENTITY_POINT,
    )
    snap_line2 = SnapLine(
        is_horizontal=False, coordinate=20.0, line_type=SnapLineType.CENTER
    )
    snap_point = SnapPoint(x=15.0, y=25.0, line_type=SnapLineType.MIDPOINT)

    producer1 = MockSnapLineProducer(
        snap_lines=[snap_line1], snap_points=[snap_point]
    )
    producer2 = MockSnapLineProducer(snap_lines=[snap_line2])
    engine.register_producer(producer1)
    engine.register_producer(producer2)

    engine.rebuild_index(registry, (15.0, 25.0), drag_context)

    assert producer1.produce_called is True
    assert producer2.produce_called is True
    assert len(engine._index._horizontal) == 1
    assert len(engine._index._vertical) == 1
    assert len(engine._cached_points) == 1


def test_snap_engine_query_caches_snap_points(engine, registry, drag_context):
    """Tests that snap points are cached during query."""
    snap_point = SnapPoint(x=10.0, y=20.0, line_type=SnapLineType.ENTITY_POINT)
    producer = MockSnapLineProducer(snap_points=[snap_point])
    engine.register_producer(producer)

    result1 = engine.query(registry, (12.0, 22.0), drag_context)
    result2 = engine.query(registry, (12.0, 22.0), drag_context)

    assert result1.primary_snap_point == result2.primary_snap_point


def test_snap_engine_query_different_positions(engine, registry, drag_context):
    """Tests that query rebuilds index for different positions."""
    snap_point1 = SnapPoint(
        x=10.0, y=20.0, line_type=SnapLineType.ENTITY_POINT
    )
    snap_point2 = SnapPoint(x=30.0, y=40.0, line_type=SnapLineType.CENTER)

    class DynamicProducer(MockSnapLineProducer):
        def produce_points(
            self, registry, drag_position, drag_context, threshold
        ):
            if drag_position[0] < 20.0:
                return iter([snap_point1])
            else:
                return iter([snap_point2])

    producer = DynamicProducer()
    engine.register_producer(producer)

    result1 = engine.query(registry, (12.0, 22.0), drag_context)
    assert result1.primary_snap_point == snap_point1

    result2 = engine.query(registry, (32.0, 42.0), drag_context)
    assert result2.primary_snap_point == snap_point2


def test_snap_engine_get_visible_snap_lines(engine, registry, drag_context):
    """Tests get_visible_snap_lines method."""
    snap_line1 = SnapLine(
        is_horizontal=True,
        coordinate=10.0,
        line_type=SnapLineType.ENTITY_POINT,
    )
    snap_line2 = SnapLine(
        is_horizontal=False, coordinate=20.0, line_type=SnapLineType.CENTER
    )
    producer = MockSnapLineProducer(snap_lines=[snap_line1, snap_line2])
    engine.register_producer(producer)

    visible_lines = engine.get_visible_snap_lines(
        registry, (5.0, 12.0), drag_context
    )

    assert len(visible_lines) == 2
    assert snap_line1 in visible_lines
    assert snap_line2 in visible_lines


def test_snap_engine_get_visible_snap_lines_disabled(engine, registry):
    """Tests that get_visible_snap_lines returns empty when disabled."""
    engine.enabled = False
    visible_lines = engine.get_visible_snap_lines(registry, (5.0, 12.0))
    assert visible_lines == []


def test_snap_engine_find_nearest_snap_point(engine, registry, drag_context):
    """Tests finding nearest snap point."""
    snap_point1 = SnapPoint(
        x=10.0, y=20.0, line_type=SnapLineType.ENTITY_POINT
    )
    snap_point2 = SnapPoint(x=15.0, y=25.0, line_type=SnapLineType.CENTER)

    producer = MockSnapLineProducer(snap_points=[snap_point1, snap_point2])
    engine.register_producer(producer)

    engine.rebuild_index(registry, (12.0, 22.0), drag_context)
    result = engine.query(registry, (12.0, 22.0), drag_context)

    assert result.primary_snap_point == snap_point1


def test_snap_engine_snap_point_priority(engine, registry, drag_context):
    """Tests that higher priority snap points are preferred."""
    snap_point_low = SnapPoint(x=12.0, y=22.0, line_type=SnapLineType.CENTER)
    snap_point_high = SnapPoint(
        x=13.0, y=23.0, line_type=SnapLineType.ENTITY_POINT
    )

    producer = MockSnapLineProducer(
        snap_points=[snap_point_low, snap_point_high]
    )
    engine.register_producer(producer)

    result = engine.query(registry, (12.5, 22.5), drag_context)

    assert result.primary_snap_point == snap_point_high


def test_snap_engine_horizontal_vertical_snap(engine, registry, drag_context):
    """Tests snapping to both horizontal and vertical lines."""
    h_line = SnapLine(
        is_horizontal=True,
        coordinate=10.0,
        line_type=SnapLineType.ENTITY_POINT,
    )
    v_line = SnapLine(
        is_horizontal=False, coordinate=20.0, line_type=SnapLineType.CENTER
    )

    producer = MockSnapLineProducer(snap_lines=[h_line, v_line])
    engine.register_producer(producer)

    result = engine.query(registry, (22.0, 12.0), drag_context)

    assert result.snapped is True
    assert result.position == (20.0, 10.0)
    assert len(result.snap_lines) == 2


def test_snap_line_producer_abstract():
    """Tests that SnapLineProducer raises NotImplementedError."""
    producer = SnapLineProducer()
    registry = EntityRegistry()
    drag_context = DragContext()

    with pytest.raises(NotImplementedError):
        list(producer.produce(registry, (0.0, 0.0), drag_context, 5.0))


def test_snap_line_producer_produce_points_default():
    """Tests that produce_points returns empty iterator by default."""
    producer = SnapLineProducer()
    registry = EntityRegistry()
    drag_context = DragContext()

    points = list(
        producer.produce_points(registry, (0.0, 0.0), drag_context, 5.0)
    )
    assert points == []


def test_snap_engine_find_crossing_lines(engine, registry, drag_context):
    """Tests finding crossing lines for a snap point."""
    h_line = SnapLine(
        is_horizontal=True,
        coordinate=10.0,
        line_type=SnapLineType.ENTITY_POINT,
    )
    v_line = SnapLine(
        is_horizontal=False, coordinate=20.0, line_type=SnapLineType.CENTER
    )
    snap_point = SnapPoint(x=20.0, y=10.0, line_type=SnapLineType.ENTITY_POINT)

    producer = MockSnapLineProducer(
        snap_lines=[h_line, v_line], snap_points=[snap_point]
    )
    engine.register_producer(producer)

    result = engine.query(registry, (20.0, 10.0), drag_context)

    assert result.primary_snap_point == snap_point
    assert len(result.snap_lines) == 2
    assert h_line in result.snap_lines
    assert v_line in result.snap_lines
