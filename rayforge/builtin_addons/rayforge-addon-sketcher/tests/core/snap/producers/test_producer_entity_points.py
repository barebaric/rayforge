import pytest

from sketcher.core.snap.types import SnapLineType, DragContext
from sketcher.core.snap.producers.entity_points import EntityPointsProducer
from sketcher.core.registry import EntityRegistry


@pytest.fixture
def registry():
    """Create a basic entity registry for testing."""
    return EntityRegistry()


@pytest.fixture
def drag_context():
    """Create an empty drag context."""
    return DragContext()


@pytest.fixture
def producer():
    """Create an EntityPointsProducer for testing."""
    return EntityPointsProducer()


def test_entity_points_producer_produce_horizontal_lines(
    producer, registry, drag_context
):
    """Tests producing horizontal snap lines from entity points."""
    registry.add_point(10.0, 20.0)
    registry.add_point(30.0, 20.0)
    registry.add_point(50.0, 40.0)

    drag_position = (15.0, 18.0)
    threshold = 5.0

    snap_lines = list(
        producer.produce(registry, drag_position, drag_context, threshold)
    )

    assert len(snap_lines) == 3
    horizontal_lines = [sl for sl in snap_lines if sl.is_horizontal]
    assert len(horizontal_lines) == 2
    assert all(
        sl.line_type == SnapLineType.ENTITY_POINT for sl in horizontal_lines
    )
    assert all(sl.coordinate == 20.0 for sl in horizontal_lines)


def test_entity_points_producer_produce_vertical_lines(
    producer, registry, drag_context
):
    """Tests producing vertical snap lines from entity points."""
    registry.add_point(10.0, 20.0)
    registry.add_point(10.0, 40.0)
    registry.add_point(30.0, 50.0)

    drag_position = (12.0, 30.0)
    threshold = 5.0

    snap_lines = list(
        producer.produce(registry, drag_position, drag_context, threshold)
    )

    assert len(snap_lines) == 2
    assert all(not sl.is_horizontal for sl in snap_lines)
    assert all(sl.line_type == SnapLineType.ENTITY_POINT for sl in snap_lines)
    assert all(sl.coordinate == 10.0 for sl in snap_lines)


def test_entity_points_producer_produce_both_axes(
    producer, registry, drag_context
):
    """Tests producing both horizontal and vertical snap lines."""
    registry.add_point(10.0, 20.0)

    drag_position = (12.0, 18.0)
    threshold = 5.0

    snap_lines = list(
        producer.produce(registry, drag_position, drag_context, threshold)
    )

    assert len(snap_lines) == 2
    horizontal_lines = [sl for sl in snap_lines if sl.is_horizontal]
    vertical_lines = [sl for sl in snap_lines if not sl.is_horizontal]
    assert len(horizontal_lines) == 1
    assert len(vertical_lines) == 1
    assert horizontal_lines[0].coordinate == 20.0
    assert vertical_lines[0].coordinate == 10.0


def test_entity_points_producer_outside_threshold(
    producer, registry, drag_context
):
    """Tests that points outside threshold don't produce snap lines."""
    registry.add_point(10.0, 20.0)
    registry.add_point(100.0, 200.0)

    drag_position = (12.0, 18.0)
    threshold = 5.0

    snap_lines = list(
        producer.produce(registry, drag_position, drag_context, threshold)
    )

    assert len(snap_lines) == 2


def test_entity_points_produce_points(producer, registry, drag_context):
    """Tests producing snap points from entity points."""
    registry.add_point(10.0, 20.0)
    registry.add_point(12.0, 22.0)
    registry.add_point(100.0, 200.0)

    drag_position = (11.0, 21.0)
    threshold = 5.0

    snap_points = list(
        producer.produce_points(
            registry, drag_position, drag_context, threshold
        )
    )

    assert len(snap_points) == 2
    assert all(sp.line_type == SnapLineType.ENTITY_POINT for sp in snap_points)
    assert (10.0, 20.0) in [(sp.x, sp.y) for sp in snap_points]
    assert (12.0, 22.0) in [(sp.x, sp.y) for sp in snap_points]


def test_entity_points_producer_dragged_point_excluded(
    producer, registry, drag_context
):
    """Tests that dragged points are excluded from snap generation."""
    p1_id = registry.add_point(10.0, 20.0)
    registry.add_point(12.0, 18.0)

    drag_context.dragged_point_ids.add(p1_id)

    drag_position = (11.0, 19.0)
    threshold = 5.0

    snap_lines = list(
        producer.produce(registry, drag_position, drag_context, threshold)
    )

    assert len(snap_lines) == 2

    snap_points = list(
        producer.produce_points(
            registry, drag_position, drag_context, threshold
        )
    )

    assert len(snap_points) == 1
    assert snap_points[0].x == 12.0
    assert snap_points[0].y == 18.0


def test_entity_points_producer_source_attribute(
    producer, registry, drag_context
):
    """Tests that source attribute is set to the point entity."""
    point_id = registry.add_point(10.0, 20.0)
    point = registry.get_point(point_id)

    drag_position = (12.0, 18.0)
    threshold = 5.0

    snap_lines = list(
        producer.produce(registry, drag_position, drag_context, threshold)
    )

    assert all(sl.source == point for sl in snap_lines)

    snap_points = list(
        producer.produce_points(
            registry, drag_position, drag_context, threshold
        )
    )

    assert all(sp.source == point for sp in snap_points)


def test_entity_points_producer_empty_registry(
    producer, registry, drag_context
):
    """Tests producer with empty registry."""
    drag_position = (10.0, 20.0)
    threshold = 5.0

    snap_lines = list(
        producer.produce(registry, drag_position, drag_context, threshold)
    )
    assert len(snap_lines) == 0

    snap_points = list(
        producer.produce_points(
            registry, drag_position, drag_context, threshold
        )
    )
    assert len(snap_points) == 0


def test_entity_points_producer_exact_match(producer, registry, drag_context):
    """Tests snap generation when drag position exactly matches point."""
    registry.add_point(10.0, 20.0)

    drag_position = (10.0, 20.0)
    threshold = 0.1

    snap_lines = list(
        producer.produce(registry, drag_position, drag_context, threshold)
    )

    assert len(snap_lines) == 2

    snap_points = list(
        producer.produce_points(
            registry, drag_position, drag_context, threshold
        )
    )

    assert len(snap_points) == 1
    assert snap_points[0].x == 10.0
    assert snap_points[0].y == 20.0


def test_entity_points_producer_multiple_points(
    producer, registry, drag_context
):
    """Tests producer with multiple points at different locations."""
    registry.add_point(10.0, 20.0)
    registry.add_point(30.0, 20.0)
    registry.add_point(10.0, 40.0)
    registry.add_point(30.0, 40.0)

    drag_position = (15.0, 25.0)
    threshold = 10.0

    snap_lines = list(
        producer.produce(registry, drag_position, drag_context, threshold)
    )

    assert len(snap_lines) == 4

    snap_points = list(
        producer.produce_points(
            registry, drag_position, drag_context, threshold
        )
    )

    assert len(snap_points) == 1


def test_entity_points_producer_threshold_boundary(
    producer, registry, drag_context
):
    """Tests snap generation at threshold boundary."""
    registry.add_point(10.0, 20.0)

    drag_position = (15.0, 20.0)
    threshold = 5.0

    snap_lines = list(
        producer.produce(registry, drag_position, drag_context, threshold)
    )

    assert len(snap_lines) == 2

    drag_position = (15.1, 20.0)
    snap_lines = list(
        producer.produce(registry, drag_position, drag_context, threshold)
    )

    assert len(snap_lines) == 1


def test_entity_points_producer_negative_coordinates(
    producer, registry, drag_context
):
    """Tests snap generation with negative coordinates."""
    registry.add_point(-10.0, -20.0)
    registry.add_point(10.0, 20.0)

    drag_position = (-8.0, -18.0)
    threshold = 5.0

    snap_lines = list(
        producer.produce(registry, drag_position, drag_context, threshold)
    )

    assert len(snap_lines) == 2

    snap_points = list(
        producer.produce_points(
            registry, drag_position, drag_context, threshold
        )
    )

    assert len(snap_points) == 1
    assert snap_points[0].x == -10.0
    assert snap_points[0].y == -20.0


def test_entity_points_producer_all_dragged(producer, registry, drag_context):
    """Tests producer when all points are dragged."""
    p1_id = registry.add_point(10.0, 20.0)
    p2_id = registry.add_point(30.0, 40.0)

    drag_context.dragged_point_ids.update([p1_id, p2_id])

    drag_position = (15.0, 25.0)
    threshold = 5.0

    snap_lines = list(
        producer.produce(registry, drag_position, drag_context, threshold)
    )

    assert len(snap_lines) == 0

    snap_points = list(
        producer.produce_points(
            registry, drag_position, drag_context, threshold
        )
    )

    assert len(snap_points) == 0
