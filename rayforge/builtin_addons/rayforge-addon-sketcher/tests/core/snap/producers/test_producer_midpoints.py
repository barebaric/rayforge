import pytest

from sketcher.core.snap.types import SnapLineType, DragContext
from sketcher.core.snap.producers.midpoints import MidpointsProducer
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
    """Create a MidpointsProducer for testing."""
    return MidpointsProducer()


def test_midpoints_producer_no_snap_lines(producer, registry, drag_context):
    """Tests that producer doesn't generate snap lines."""
    p1 = registry.add_point(0.0, 0.0)
    p2 = registry.add_point(10.0, 10.0)
    registry.add_line(p1, p2)

    drag_position = (5.0, 5.0)
    threshold = 5.0

    snap_lines = list(
        producer.produce(registry, drag_position, drag_context, threshold)
    )

    assert len(snap_lines) == 0


def test_midpoints_producer_line_midpoint(producer, registry, drag_context):
    """Tests producing midpoint from a line."""
    p1 = registry.add_point(0.0, 0.0)
    p2 = registry.add_point(10.0, 10.0)
    registry.add_line(p1, p2)

    drag_position = (5.0, 5.0)
    threshold = 1.0

    snap_points = list(
        producer.produce_points(
            registry, drag_position, drag_context, threshold
        )
    )

    assert len(snap_points) == 1
    assert abs(snap_points[0].x - 5.0) < 0.001
    assert abs(snap_points[0].y - 5.0) < 0.001
    assert snap_points[0].line_type == SnapLineType.MIDPOINT


def test_midpoints_producer_arc_midpoint(producer, registry, drag_context):
    """Tests producing midpoint from an arc."""
    center = registry.add_point(0.0, 0.0)
    start = registry.add_point(10.0, 0.0)
    end = registry.add_point(0.0, 10.0)
    registry.add_arc(start, end, center)

    drag_position = (7.07, 7.07)
    threshold = 5.0

    snap_points = list(
        producer.produce_points(
            registry, drag_position, drag_context, threshold
        )
    )

    assert len(snap_points) == 1
    assert abs(snap_points[0].x - 7.07) < 0.1
    assert abs(snap_points[0].y - 7.07) < 0.1
    assert snap_points[0].line_type == SnapLineType.MIDPOINT


def test_midpoints_producer_dragged_entity_excluded(
    producer, registry, drag_context
):
    """Tests that dragged entities are excluded from snap generation."""
    p1 = registry.add_point(0.0, 0.0)
    p2 = registry.add_point(10.0, 10.0)
    line_id = registry.add_line(p1, p2)

    drag_context.dragged_entity_ids.add(line_id)

    drag_position = (5.0, 5.0)
    threshold = 1.0

    snap_points = list(
        producer.produce_points(
            registry, drag_position, drag_context, threshold
        )
    )

    assert len(snap_points) == 0


def test_midpoints_producer_dragged_point_excluded(
    producer, registry, drag_context
):
    """Tests that entities with dragged points are excluded."""
    p1 = registry.add_point(0.0, 0.0)
    p2 = registry.add_point(10.0, 10.0)
    registry.add_line(p1, p2)

    drag_context.dragged_point_ids.add(p1)

    drag_position = (5.0, 5.0)
    threshold = 1.0

    snap_points = list(
        producer.produce_points(
            registry, drag_position, drag_context, threshold
        )
    )

    assert len(snap_points) == 0


def test_midpoints_producer_outside_threshold(
    producer, registry, drag_context
):
    """Tests that midpoints outside threshold don't produce snaps."""
    p1 = registry.add_point(0.0, 0.0)
    p2 = registry.add_point(10.0, 10.0)
    registry.add_line(p1, p2)

    drag_position = (50.0, 50.0)
    threshold = 5.0

    snap_points = list(
        producer.produce_points(
            registry, drag_position, drag_context, threshold
        )
    )

    assert len(snap_points) == 0


def test_midpoints_producer_multiple_lines(producer, registry, drag_context):
    """Tests producer with multiple lines."""
    p1 = registry.add_point(0.0, 0.0)
    p2 = registry.add_point(10.0, 10.0)
    registry.add_line(p1, p2)

    p3 = registry.add_point(20.0, 20.0)
    p4 = registry.add_point(30.0, 30.0)
    registry.add_line(p3, p4)

    drag_position = (5.0, 5.0)
    threshold = 20.0

    snap_points = list(
        producer.produce_points(
            registry, drag_position, drag_context, threshold
        )
    )

    assert len(snap_points) == 1


def test_midpoints_producer_source_attribute(producer, registry, drag_context):
    """Tests that source attribute is set to the entity."""
    p1 = registry.add_point(0.0, 0.0)
    p2 = registry.add_point(10.0, 10.0)
    line_id = registry.add_line(p1, p2)
    line = registry.get_entity(line_id)

    drag_position = (5.0, 5.0)
    threshold = 1.0

    snap_points = list(
        producer.produce_points(
            registry, drag_position, drag_context, threshold
        )
    )

    assert snap_points[0].source == line


def test_midpoints_producer_negative_coordinates(
    producer, registry, drag_context
):
    """Tests snap generation with negative coordinates."""
    p1 = registry.add_point(-10.0, -10.0)
    p2 = registry.add_point(10.0, 10.0)
    registry.add_line(p1, p2)

    drag_position = (0.0, 0.0)
    threshold = 1.0

    snap_points = list(
        producer.produce_points(
            registry, drag_position, drag_context, threshold
        )
    )

    assert len(snap_points) == 1
    assert abs(snap_points[0].x) < 0.001
    assert abs(snap_points[0].y) < 0.001


def test_midpoints_producer_empty_registry(producer, registry, drag_context):
    """Tests producer with empty registry."""
    drag_position = (5.0, 5.0)
    threshold = 5.0

    snap_points = list(
        producer.produce_points(
            registry, drag_position, drag_context, threshold
        )
    )

    assert len(snap_points) == 0


def test_midpoints_producer_line_midpoint_calculation(
    producer, registry, drag_context
):
    """Tests correct midpoint calculation for various lines."""
    test_cases = [
        ((0.0, 0.0), (10.0, 0.0), (5.0, 0.0)),
        ((0.0, 0.0), (0.0, 10.0), (0.0, 5.0)),
        ((-5.0, -5.0), (5.0, 5.0), (0.0, 0.0)),
    ]

    for (x1, y1), (x2, y2), (expected_x, expected_y) in test_cases:
        test_registry = EntityRegistry()
        p1 = test_registry.add_point(x1, y1)
        p2 = test_registry.add_point(x2, y2)
        test_registry.add_line(p1, p2)

        drag_position = (expected_x, expected_y)
        threshold = 1.0

        snap_points = list(
            producer.produce_points(
                test_registry, drag_position, drag_context, threshold
            )
        )

        assert len(snap_points) == 1
        assert abs(snap_points[0].x - expected_x) < 0.001
        assert abs(snap_points[0].y - expected_y) < 0.001


def test_midpoints_producer_exact_match(producer, registry, drag_context):
    """Tests snap generation when drag position exactly matches midpoint."""
    p1 = registry.add_point(0.0, 0.0)
    p2 = registry.add_point(10.0, 10.0)
    registry.add_line(p1, p2)

    drag_position = (5.0, 5.0)
    threshold = 0.1

    snap_points = list(
        producer.produce_points(
            registry, drag_position, drag_context, threshold
        )
    )

    assert len(snap_points) == 1
    assert abs(snap_points[0].x - 5.0) < 0.001
    assert abs(snap_points[0].y - 5.0) < 0.001


def test_midpoints_producer_non_midpoint_entities(
    producer, registry, drag_context
):
    """Tests that non-midpoint entities don't produce snaps."""
    p1 = registry.add_point(0.0, 0.0)
    p2 = registry.add_point(10.0, 10.0)
    registry.add_line(p1, p2)

    center = registry.add_point(5.0, 5.0)
    radius = registry.add_point(10.0, 5.0)
    registry.add_circle(center, radius)

    drag_position = (7.5, 5.0)
    threshold = 5.0

    snap_points = list(
        producer.produce_points(
            registry, drag_position, drag_context, threshold
        )
    )

    assert len(snap_points) == 1
    assert snap_points[0].line_type == SnapLineType.MIDPOINT


def test_midpoints_producer_horizontal_line(producer, registry, drag_context):
    """Tests midpoint of horizontal line."""
    p1 = registry.add_point(0.0, 10.0)
    p2 = registry.add_point(20.0, 10.0)
    registry.add_line(p1, p2)

    drag_position = (10.0, 10.0)
    threshold = 1.0

    snap_points = list(
        producer.produce_points(
            registry, drag_position, drag_context, threshold
        )
    )

    assert len(snap_points) == 1
    assert snap_points[0].x == 10.0
    assert snap_points[0].y == 10.0


def test_midpoints_producer_vertical_line(producer, registry, drag_context):
    """Tests midpoint of vertical line."""
    p1 = registry.add_point(10.0, 0.0)
    p2 = registry.add_point(10.0, 20.0)
    registry.add_line(p1, p2)

    drag_position = (10.0, 10.0)
    threshold = 1.0

    snap_points = list(
        producer.produce_points(
            registry, drag_position, drag_context, threshold
        )
    )

    assert len(snap_points) == 1
    assert snap_points[0].x == 10.0
    assert snap_points[0].y == 10.0
