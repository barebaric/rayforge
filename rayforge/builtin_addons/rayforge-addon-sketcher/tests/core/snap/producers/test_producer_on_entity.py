import pytest

from sketcher.core.snap.types import SnapLineType, DragContext
from sketcher.core.snap.producers.on_entity import OnEntityProducer
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
    """Create an OnEntityProducer for testing."""
    return OnEntityProducer()


def test_on_entity_producer_no_snap_lines(producer, registry, drag_context):
    """Tests that producer doesn't generate snap lines."""
    p1 = registry.add_point(0.0, 0.0)
    p2 = registry.add_point(10.0, 0.0)
    registry.add_line(p1, p2)

    drag_position = (5.0, 5.0)
    threshold = 5.0

    snap_lines = list(
        producer.produce(registry, drag_position, drag_context, threshold)
    )

    assert len(snap_lines) == 0


def test_on_entity_producer_line_snap(producer, registry, drag_context):
    """Tests snapping to a line."""
    p1 = registry.add_point(0.0, 0.0)
    p2 = registry.add_point(10.0, 0.0)
    registry.add_line(p1, p2)

    drag_position = (5.0, 3.0)
    threshold = 5.0

    snap_points = list(
        producer.produce_points(
            registry, drag_position, drag_context, threshold
        )
    )

    assert len(snap_points) == 1
    assert abs(snap_points[0].x - 5.0) < 0.001
    assert abs(snap_points[0].y - 0.0) < 0.001
    assert snap_points[0].line_type == SnapLineType.ON_ENTITY


def test_on_entity_producer_line_endpoint_snap(
    producer, registry, drag_context
):
    """Tests snapping to line endpoint."""
    p1 = registry.add_point(0.0, 0.0)
    p2 = registry.add_point(10.0, 0.0)
    registry.add_line(p1, p2)

    drag_position = (10.0, 3.0)
    threshold = 5.0

    snap_points = list(
        producer.produce_points(
            registry, drag_position, drag_context, threshold
        )
    )

    assert len(snap_points) == 1
    assert abs(snap_points[0].x - 10.0) < 0.001
    assert abs(snap_points[0].y - 0.0) < 0.001


def test_on_entity_producer_circle_snap(producer, registry, drag_context):
    """Tests snapping to a circle."""
    center = registry.add_point(10.0, 10.0)
    radius_pt = registry.add_point(20.0, 10.0)
    registry.add_circle(center, radius_pt)

    drag_position = (20.0, 15.0)
    threshold = 5.0

    snap_points = list(
        producer.produce_points(
            registry, drag_position, drag_context, threshold
        )
    )

    assert len(snap_points) == 1
    assert abs(snap_points[0].x - 18.94) < 0.01
    assert abs(snap_points[0].y - 14.47) < 0.01


def test_on_entity_producer_arc_snap(producer, registry, drag_context):
    """Tests snapping to an arc."""
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


def test_on_entity_producer_arc_outside_sweep(
    producer, registry, drag_context
):
    """Tests that points outside arc sweep don't produce snaps."""
    center = registry.add_point(0.0, 0.0)
    start = registry.add_point(10.0, 0.0)
    end = registry.add_point(0.0, 10.0)
    registry.add_arc(start, end, center)

    drag_position = (-7.07, -7.07)
    threshold = 5.0

    snap_points = list(
        producer.produce_points(
            registry, drag_position, drag_context, threshold
        )
    )

    assert len(snap_points) == 0


def test_on_entity_producer_dragged_entity_excluded(
    producer, registry, drag_context
):
    """Tests that dragged entities are excluded from snap generation."""
    p1 = registry.add_point(0.0, 0.0)
    p2 = registry.add_point(10.0, 0.0)
    line_id = registry.add_line(p1, p2)

    drag_context.dragged_entity_ids.add(line_id)

    drag_position = (5.0, 3.0)
    threshold = 5.0

    snap_points = list(
        producer.produce_points(
            registry, drag_position, drag_context, threshold
        )
    )

    assert len(snap_points) == 0


def test_on_entity_producer_dragged_point_excluded(
    producer, registry, drag_context
):
    """Tests that entities with dragged points are excluded."""
    p1 = registry.add_point(0.0, 0.0)
    p2 = registry.add_point(10.0, 0.0)
    registry.add_line(p1, p2)

    drag_context.dragged_point_ids.add(p1)

    drag_position = (5.0, 3.0)
    threshold = 5.0

    snap_points = list(
        producer.produce_points(
            registry, drag_position, drag_context, threshold
        )
    )

    assert len(snap_points) == 0


def test_on_entity_producer_outside_threshold(
    producer, registry, drag_context
):
    """Tests that points outside threshold don't produce snaps."""
    p1 = registry.add_point(0.0, 0.0)
    p2 = registry.add_point(10.0, 0.0)
    registry.add_line(p1, p2)

    drag_position = (5.0, 50.0)
    threshold = 5.0

    snap_points = list(
        producer.produce_points(
            registry, drag_position, drag_context, threshold
        )
    )

    assert len(snap_points) == 0


def test_on_entity_producer_multiple_entities(
    producer, registry, drag_context
):
    """Tests producer with multiple entities."""
    p1 = registry.add_point(0.0, 0.0)
    p2 = registry.add_point(10.0, 0.0)
    registry.add_line(p1, p2)

    center = registry.add_point(10.0, 10.0)
    radius = registry.add_point(20.0, 10.0)
    registry.add_circle(center, radius)

    drag_position = (5.0, 3.0)
    threshold = 5.0

    snap_points = list(
        producer.produce_points(
            registry, drag_position, drag_context, threshold
        )
    )

    assert len(snap_points) >= 1


def test_on_entity_producer_source_attribute(producer, registry, drag_context):
    """Tests that source attribute is set to the entity."""
    p1 = registry.add_point(0.0, 0.0)
    p2 = registry.add_point(10.0, 0.0)
    line_id = registry.add_line(p1, p2)
    line = registry.get_entity(line_id)

    drag_position = (5.0, 3.0)
    threshold = 5.0

    snap_points = list(
        producer.produce_points(
            registry, drag_position, drag_context, threshold
        )
    )

    assert snap_points[0].source == line


def test_on_entity_producer_negative_coordinates(
    producer, registry, drag_context
):
    """Tests snap generation with negative coordinates."""
    p1 = registry.add_point(-10.0, -10.0)
    p2 = registry.add_point(10.0, -10.0)
    registry.add_line(p1, p2)

    drag_position = (0.0, -7.0)
    threshold = 5.0

    snap_points = list(
        producer.produce_points(
            registry, drag_position, drag_context, threshold
        )
    )

    assert len(snap_points) == 1
    assert abs(snap_points[0].x) < 0.001
    assert abs(snap_points[0].y - (-10.0)) < 0.001


def test_on_entity_producer_empty_registry(producer, registry, drag_context):
    """Tests producer with empty registry."""
    drag_position = (5.0, 5.0)
    threshold = 5.0

    snap_points = list(
        producer.produce_points(
            registry, drag_position, drag_context, threshold
        )
    )

    assert len(snap_points) == 0


def test_on_entity_producer_line_projection(producer, registry, drag_context):
    """Tests projection onto line endpoint outside segment."""
    p1 = registry.add_point(10.0, 0.0)
    p2 = registry.add_point(20.0, 0.0)
    registry.add_line(p1, p2)

    drag_position = (5.0, 3.0)
    threshold = 5.0

    snap_points = list(
        producer.produce_points(
            registry, drag_position, drag_context, threshold
        )
    )

    assert len(snap_points) == 0


def test_on_entity_producer_diagonal_line(producer, registry, drag_context):
    """Tests snapping to diagonal line."""
    p1 = registry.add_point(0.0, 0.0)
    p2 = registry.add_point(10.0, 10.0)
    registry.add_line(p1, p2)

    drag_position = (5.0, 7.0)
    threshold = 5.0

    snap_points = list(
        producer.produce_points(
            registry, drag_position, drag_context, threshold
        )
    )

    assert len(snap_points) == 1
    assert abs(snap_points[0].x - 6.0) < 0.001
    assert abs(snap_points[0].y - 6.0) < 0.001


def test_on_entity_producer_circle_center_snap(
    producer, registry, drag_context
):
    """Tests snapping from near circle center."""
    center = registry.add_point(10.0, 10.0)
    radius_pt = registry.add_point(20.0, 10.0)
    registry.add_circle(center, radius_pt)

    drag_position = (10.0, 10.5)
    threshold = 5.0

    snap_points = list(
        producer.produce_points(
            registry, drag_position, drag_context, threshold
        )
    )

    assert len(snap_points) == 0


def test_on_entity_producer_zero_radius_circle(
    producer, registry, drag_context
):
    """Tests that zero radius circle doesn't produce snaps."""
    center = registry.add_point(10.0, 10.0)
    radius_pt = registry.add_point(10.0, 10.0)
    registry.add_circle(center, radius_pt)

    drag_position = (10.0, 15.0)
    threshold = 5.0

    snap_points = list(
        producer.produce_points(
            registry, drag_position, drag_context, threshold
        )
    )

    assert len(snap_points) == 0


def test_on_entity_producer_degenerate_line(producer, registry, drag_context):
    """Tests that degenerate line (zero length) produces snap at endpoint."""
    p1 = registry.add_point(10.0, 10.0)
    p2 = registry.add_point(10.0, 10.0)
    registry.add_line(p1, p2)

    drag_position = (10.0, 15.0)
    threshold = 5.0

    snap_points = list(
        producer.produce_points(
            registry, drag_position, drag_context, threshold
        )
    )

    assert len(snap_points) == 1
    assert abs(snap_points[0].x - 10.0) < 0.001
    assert abs(snap_points[0].y - 10.0) < 0.001


def test_on_entity_producer_exact_on_circle(producer, registry, drag_context):
    """Tests snap when exactly on circle."""
    center = registry.add_point(10.0, 10.0)
    radius_pt = registry.add_point(20.0, 10.0)
    registry.add_circle(center, radius_pt)

    drag_position = (20.0, 10.0)
    threshold = 0.1

    snap_points = list(
        producer.produce_points(
            registry, drag_position, drag_context, threshold
        )
    )

    assert len(snap_points) == 1
    assert abs(snap_points[0].x - 20.0) < 0.001
    assert abs(snap_points[0].y - 10.0) < 0.001


def test_on_entity_producer_threshold_boundary(
    producer, registry, drag_context
):
    """Tests snap generation at threshold boundary."""
    p1 = registry.add_point(0.0, 0.0)
    p2 = registry.add_point(10.0, 0.0)
    registry.add_line(p1, p2)

    drag_position = (5.0, 5.0)
    threshold = 5.0

    snap_points = list(
        producer.produce_points(
            registry, drag_position, drag_context, threshold
        )
    )

    assert len(snap_points) == 1

    drag_position = (5.0, 5.1)
    snap_points = list(
        producer.produce_points(
            registry, drag_position, drag_context, threshold
        )
    )

    assert len(snap_points) == 0
