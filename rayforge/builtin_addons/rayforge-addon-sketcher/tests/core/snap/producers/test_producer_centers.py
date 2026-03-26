import pytest

from sketcher.core.snap.types import SnapLineType, DragContext
from sketcher.core.snap.producers.centers import CentersProducer
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
    """Create a CentersProducer for testing."""
    return CentersProducer()


def test_centers_producer_initialization_default():
    """Tests CentersProducer initialization with defaults."""
    producer = CentersProducer()
    assert producer._include_construction is True


def test_centers_producer_initialization_custom():
    """Tests CentersProducer initialization with custom settings."""
    producer = CentersProducer(include_construction=False)
    assert producer._include_construction is False


def test_centers_producer_produce_circle_center(
    producer, registry, drag_context
):
    """Tests producing snap lines from circle center."""
    center_point = registry.add_point(10.0, 20.0)
    radius_point = registry.add_point(20.0, 20.0)
    registry.add_circle(center_point, radius_point)

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


def test_centers_producer_produce_arc_center(producer, registry, drag_context):
    """Tests producing snap lines from arc center."""
    start_point = registry.add_point(20.0, 20.0)
    end_point = registry.add_point(20.0, 30.0)
    center_point = registry.add_point(10.0, 20.0)
    registry.add_arc(start_point, end_point, center_point)

    drag_position = (12.0, 18.0)
    threshold = 5.0

    snap_lines = list(
        producer.produce(registry, drag_position, drag_context, threshold)
    )

    assert len(snap_lines) == 2


def test_centers_producer_produce_ellipse_center(
    producer, registry, drag_context
):
    """Tests producing snap lines from ellipse center."""
    center_point = registry.add_point(10.0, 20.0)
    radius_x_point = registry.add_point(20.0, 20.0)
    radius_y_point = registry.add_point(10.0, 30.0)
    registry.add_ellipse(center_point, radius_x_point, radius_y_point)

    drag_position = (12.0, 18.0)
    threshold = 5.0

    snap_lines = list(
        producer.produce(registry, drag_position, drag_context, threshold)
    )

    assert len(snap_lines) == 2


def test_centers_producer_produce_points(producer, registry, drag_context):
    """Tests producing snap points from entity centers."""
    center_point = registry.add_point(10.0, 20.0)
    radius_point = registry.add_point(20.0, 20.0)
    registry.add_circle(center_point, radius_point)

    drag_position = (12.0, 18.0)
    threshold = 5.0

    snap_points = list(
        producer.produce_points(
            registry, drag_position, drag_context, threshold
        )
    )

    assert len(snap_points) == 1
    assert snap_points[0].x == 10.0
    assert snap_points[0].y == 20.0
    assert snap_points[0].line_type == SnapLineType.CENTER


def test_centers_producer_dragged_entity_excluded(
    producer, registry, drag_context
):
    """Tests that dragged entities are excluded from snap generation."""
    center_point = registry.add_point(10.0, 20.0)
    radius_point = registry.add_point(20.0, 20.0)
    circle_id = registry.add_circle(center_point, radius_point)

    drag_context.dragged_entity_ids.add(circle_id)

    drag_position = (12.0, 18.0)
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


def test_centers_producer_construction_excluded(
    producer, registry, drag_context
):
    """Tests that construction entities are excluded when configured."""
    producer = CentersProducer(include_construction=False)

    center_point_id = registry.add_point(10.0, 20.0)
    radius_point_id = registry.add_point(20.0, 20.0)
    circle_id = registry.add_circle(center_point_id, radius_point_id)
    registry.get_entity(circle_id).construction = True

    drag_position = (12.0, 18.0)
    threshold = 5.0

    snap_lines = list(
        producer.produce(registry, drag_position, drag_context, threshold)
    )

    assert len(snap_lines) == 0


def test_centers_producer_construction_included(
    producer, registry, drag_context
):
    """Tests that construction entities are included when configured."""
    center_point_id = registry.add_point(10.0, 20.0)
    radius_point_id = registry.add_point(20.0, 20.0)
    circle_id = registry.add_circle(center_point_id, radius_point_id)
    registry.get_entity(circle_id).construction = True

    drag_position = (12.0, 18.0)
    threshold = 5.0

    snap_lines = list(
        producer.produce(registry, drag_position, drag_context, threshold)
    )

    assert len(snap_lines) == 2


def test_centers_producer_outside_threshold(producer, registry, drag_context):
    """Tests that centers outside threshold don't produce snaps."""
    center_point = registry.add_point(100.0, 200.0)
    radius_point = registry.add_point(110.0, 200.0)
    registry.add_circle(center_point, radius_point)

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


def test_centers_producer_source_attribute(producer, registry, drag_context):
    """Tests that source attribute is set to the entity."""
    center_point_id = registry.add_point(10.0, 20.0)
    radius_point_id = registry.add_point(20.0, 20.0)
    circle_id = registry.add_circle(center_point_id, radius_point_id)
    circle = registry.get_entity(circle_id)

    drag_position = (12.0, 18.0)
    threshold = 5.0

    snap_lines = list(
        producer.produce(registry, drag_position, drag_context, threshold)
    )

    assert all(sl.source == circle for sl in snap_lines)

    snap_points = list(
        producer.produce_points(
            registry, drag_position, drag_context, threshold
        )
    )

    assert snap_points[0].source == circle


def test_centers_producer_multiple_entities(producer, registry, drag_context):
    """Tests producer with multiple entities."""
    center1_id = registry.add_point(10.0, 20.0)
    radius1_id = registry.add_point(20.0, 20.0)
    registry.add_circle(center1_id, radius1_id)

    center2_id = registry.add_point(30.0, 40.0)
    radius2_id = registry.add_point(40.0, 40.0)
    registry.add_circle(center2_id, radius2_id)

    drag_position = (15.0, 25.0)
    threshold = 10.0

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


def test_centers_producer_line_entity_no_snap(
    producer, registry, drag_context
):
    """Tests that line entities don't produce center snaps."""
    p1 = registry.add_point(0.0, 0.0)
    p2 = registry.add_point(10.0, 10.0)
    registry.add_line(p1, p2)

    drag_position = (5.0, 5.0)
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


def test_centers_producer_negative_coordinates(
    producer, registry, drag_context
):
    """Tests snap generation with negative coordinates."""
    center_point = registry.add_point(-10.0, -20.0)
    radius_point = registry.add_point(0.0, -20.0)
    registry.add_circle(center_point, radius_point)

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


def test_centers_producer_empty_registry(producer, registry, drag_context):
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


def test_centers_producer_exact_match(producer, registry, drag_context):
    """Tests snap generation when drag position exactly matches center."""
    center_point = registry.add_point(10.0, 20.0)
    radius_point = registry.add_point(20.0, 20.0)
    registry.add_circle(center_point, radius_point)

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
