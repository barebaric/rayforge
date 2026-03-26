import pytest

from sketcher.core.snap.types import DragContext
from sketcher.core.snap.producers.intersections import IntersectionsProducer
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
    """Create an IntersectionsProducer for testing."""
    return IntersectionsProducer()


def test_intersections_producer_initialization_default():
    """Tests IntersectionsProducer initialization with defaults."""
    producer = IntersectionsProducer()
    assert producer._include_construction is True


def test_intersections_producer_initialization_custom():
    """Tests IntersectionsProducer initialization with custom settings."""
    producer = IntersectionsProducer(include_construction=False)
    assert producer._include_construction is False


def test_line_line_intersection(producer, registry, drag_context):
    """Tests finding intersection of two lines."""
    p1 = registry.add_point(0.0, 0.0)
    p2 = registry.add_point(10.0, 10.0)
    registry.add_line(p1, p2)

    p3 = registry.add_point(0.0, 10.0)
    p4 = registry.add_point(10.0, 0.0)
    registry.add_line(p3, p4)

    drag_position = (5.0, 5.0)
    threshold = 5.0

    snap_lines = list(
        producer.produce(registry, drag_position, drag_context, threshold)
    )

    assert len(snap_lines) == 2
    horizontal_lines = [sl for sl in snap_lines if sl.is_horizontal]
    vertical_lines = [sl for sl in snap_lines if not sl.is_horizontal]
    assert len(horizontal_lines) == 1
    assert len(vertical_lines) == 1
    assert horizontal_lines[0].coordinate == 5.0
    assert vertical_lines[0].coordinate == 5.0

    snap_points = list(
        producer.produce_points(
            registry, drag_position, drag_context, threshold
        )
    )

    assert len(snap_points) == 1
    assert abs(snap_points[0].x - 5.0) < 0.001
    assert abs(snap_points[0].y - 5.0) < 0.001


def test_line_circle_intersection(producer, registry, drag_context):
    """Tests finding intersection of line and circle."""
    p1 = registry.add_point(0.0, 5.0)
    p2 = registry.add_point(10.0, 5.0)
    registry.add_line(p1, p2)

    center = registry.add_point(5.0, 5.0)
    radius_pt = registry.add_point(10.0, 5.0)
    registry.add_circle(center, radius_pt)

    drag_position = (10.0, 5.0)
    threshold = 5.0

    snap_points = list(
        producer.produce_points(
            registry, drag_position, drag_context, threshold
        )
    )

    assert len(snap_points) == 1


def test_line_arc_intersection(producer, registry, drag_context):
    """Tests finding intersection of line and arc."""
    p1 = registry.add_point(0.0, 0.0)
    p2 = registry.add_point(10.0, 0.0)
    registry.add_line(p1, p2)

    center = registry.add_point(5.0, 0.0)
    start = registry.add_point(10.0, 0.0)
    end = registry.add_point(0.0, 0.0)
    registry.add_arc(start, end, center)

    drag_position = (10.0, 0.0)
    threshold = 5.0

    snap_points = list(
        producer.produce_points(
            registry, drag_position, drag_context, threshold
        )
    )

    assert len(snap_points) == 1


def test_circle_circle_intersection(producer, registry, drag_context):
    """Tests finding intersection of two circles."""
    center1 = registry.add_point(0.0, 0.0)
    radius1 = registry.add_point(5.0, 0.0)
    registry.add_circle(center1, radius1)

    center2 = registry.add_point(5.0, 0.0)
    radius2 = registry.add_point(10.0, 0.0)
    registry.add_circle(center2, radius2)

    drag_position = (5.0, 4.0)
    threshold = 10.0

    snap_points = list(
        producer.produce_points(
            registry, drag_position, drag_context, threshold
        )
    )

    assert len(snap_points) == 2


def test_arc_arc_intersection(producer, registry, drag_context):
    """Concentric arcs (same center, same radius) produce no intersections."""
    center1 = registry.add_point(0.0, 0.0)
    start1 = registry.add_point(5.0, 0.0)
    end1 = registry.add_point(0.0, 5.0)
    registry.add_arc(start1, end1, center1)

    center2 = registry.add_point(0.0, 0.0)
    start2 = registry.add_point(0.0, 5.0)
    end2 = registry.add_point(-5.0, 0.0)
    registry.add_arc(start2, end2, center2)

    drag_position = (0.0, 5.0)
    threshold = 5.0

    snap_points = list(
        producer.produce_points(
            registry, drag_position, drag_context, threshold
        )
    )

    assert len(snap_points) == 0


def test_parallel_lines_no_intersection(producer, registry, drag_context):
    """Tests that parallel lines don't produce intersections."""
    p1 = registry.add_point(0.0, 0.0)
    p2 = registry.add_point(10.0, 0.0)
    registry.add_line(p1, p2)

    p3 = registry.add_point(0.0, 5.0)
    p4 = registry.add_point(10.0, 5.0)
    registry.add_line(p3, p4)

    drag_position = (5.0, 2.5)
    threshold = 5.0

    snap_points = list(
        producer.produce_points(
            registry, drag_position, drag_context, threshold
        )
    )

    assert len(snap_points) == 0


def test_outside_segment_no_intersection(producer, registry, drag_context):
    """Tests that line intersections outside segments don't produce snaps."""
    p1 = registry.add_point(0.0, 0.0)
    p2 = registry.add_point(10.0, 0.0)
    registry.add_line(p1, p2)

    p3 = registry.add_point(15.0, 5.0)
    p4 = registry.add_point(25.0, -5.0)
    registry.add_line(p3, p4)

    drag_position = (5.0, 0.0)
    threshold = 5.0

    snap_points = list(
        producer.produce_points(
            registry, drag_position, drag_context, threshold
        )
    )

    assert len(snap_points) == 0


def test_dragged_entity_excluded(producer, registry, drag_context):
    """Tests that dragged entities are excluded."""
    p1 = registry.add_point(0.0, 0.0)
    p2 = registry.add_point(10.0, 10.0)
    registry.add_line(p1, p2)

    p3 = registry.add_point(0.0, 10.0)
    p4 = registry.add_point(10.0, 0.0)
    line2 = registry.add_line(p3, p4)

    drag_context.dragged_entity_ids.add(line2)

    drag_position = (5.0, 5.0)
    threshold = 5.0

    snap_points = list(
        producer.produce_points(
            registry, drag_position, drag_context, threshold
        )
    )

    assert len(snap_points) == 0


def test_dragged_point_excluded(producer, registry, drag_context):
    """Tests that entities with dragged points are excluded."""
    p1 = registry.add_point(0.0, 0.0)
    p2 = registry.add_point(10.0, 10.0)
    registry.add_line(p1, p2)

    p3 = registry.add_point(0.0, 10.0)
    p4 = registry.add_point(10.0, 0.0)
    registry.add_line(p3, p4)

    drag_context.dragged_point_ids.add(p3)

    drag_position = (5.0, 5.0)
    threshold = 5.0

    snap_points = list(
        producer.produce_points(
            registry, drag_position, drag_context, threshold
        )
    )

    assert len(snap_points) == 0


def test_construction_excluded(producer, registry, drag_context):
    """Tests that construction entities are excluded when configured."""
    producer = IntersectionsProducer(include_construction=False)

    p1 = registry.add_point(0.0, 0.0)
    p2 = registry.add_point(10.0, 10.0)
    registry.add_line(p1, p2)

    p3 = registry.add_point(0.0, 10.0)
    p4 = registry.add_point(10.0, 0.0)
    line2_id = registry.add_line(p3, p4)
    registry.get_entity(line2_id).construction = True

    drag_position = (5.0, 5.0)
    threshold = 5.0

    snap_points = list(
        producer.produce_points(
            registry, drag_position, drag_context, threshold
        )
    )

    assert len(snap_points) == 0


def test_construction_included(producer, registry, drag_context):
    """Tests that construction entities are included when configured."""
    p1 = registry.add_point(0.0, 0.0)
    p2 = registry.add_point(10.0, 10.0)
    line1_id = registry.add_line(p1, p2)
    registry.get_entity(line1_id).construction = True

    p3 = registry.add_point(0.0, 10.0)
    p4 = registry.add_point(10.0, 0.0)
    line2_id = registry.add_line(p3, p4)
    registry.get_entity(line2_id).construction = True

    drag_position = (5.0, 5.0)
    threshold = 5.0

    snap_points = list(
        producer.produce_points(
            registry, drag_position, drag_context, threshold
        )
    )

    assert len(snap_points) == 1


def test_outside_threshold(producer, registry, drag_context):
    """Tests that intersections outside threshold don't produce snaps."""
    p1 = registry.add_point(0.0, 0.0)
    p2 = registry.add_point(10.0, 10.0)
    registry.add_line(p1, p2)

    p3 = registry.add_point(0.0, 10.0)
    p4 = registry.add_point(10.0, 0.0)
    registry.add_line(p3, p4)

    drag_position = (100.0, 100.0)
    threshold = 5.0

    snap_points = list(
        producer.produce_points(
            registry, drag_position, drag_context, threshold
        )
    )

    assert len(snap_points) == 0


def test_multiple_intersections(producer, registry, drag_context):
    """Tests producer with multiple entity pairs."""
    p1 = registry.add_point(0.0, 0.0)
    p2 = registry.add_point(10.0, 10.0)
    registry.add_line(p1, p2)

    p3 = registry.add_point(0.0, 10.0)
    p4 = registry.add_point(10.0, 0.0)
    registry.add_line(p3, p4)

    p5 = registry.add_point(-5.0, 5.0)
    p6 = registry.add_point(15.0, 5.0)
    registry.add_line(p5, p6)

    drag_position = (5.0, 5.0)
    threshold = 10.0

    snap_points = list(
        producer.produce_points(
            registry, drag_position, drag_context, threshold
        )
    )

    assert len(snap_points) >= 2


def test_tangent_line_circle(producer, registry, drag_context):
    """Tests line tangent to circle."""
    p1 = registry.add_point(0.0, 5.0)
    p2 = registry.add_point(10.0, 5.0)
    registry.add_line(p1, p2)

    center = registry.add_point(5.0, 0.0)
    radius_pt = registry.add_point(5.0, 5.0)
    registry.add_circle(center, radius_pt)

    drag_position = (5.0, 5.0)
    threshold = 1.0

    snap_points = list(
        producer.produce_points(
            registry, drag_position, drag_context, threshold
        )
    )

    assert len(snap_points) == 1


def test_circle_tangent_intersection(producer, registry, drag_context):
    """Tests circles tangent at one point."""
    center1 = registry.add_point(0.0, 0.0)
    radius1 = registry.add_point(5.0, 0.0)
    registry.add_circle(center1, radius1)

    center2 = registry.add_point(10.0, 0.0)
    radius2 = registry.add_point(15.0, 0.0)
    registry.add_circle(center2, radius2)

    drag_position = (5.0, 0.0)
    threshold = 5.0

    snap_points = list(
        producer.produce_points(
            registry, drag_position, drag_context, threshold
        )
    )

    assert len(snap_points) == 1


def test_no_circles_overlap(producer, registry, drag_context):
    """Tests non-overlapping circles."""
    center1 = registry.add_point(0.0, 0.0)
    radius1 = registry.add_point(5.0, 0.0)
    registry.add_circle(center1, radius1)

    center2 = registry.add_point(20.0, 0.0)
    radius2 = registry.add_point(25.0, 0.0)
    registry.add_circle(center2, radius2)

    drag_position = (5.0, 0.0)
    threshold = 5.0

    snap_points = list(
        producer.produce_points(
            registry, drag_position, drag_context, threshold
        )
    )

    assert len(snap_points) == 0


def test_circle_inside_circle(producer, registry, drag_context):
    """Tests one circle inside another."""
    center1 = registry.add_point(0.0, 0.0)
    radius1 = registry.add_point(10.0, 0.0)
    registry.add_circle(center1, radius1)

    center2 = registry.add_point(0.0, 0.0)
    radius2 = registry.add_point(5.0, 0.0)
    registry.add_circle(center2, radius2)

    drag_position = (5.0, 0.0)
    threshold = 5.0

    snap_points = list(
        producer.produce_points(
            registry, drag_position, drag_context, threshold
        )
    )

    assert len(snap_points) == 0


def test_zero_radius_circle(producer, registry, drag_context):
    """Tests that zero radius circle doesn't produce intersections."""
    center1 = registry.add_point(0.0, 0.0)
    radius1 = registry.add_point(0.0, 0.0)
    registry.add_circle(center1, radius1)

    p1 = registry.add_point(-5.0, 0.0)
    p2 = registry.add_point(5.0, 0.0)
    registry.add_line(p1, p2)

    drag_position = (0.0, 0.0)
    threshold = 5.0

    snap_points = list(
        producer.produce_points(
            registry, drag_position, drag_context, threshold
        )
    )

    assert len(snap_points) == 0


def test_empty_registry(producer, registry, drag_context):
    """Tests producer with empty registry."""
    drag_position = (5.0, 5.0)
    threshold = 5.0

    snap_points = list(
        producer.produce_points(
            registry, drag_position, drag_context, threshold
        )
    )

    assert len(snap_points) == 0


def test_single_entity_no_intersection(producer, registry, drag_context):
    """Tests that single entity doesn't produce intersections."""
    p1 = registry.add_point(0.0, 0.0)
    p2 = registry.add_point(10.0, 10.0)
    registry.add_line(p1, p2)

    drag_position = (5.0, 5.0)
    threshold = 5.0

    snap_points = list(
        producer.produce_points(
            registry, drag_position, drag_context, threshold
        )
    )

    assert len(snap_points) == 0


def test_intersection_source_attribute(producer, registry, drag_context):
    """Tests that intersection snap points don't have source set."""
    p1 = registry.add_point(0.0, 0.0)
    p2 = registry.add_point(10.0, 10.0)
    registry.add_line(p1, p2)

    p3 = registry.add_point(0.0, 10.0)
    p4 = registry.add_point(10.0, 0.0)
    registry.add_line(p3, p4)

    drag_position = (5.0, 5.0)
    threshold = 5.0

    snap_points = list(
        producer.produce_points(
            registry, drag_position, drag_context, threshold
        )
    )

    assert snap_points[0].source is None


def test_arc_intersection_outside_sweep(producer, registry, drag_context):
    """Tests that arc intersections outside sweep don't produce snaps."""
    center = registry.add_point(0.0, 0.0)
    start = registry.add_point(10.0, 0.0)
    end = registry.add_point(0.0, 10.0)
    registry.add_arc(start, end, center)

    p1 = registry.add_point(5.0, -5.0)
    p2 = registry.add_point(5.0, 15.0)
    registry.add_line(p1, p2)

    drag_position = (5.0, 0.0)
    threshold = 5.0

    snap_points = list(
        producer.produce_points(
            registry, drag_position, drag_context, threshold
        )
    )

    assert len(snap_points) == 0
