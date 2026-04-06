import pytest

from sketcher.core.snap.types import SnapLineType, DragContext
from sketcher.core.snap.producers.equidistant import EquidistantLinesProducer
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
    """Create an EquidistantLinesProducer for testing."""
    return EquidistantLinesProducer()


def test_equidistant_producer_initialization_defaults():
    """Tests EquidistantLinesProducer initialization with defaults."""
    producer = EquidistantLinesProducer()
    assert producer._spacing_tolerance == 0.5
    assert producer._max_spacing == 100.0
    assert producer._include_construction is True


def test_equidistant_producer_initialization_custom():
    """Tests EquidistantLinesProducer initialization with custom settings."""
    producer = EquidistantLinesProducer(
        spacing_tolerance=1.0, max_spacing=50.0, include_construction=False
    )
    assert producer._spacing_tolerance == 1.0
    assert producer._max_spacing == 50.0
    assert producer._include_construction is False


def test_equidistant_producer_no_snap_lines(producer, registry, drag_context):
    """Tests that producer doesn't generate snap lines."""
    registry.add_point(0.0, 0.0)
    registry.add_point(0.0, 10.0)
    registry.add_point(0.0, 20.0)

    drag_position = (0.0, 5.0)
    threshold = 5.0

    snap_lines = list(
        producer.produce(registry, drag_position, drag_context, threshold)
    )

    assert len(snap_lines) == 0


def test_equidistant_vertical_pattern(producer, registry, drag_context):
    """Tests detecting equidistant vertical pattern."""
    registry.add_point(10.0, 0.0)
    registry.add_point(10.0, 10.0)
    registry.add_point(10.0, 20.0)

    drag_position = (10.0, 30.0)
    threshold = 5.0

    snap_points = list(
        producer.produce_points(
            registry, drag_position, drag_context, threshold
        )
    )

    assert len(snap_points) == 1
    assert abs(snap_points[0].x - 10.0) < 0.001
    assert abs(snap_points[0].y - 30.0) < 0.001
    assert snap_points[0].line_type == SnapLineType.EQUIDISTANT
    assert snap_points[0].is_horizontal is True


def test_equidistant_horizontal_pattern(producer, registry, drag_context):
    """Tests detecting equidistant horizontal pattern."""
    registry.add_point(0.0, 10.0)
    registry.add_point(10.0, 10.0)
    registry.add_point(20.0, 10.0)

    drag_position = (30.0, 10.0)
    threshold = 5.0

    snap_points = list(
        producer.produce_points(
            registry, drag_position, drag_context, threshold
        )
    )

    assert len(snap_points) == 1
    assert abs(snap_points[0].x - 30.0) < 0.001
    assert abs(snap_points[0].y - 10.0) < 0.001
    assert snap_points[0].line_type == SnapLineType.EQUIDISTANT
    assert snap_points[0].is_horizontal is False


def test_equidistant_spacing_attribute(producer, registry, drag_context):
    """Tests that spacing attribute is set correctly."""
    registry.add_point(10.0, 0.0)
    registry.add_point(10.0, 10.0)
    registry.add_point(10.0, 20.0)

    drag_position = (10.0, 30.0)
    threshold = 5.0

    snap_points = list(
        producer.produce_points(
            registry, drag_position, drag_context, threshold
        )
    )

    assert len(snap_points) == 1
    assert abs(snap_points[0].spacing - 10.0) < 0.001


def test_equidistant_pattern_coords(producer, registry, drag_context):
    """Tests that pattern_coords includes all pattern points."""
    registry.add_point(10.0, 0.0)
    registry.add_point(10.0, 10.0)
    registry.add_point(10.0, 20.0)

    drag_position = (10.0, 30.0)
    threshold = 5.0

    snap_points = list(
        producer.produce_points(
            registry, drag_position, drag_context, threshold
        )
    )

    assert len(snap_points) == 1
    assert snap_points[0].pattern_coords is not None
    assert len(snap_points[0].pattern_coords) == 4


def test_equidistant_axis_coord(producer, registry, drag_context):
    """Tests that axis_coord is set correctly."""
    registry.add_point(10.0, 0.0)
    registry.add_point(10.0, 10.0)
    registry.add_point(10.0, 20.0)

    drag_position = (10.0, 30.0)
    threshold = 5.0

    snap_points = list(
        producer.produce_points(
            registry, drag_position, drag_context, threshold
        )
    )

    assert len(snap_points) == 1
    assert abs(snap_points[0].axis_coord - 10.0) < 0.001


def test_equidistant_insufficient_points(producer, registry, drag_context):
    """Tests that fewer than 2 points don't produce snaps."""
    registry.add_point(10.0, 0.0)

    drag_position = (10.0, 10.0)
    threshold = 5.0

    snap_points = list(
        producer.produce_points(
            registry, drag_position, drag_context, threshold
        )
    )

    assert len(snap_points) == 0


def test_equidistant_irregular_spacing(producer, registry, drag_context):
    """Tests irregular spacing can produce snaps based on partial patterns."""
    registry.add_point(10.0, 0.0)
    registry.add_point(10.0, 5.0)
    registry.add_point(10.0, 20.0)

    drag_position = (10.0, 30.0)
    threshold = 5.0

    snap_points = list(
        producer.produce_points(
            registry, drag_position, drag_context, threshold
        )
    )

    assert len(snap_points) >= 1


def test_equidistant_outside_threshold(producer, registry, drag_context):
    """Tests that snaps outside threshold don't produce results."""
    registry.add_point(10.0, 0.0)
    registry.add_point(10.0, 10.0)
    registry.add_point(10.0, 20.0)

    drag_position = (10.0, 50.0)
    threshold = 5.0

    snap_points = list(
        producer.produce_points(
            registry, drag_position, drag_context, threshold
        )
    )

    assert len(snap_points) == 0


def test_equidistant_max_spacing_exceeded(producer, registry, drag_context):
    """Tests that spacings exceeding max_spacing are ignored."""
    producer = EquidistantLinesProducer(max_spacing=5.0)

    registry.add_point(10.0, 0.0)
    registry.add_point(10.0, 20.0)
    registry.add_point(10.0, 40.0)

    drag_position = (10.0, 60.0)
    threshold = 5.0

    snap_points = list(
        producer.produce_points(
            registry, drag_position, drag_context, threshold
        )
    )

    assert len(snap_points) == 0


def test_equidistant_dragged_point_excluded(producer, registry, drag_context):
    """Tests that dragged points are excluded from pattern."""
    registry.add_point(10.0, 0.0)
    p2 = registry.add_point(10.0, 10.0)
    registry.add_point(10.0, 20.0)

    drag_context.dragged_point_ids.add(p2)

    drag_position = (10.0, 30.0)
    threshold = 5.0

    snap_points = list(
        producer.produce_points(
            registry, drag_position, drag_context, threshold
        )
    )

    assert len(snap_points) == 0


def test_equidistant_threshold_for_alignment(producer, registry, drag_context):
    """Tests that threshold applies to point alignment."""
    registry.add_point(10.0, 0.0)
    registry.add_point(10.1, 10.0)
    registry.add_point(10.0, 20.0)

    drag_position = (10.0, 30.0)
    threshold = 5.0

    snap_points = list(
        producer.produce_points(
            registry, drag_position, drag_context, threshold
        )
    )

    assert len(snap_points) == 1


def test_equidistant_tolerance_for_spacing(producer, registry, drag_context):
    """Tests that spacing tolerance allows near-equal spacing."""
    producer = EquidistantLinesProducer(spacing_tolerance=1.0)

    registry.add_point(10.0, 0.0)
    registry.add_point(10.0, 10.0)
    registry.add_point(10.0, 20.5)

    drag_position = (10.0, 30.5)
    threshold = 5.0

    snap_points = list(
        producer.produce_points(
            registry, drag_position, drag_context, threshold
        )
    )

    assert len(snap_points) >= 1


def test_equidistant_backward_extension(producer, registry, drag_context):
    """Tests extending pattern backwards."""
    registry.add_point(10.0, 10.0)
    registry.add_point(10.0, 20.0)
    registry.add_point(10.0, 30.0)

    drag_position = (10.0, 0.0)
    threshold = 5.0

    snap_points = list(
        producer.produce_points(
            registry, drag_position, drag_context, threshold
        )
    )

    assert len(snap_points) == 1
    assert abs(snap_points[0].y - 0.0) < 0.001


def test_equidistant_middle_of_pattern(producer, registry, drag_context):
    """Tests detecting position in middle of pattern."""
    registry.add_point(10.0, 0.0)
    registry.add_point(10.0, 10.0)
    registry.add_point(10.0, 30.0)
    registry.add_point(10.0, 40.0)

    drag_position = (10.0, 20.0)
    threshold = 5.0

    snap_points = list(
        producer.produce_points(
            registry, drag_position, drag_context, threshold
        )
    )

    assert len(snap_points) == 1
    assert abs(snap_points[0].y - 20.0) < 0.001


def test_equidistant_negative_coordinates(producer, registry, drag_context):
    """Tests snap generation with negative coordinates."""
    registry.add_point(10.0, -20.0)
    registry.add_point(10.0, -10.0)
    registry.add_point(10.0, 0.0)

    drag_position = (10.0, 10.0)
    threshold = 5.0

    snap_points = list(
        producer.produce_points(
            registry, drag_position, drag_context, threshold
        )
    )

    assert len(snap_points) == 1
    assert abs(snap_points[0].y - 10.0) < 0.001


def test_equidistant_empty_registry(producer, registry, drag_context):
    """Tests producer with empty registry."""
    drag_position = (10.0, 10.0)
    threshold = 5.0

    snap_points = list(
        producer.produce_points(
            registry, drag_position, drag_context, threshold
        )
    )

    assert len(snap_points) == 0


def test_equidistant_single_pattern(producer, registry, drag_context):
    """Tests detection of a single equidistant pattern."""
    registry.add_point(0.0, 10.0)
    registry.add_point(10.0, 10.0)
    registry.add_point(20.0, 10.0)

    drag_position = (30.0, 10.0)
    threshold = 5.0

    snap_points = list(
        producer.produce_points(
            registry, drag_position, drag_context, threshold
        )
    )

    assert len(snap_points) == 1
    assert abs(snap_points[0].x - 30.0) < 0.001


def test_equidistant_multiple_patterns(producer, registry, drag_context):
    """Tests detection of equidistant pattern with merged columns."""
    registry.add_point(0.0, 0.0)
    registry.add_point(0.0, 10.0)
    registry.add_point(0.0, 20.0)

    registry.add_point(10.0, 0.0)
    registry.add_point(10.0, 10.0)
    registry.add_point(10.0, 20.0)

    drag_position = (5.0, 30.0)
    threshold = 5.0

    snap_points = list(
        producer.produce_points(
            registry, drag_position, drag_context, threshold
        )
    )

    assert len(snap_points) == 1


def test_equidistant_min_spacing_requirement(producer, registry, drag_context):
    """Tests that spacing smaller than 1e-6 is ignored."""
    registry.add_point(10.0, 0.0)
    registry.add_point(10.0, 1e-9)
    registry.add_point(10.0, 2e-9)

    drag_position = (10.0, 3e-9)
    threshold = 5.0

    snap_points = list(
        producer.produce_points(
            registry, drag_position, drag_context, threshold
        )
    )

    assert len(snap_points) == 0


def test_equidistant_duplicate_points(producer, registry, drag_context):
    """Tests that duplicate points don't break pattern detection."""
    registry.add_point(10.0, 0.0)
    registry.add_point(10.0, 10.0)
    registry.add_point(10.0, 10.0)
    registry.add_point(10.0, 20.0)

    drag_position = (10.0, 30.0)
    threshold = 5.0

    snap_points = list(
        producer.produce_points(
            registry, drag_position, drag_context, threshold
        )
    )

    assert len(snap_points) == 1


def test_equidistant_exact_match(producer, registry, drag_context):
    """Tests snap generation at exact pattern position."""
    registry.add_point(10.0, 0.0)
    registry.add_point(10.0, 10.0)
    registry.add_point(10.0, 20.0)

    drag_position = (10.0, 30.0)
    threshold = 0.1

    snap_points = list(
        producer.produce_points(
            registry, drag_position, drag_context, threshold
        )
    )

    assert len(snap_points) == 1
    assert abs(snap_points[0].y - 30.0) < 0.001


def test_equidistant_non_aligned_points(producer, registry, drag_context):
    """Tests non-aligned points excluded, aligned subset can form pattern."""
    registry.add_point(10.0, 0.0)
    registry.add_point(10.0, 10.0)
    registry.add_point(20.0, 20.0)

    drag_position = (10.0, 20.0)
    threshold = 5.0

    snap_points = list(
        producer.produce_points(
            registry, drag_position, drag_context, threshold
        )
    )

    assert len(snap_points) == 1
