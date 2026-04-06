import pytest

from sketcher.core.snap.types import (
    SnapLineType,
    SnapLineStyle,
    SNAP_LINE_STYLES,
    SnapPoint,
    SnapLine,
    SnapResult,
    DragContext,
)


def test_snap_line_type_priority():
    """Tests that SnapLineType priority values are correct."""
    assert SnapLineType.ENTITY_POINT.priority == 100
    assert SnapLineType.MIDPOINT.priority == 90
    assert SnapLineType.ON_ENTITY.priority == 80
    assert SnapLineType.INTERSECTION.priority == 70
    assert SnapLineType.EQUIDISTANT.priority == 60
    assert SnapLineType.TANGENT.priority == 40
    assert SnapLineType.CENTER.priority == 30


def test_snap_line_style_defaults():
    """Tests default SnapLineStyle values."""
    style = SnapLineStyle()
    assert style.color == (0.0, 0.6, 1.0, 0.8)
    assert style.dash is None
    assert style.line_width == 1.0


def test_snap_line_styles_dict():
    """Tests that SNAP_LINE_STYLES contains all SnapLineType entries."""
    assert SnapLineType.ENTITY_POINT in SNAP_LINE_STYLES
    assert SnapLineType.ON_ENTITY in SNAP_LINE_STYLES
    assert SnapLineType.INTERSECTION in SNAP_LINE_STYLES
    assert SnapLineType.MIDPOINT in SNAP_LINE_STYLES
    assert SnapLineType.EQUIDISTANT in SNAP_LINE_STYLES
    assert SnapLineType.TANGENT in SNAP_LINE_STYLES
    assert SnapLineType.CENTER in SNAP_LINE_STYLES


def test_snap_point_creation():
    """Tests basic SnapPoint creation and properties."""
    snap_point = SnapPoint(x=10.5, y=20.3, line_type=SnapLineType.ENTITY_POINT)
    assert snap_point.x == 10.5
    assert snap_point.y == 20.3
    assert snap_point.line_type == SnapLineType.ENTITY_POINT
    assert snap_point.source is None
    assert snap_point.spacing is None
    assert snap_point.is_horizontal is False
    assert snap_point.pattern_coords is None
    assert snap_point.axis_coord is None
    assert snap_point.pos == (10.5, 20.3)


def test_snap_point_with_all_fields():
    """Tests SnapPoint creation with all fields."""
    snap_point = SnapPoint(
        x=5.0,
        y=10.0,
        line_type=SnapLineType.EQUIDISTANT,
        source="test_source",
        spacing=2.5,
        is_horizontal=True,
        pattern_coords=(0.0, 2.5, 5.0, 7.5, 10.0),
        axis_coord=5.0,
    )
    assert snap_point.x == 5.0
    assert snap_point.y == 10.0
    assert snap_point.line_type == SnapLineType.EQUIDISTANT
    assert snap_point.source == "test_source"
    assert snap_point.spacing == 2.5
    assert snap_point.is_horizontal is True
    assert snap_point.pattern_coords == (0.0, 2.5, 5.0, 7.5, 10.0)
    assert snap_point.axis_coord == 5.0


def test_snap_line_creation_horizontal():
    """Tests SnapLine creation for horizontal line."""
    snap_line = SnapLine(
        is_horizontal=True,
        coordinate=10.5,
        line_type=SnapLineType.ENTITY_POINT,
        source="test_source",
    )
    assert snap_line.is_horizontal is True
    assert snap_line.coordinate == 10.5
    assert snap_line.line_type == SnapLineType.ENTITY_POINT
    assert snap_line.source == "test_source"
    assert snap_line.style == SNAP_LINE_STYLES[SnapLineType.ENTITY_POINT]


def test_snap_line_creation_vertical():
    """Tests SnapLine creation for vertical line."""
    snap_line = SnapLine(
        is_horizontal=False,
        coordinate=20.3,
        line_type=SnapLineType.CENTER,
    )
    assert snap_line.is_horizontal is False
    assert snap_line.coordinate == 20.3
    assert snap_line.line_type == SnapLineType.CENTER
    assert snap_line.source is None


def test_snap_line_distance_to_horizontal():
    """Tests SnapLine.distance_to for horizontal line."""
    snap_line = SnapLine(
        is_horizontal=True,
        coordinate=10.0,
        line_type=SnapLineType.ENTITY_POINT,
    )
    assert snap_line.distance_to(5.0, 12.0) == 2.0
    assert snap_line.distance_to(5.0, 8.0) == 2.0
    assert snap_line.distance_to(5.0, 10.0) == 0.0


def test_snap_line_distance_to_vertical():
    """Tests SnapLine.distance_to for vertical line."""
    snap_line = SnapLine(
        is_horizontal=False, coordinate=20.0, line_type=SnapLineType.CENTER
    )
    assert snap_line.distance_to(22.0, 5.0) == 2.0
    assert snap_line.distance_to(18.0, 5.0) == 2.0
    assert snap_line.distance_to(20.0, 5.0) == 0.0


def test_snap_line_get_snap_position_horizontal():
    """Tests SnapLine.get_snap_position for horizontal line."""
    snap_line = SnapLine(
        is_horizontal=True,
        coordinate=10.0,
        line_type=SnapLineType.ENTITY_POINT,
    )
    assert snap_line.get_snap_position(5.0, 15.0) == (5.0, 10.0)
    assert snap_line.get_snap_position(20.0, 8.0) == (20.0, 10.0)


def test_snap_line_get_snap_position_vertical():
    """Tests SnapLine.get_snap_position for vertical line."""
    snap_line = SnapLine(
        is_horizontal=False, coordinate=20.0, line_type=SnapLineType.CENTER
    )
    assert snap_line.get_snap_position(25.0, 5.0) == (20.0, 5.0)
    assert snap_line.get_snap_position(18.0, 10.0) == (20.0, 10.0)


def test_snap_result_no_snap():
    """Tests SnapResult.no_snap classmethod."""
    result = SnapResult.no_snap((10.0, 20.0))
    assert result.snapped is False
    assert result.position == (10.0, 20.0)
    assert result.snap_lines == []
    assert result.snap_points == []
    assert result.primary_snap_line is None
    assert result.primary_snap_point is None
    assert result.distance == float("inf")


def test_snap_result_from_snap_line():
    """Tests SnapResult.from_snap_line classmethod."""
    snap_line = SnapLine(
        is_horizontal=True,
        coordinate=10.0,
        line_type=SnapLineType.ENTITY_POINT,
    )
    result = SnapResult.from_snap_line(snap_line, (5.0, 15.0), 5.0)
    assert result.snapped is True
    assert result.position == (5.0, 10.0)
    assert len(result.snap_lines) == 1
    assert result.primary_snap_line == snap_line
    assert result.primary_snap_point is None
    assert result.distance == 5.0


def test_snap_result_from_snap_point():
    """Tests SnapResult.from_snap_point classmethod."""
    snap_point = SnapPoint(x=10.0, y=20.0, line_type=SnapLineType.ENTITY_POINT)
    result = SnapResult.from_snap_point(snap_point, 5.0)
    assert result.snapped is True
    assert result.position == (10.0, 20.0)
    assert len(result.snap_points) == 1
    assert result.primary_snap_point == snap_point
    assert result.primary_snap_line is None
    assert result.distance == 5.0
    assert result.snap_lines == []


def test_snap_result_from_snap_point_with_lines():
    """Tests SnapResult.from_snap_point with snap lines."""
    snap_point = SnapPoint(x=10.0, y=20.0, line_type=SnapLineType.ENTITY_POINT)
    snap_line1 = SnapLine(
        is_horizontal=True,
        coordinate=20.0,
        line_type=SnapLineType.ENTITY_POINT,
    )
    snap_line2 = SnapLine(
        is_horizontal=False,
        coordinate=10.0,
        line_type=SnapLineType.ENTITY_POINT,
    )
    result = SnapResult.from_snap_point(
        snap_point, 3.0, [snap_line1, snap_line2]
    )
    assert result.snapped is True
    assert result.position == (10.0, 20.0)
    assert len(result.snap_points) == 1
    assert len(result.snap_lines) == 2
    assert snap_line1 in result.snap_lines
    assert snap_line2 in result.snap_lines
    assert result.primary_snap_point == snap_point
    assert result.primary_snap_line is None
    assert result.distance == 3.0


def test_snap_result_defaults():
    """Tests SnapResult default values."""
    result = SnapResult()
    assert result.snapped is False
    assert result.position == (0.0, 0.0)
    assert result.snap_lines == []
    assert result.snap_points == []
    assert result.primary_snap_line is None
    assert result.primary_snap_point is None
    assert result.distance == float("inf")


def test_drag_context_creation_empty():
    """Tests DragContext creation with no arguments."""
    context = DragContext()
    assert context.dragged_point_ids == set()
    assert context.dragged_entity_ids == set()
    assert context.initial_positions == {}


def test_drag_context_creation_with_args():
    """Tests DragContext creation with arguments."""
    context = DragContext(
        dragged_point_ids={1, 2, 3},
        dragged_entity_ids={10, 20},
        initial_positions={1: (0.0, 0.0), 2: (10.0, 10.0)},
    )
    assert context.dragged_point_ids == {1, 2, 3}
    assert context.dragged_entity_ids == {10, 20}
    assert context.initial_positions == {1: (0.0, 0.0), 2: (10.0, 10.0)}


def test_drag_context_is_point_dragged():
    """Tests DragContext.is_point_dragged method."""
    context = DragContext(dragged_point_ids={1, 2, 3})
    assert context.is_point_dragged(1) is True
    assert context.is_point_dragged(2) is True
    assert context.is_point_dragged(3) is True
    assert context.is_point_dragged(4) is False


def test_drag_context_is_entity_dragged():
    """Tests DragContext.is_entity_dragged method."""
    context = DragContext(dragged_entity_ids={10, 20})
    assert context.is_entity_dragged(10) is True
    assert context.is_entity_dragged(20) is True
    assert context.is_entity_dragged(30) is False


def test_snap_line_frozen():
    """Tests that SnapLine is immutable."""
    snap_line = SnapLine(
        is_horizontal=True,
        coordinate=10.0,
        line_type=SnapLineType.ENTITY_POINT,
    )
    with pytest.raises(AttributeError):
        snap_line.coordinate = 20.0  # type: ignore


def test_snap_point_frozen():
    """Tests that SnapPoint is immutable."""
    snap_point = SnapPoint(x=10.0, y=20.0, line_type=SnapLineType.ENTITY_POINT)
    with pytest.raises(AttributeError):
        snap_point.x = 30.0  # type: ignore


def test_snap_line_style_frozen():
    """Tests that SnapLineStyle is immutable."""
    style = SnapLineStyle()
    with pytest.raises(AttributeError):
        style.color = (1.0, 0.0, 0.0, 1.0)  # type: ignore
