from rayforge.core.sketcher import Sketch
from rayforge.core.sketcher.commands import TextBoxCommand
from rayforge.core.sketcher.constraints import (
    AspectRatioConstraint,
    HorizontalConstraint,
    PerpendicularConstraint,
    ParallelogramConstraint,
)
from rayforge.core.sketcher.entities import Line, TextBoxEntity


def test_text_box_calculate_geometry_default():
    """Test static calculation with default dimensions."""
    result = TextBoxCommand.calculate_geometry((0, 0), 10.0, 10.0)
    assert result is not None
    points = result["points"]
    entities = result["entities"]
    constraints = result["constraints"]

    assert len(points) == 4
    assert len(entities) == 5
    assert len(constraints) == 4

    assert points[0].x == 0 and points[0].y == 0
    assert points[1].x == 10 and points[1].y == 0
    assert points[2].x == 0 and points[2].y == 10
    assert points[3].x == 10 and points[3].y == 10

    assert result["text_box_id"] == -9


def test_text_box_calculate_geometry_custom_dimensions():
    """Test static calculation with custom dimensions."""
    result = TextBoxCommand.calculate_geometry((5, 10), 20.0, 15.0)
    assert result is not None
    points = result["points"]

    assert points[0].x == 5 and points[0].y == 10
    assert points[1].x == 25 and points[1].y == 10
    assert points[2].x == 5 and points[2].y == 25
    assert points[3].x == 25 and points[3].y == 25


def test_text_box_calculate_geometry_entities():
    """Test that correct entities are created."""
    result = TextBoxCommand.calculate_geometry((0, 0), 10.0, 10.0)
    entities = result["entities"]

    lines = [e for e in entities if isinstance(e, Line)]
    text_boxes = [e for e in entities if isinstance(e, TextBoxEntity)]

    assert len(lines) == 4
    assert len(text_boxes) == 1

    assert all(line.construction for line in lines)


def test_text_box_calculate_geometry_constraints():
    """Test that correct constraints are created."""
    result = TextBoxCommand.calculate_geometry((0, 0), 10.0, 10.0)
    constraints = result["constraints"]

    aspect_ratio = [
        c for c in constraints if isinstance(c, AspectRatioConstraint)
    ]
    parallelogram = [
        c for c in constraints if isinstance(c, ParallelogramConstraint)
    ]
    horizontal = [
        c for c in constraints if isinstance(c, HorizontalConstraint)
    ]
    perpendicular = [
        c for c in constraints if isinstance(c, PerpendicularConstraint)
    ]

    assert len(aspect_ratio) == 1
    assert aspect_ratio[0].user_visible is True

    assert len(parallelogram) == 1
    assert parallelogram[0].user_visible is False

    assert len(horizontal) == 1
    assert horizontal[0].user_visible is True

    assert len(perpendicular) == 1
    assert perpendicular[0].user_visible is True


def test_text_box_command_initialization():
    """Test command initialization."""
    sketch = Sketch()
    cmd = TextBoxCommand(sketch, (0, 0))

    assert cmd.sketch is sketch
    assert cmd.origin == (0, 0)
    assert cmd.width == 10.0
    assert cmd.height == 10.0
    assert cmd.add_cmd is None
    assert cmd.text_box_id is None


def test_text_box_command_initialization_custom_dimensions():
    """Test command initialization with custom dimensions."""
    sketch = Sketch()
    cmd = TextBoxCommand(sketch, (5, 10), 20.0, 15.0)

    assert cmd.origin == (5, 10)
    assert cmd.width == 20.0
    assert cmd.height == 15.0


def test_text_box_command_execute():
    """Test command execution."""
    sketch = Sketch()
    cmd = TextBoxCommand(sketch, (0, 0), 10.0, 10.0)
    cmd.execute()

    assert len(sketch.registry.points) == 5
    assert len(sketch.registry.entities) == 5
    assert len(sketch.constraints) == 4
    assert cmd.text_box_id is not None


def test_text_box_command_execute_custom_dimensions():
    """Test command execution with custom dimensions."""
    sketch = Sketch()
    cmd = TextBoxCommand(sketch, (5, 10), 20.0, 15.0)
    cmd.execute()

    assert len(sketch.registry.points) == 5
    assert len(sketch.registry.entities) == 5
    assert len(sketch.constraints) == 4
    assert cmd.text_box_id is not None

    text_box = sketch.registry.get_entity(cmd.text_box_id)
    assert isinstance(text_box, TextBoxEntity)


def test_text_box_command_undo():
    """Test command undo."""
    sketch = Sketch()
    cmd = TextBoxCommand(sketch, (0, 0), 10.0, 10.0)
    cmd.execute()

    assert len(sketch.registry.points) == 5
    assert len(sketch.registry.entities) == 5
    assert len(sketch.constraints) == 4

    cmd.undo()

    assert len(sketch.registry.points) == 1
    assert len(sketch.registry.entities) == 0
    assert len(sketch.constraints) == 0


def test_text_box_command_execute_after_undo():
    """Test command execution after undo."""
    sketch = Sketch()
    cmd = TextBoxCommand(sketch, (0, 0), 10.0, 10.0)
    cmd.execute()

    initial_points = len(sketch.registry.points)
    initial_entities = len(sketch.registry.entities)
    initial_constraints = len(sketch.constraints)

    cmd.undo()
    cmd.execute()

    assert len(sketch.registry.points) == initial_points
    assert len(sketch.registry.entities) == initial_entities
    assert len(sketch.constraints) == initial_constraints


def test_text_box_command_construction_lines():
    """Test that construction lines are created."""
    sketch = Sketch()
    cmd = TextBoxCommand(sketch, (0, 0), 10.0, 10.0)
    cmd.execute()

    lines = [
        e
        for e in sketch.registry.entities
        if isinstance(e, Line) and e.construction
    ]

    assert len(lines) == 4


def test_text_box_command_text_box_entity():
    """Test that TextBoxEntity is created with correct properties."""
    sketch = Sketch()
    cmd = TextBoxCommand(sketch, (0, 0), 10.0, 10.0)
    cmd.execute()

    assert cmd.text_box_id is not None
    text_box = sketch.registry.get_entity(cmd.text_box_id)
    assert isinstance(text_box, TextBoxEntity)
    assert text_box.content == ""
    assert text_box.construction is False
    assert len(text_box.construction_line_ids) == 4
