from sketcher.core import Sketch
from sketcher.core.commands import GridCommand
from sketcher.core.constraints import HorizontalConstraint, VerticalConstraint
from sketcher.core.entities import Line


def test_grid_calculate_geometry_2x2():
    result = GridCommand.calculate_geometry(2, 2, (0, 0), 10, 10)
    assert result is not None
    assert len(result["points"]) == 4
    assert len(result["entities"]) == 4
    assert len(result["constraints"]) == 4


def test_grid_calculate_geometry_3x3():
    result = GridCommand.calculate_geometry(3, 3, (0, 0), 10, 10)
    assert result is not None
    assert len(result["points"]) == 9
    assert len(result["entities"]) == 12
    assert len(result["constraints"]) == 12


def test_grid_calculate_geometry_with_origin():
    result = GridCommand.calculate_geometry(2, 2, (50, 100), 10, 20)
    assert result is not None
    points = result["points"]
    assert points[0].x == 50 and points[0].y == 100
    assert points[1].x == 60 and points[1].y == 100
    assert points[2].x == 50 and points[2].y == 120
    assert points[3].x == 60 and points[3].y == 120


def test_grid_calculate_geometry_construction_flag():
    result = GridCommand.calculate_geometry(
        2, 2, (0, 0), 10, 10, construction=True
    )
    assert result is not None
    for entity in result["entities"]:
        assert entity.construction is True

    result = GridCommand.calculate_geometry(
        2, 2, (0, 0), 10, 10, construction=False
    )
    assert result is not None
    for entity in result["entities"]:
        assert entity.construction is False


def test_grid_calculate_geometry_invalid_rows():
    assert GridCommand.calculate_geometry(1, 2, (0, 0), 10, 10) is None
    assert GridCommand.calculate_geometry(0, 2, (0, 0), 10, 10) is None


def test_grid_calculate_geometry_invalid_cols():
    assert GridCommand.calculate_geometry(2, 1, (0, 0), 10, 10) is None
    assert GridCommand.calculate_geometry(2, 0, (0, 0), 10, 10) is None


def test_grid_calculate_geometry_invalid_cell_size():
    assert GridCommand.calculate_geometry(2, 2, (0, 0), 0, 10) is None
    assert GridCommand.calculate_geometry(2, 2, (0, 0), 10, 0) is None
    assert GridCommand.calculate_geometry(2, 2, (0, 0), -5, 10) is None


def test_grid_command_execute():
    sketch = Sketch()
    initial_count = len(sketch.registry.points)
    cmd = GridCommand(sketch, 3, 4, (0, 0), 10, 10)
    cmd.execute()

    assert len(sketch.registry.points) == initial_count + 12
    assert len(sketch.registry.entities) == 17


def test_grid_command_execute_with_origin():
    sketch = Sketch()
    initial_count = len(sketch.registry.points)
    cmd = GridCommand(sketch, 2, 2, (100, 200), 50, 75)
    cmd.execute()

    points = sketch.registry.points
    assert len(points) == initial_count + 4

    p0 = points[initial_count]
    p3 = points[initial_count + 3]
    assert p0.x == 100 and p0.y == 200
    assert p3.x == 150 and p3.y == 275


def test_grid_command_execute_construction():
    sketch = Sketch()
    cmd = GridCommand(sketch, 2, 2, (0, 0), 10, 10, construction=True)
    cmd.execute()

    for entity in sketch.registry.entities:
        assert entity.construction is True


def test_grid_command_execute_normal():
    sketch = Sketch()
    cmd = GridCommand(sketch, 2, 2, (0, 0), 10, 10, construction=False)
    cmd.execute()

    for entity in sketch.registry.entities:
        assert entity.construction is False


def test_grid_command_undo():
    sketch = Sketch()
    initial_point_count = len(sketch.registry.points)
    initial_entity_count = len(sketch.registry.entities)

    cmd = GridCommand(sketch, 2, 2, (0, 0), 10, 10)
    cmd.execute()

    assert len(sketch.registry.points) == initial_point_count + 4
    assert len(sketch.registry.entities) == initial_entity_count + 4

    cmd.undo()

    assert len(sketch.registry.points) == initial_point_count
    assert len(sketch.registry.entities) == initial_entity_count


def test_grid_command_invalid_does_nothing():
    sketch = Sketch()
    initial_point_count = len(sketch.registry.points)

    cmd = GridCommand(sketch, 1, 1, (0, 0), 10, 10)
    cmd.execute()

    assert len(sketch.registry.points) == initial_point_count


def test_grid_command_horizontal_line_count():
    sketch = Sketch()
    cmd = GridCommand(sketch, 3, 5, (0, 0), 10, 10)
    cmd.execute()

    horizontal_lines = [
        e
        for e in sketch.registry.entities
        if isinstance(e, Line)
        and abs(
            sketch.registry.get_point(e.p1_idx).y
            - sketch.registry.get_point(e.p2_idx).y
        )
        < 1e-6
    ]
    expected_horizontal = 3 * (5 - 1)
    assert len(horizontal_lines) == expected_horizontal


def test_grid_command_vertical_line_count():
    sketch = Sketch()
    cmd = GridCommand(sketch, 3, 5, (0, 0), 10, 10)
    cmd.execute()

    vertical_lines = [
        e
        for e in sketch.registry.entities
        if isinstance(e, Line)
        and abs(
            sketch.registry.get_point(e.p1_idx).x
            - sketch.registry.get_point(e.p2_idx).x
        )
        < 1e-6
    ]
    expected_vertical = 5 * (3 - 1)
    assert len(vertical_lines) == expected_vertical


def test_grid_command_creates_horizontal_constraints():
    sketch = Sketch()
    cmd = GridCommand(sketch, 3, 5, (0, 0), 10, 10)
    cmd.execute()

    horizontal_constraints = [
        c for c in sketch.constraints if isinstance(c, HorizontalConstraint)
    ]
    expected_horizontal = 3 * (5 - 1)
    assert len(horizontal_constraints) == expected_horizontal


def test_grid_command_creates_vertical_constraints():
    sketch = Sketch()
    cmd = GridCommand(sketch, 3, 5, (0, 0), 10, 10)
    cmd.execute()

    vertical_constraints = [
        c for c in sketch.constraints if isinstance(c, VerticalConstraint)
    ]
    expected_vertical = 5 * (3 - 1)
    assert len(vertical_constraints) == expected_vertical


def test_grid_command_constraints_count():
    sketch = Sketch()
    initial_constraint_count = len(sketch.constraints)
    cmd = GridCommand(sketch, 2, 3, (0, 0), 10, 10)
    cmd.execute()

    horizontal_lines = 2 * (3 - 1)
    vertical_lines = 3 * (2 - 1)
    expected_constraints = horizontal_lines + vertical_lines
    assert (
        len(sketch.constraints)
        == initial_constraint_count + expected_constraints
    )


def test_grid_command_undo_removes_constraints():
    sketch = Sketch()
    initial_constraint_count = len(sketch.constraints)

    cmd = GridCommand(sketch, 2, 2, (0, 0), 10, 10)
    cmd.execute()

    assert len(sketch.constraints) == initial_constraint_count + 4

    cmd.undo()

    assert len(sketch.constraints) == initial_constraint_count
