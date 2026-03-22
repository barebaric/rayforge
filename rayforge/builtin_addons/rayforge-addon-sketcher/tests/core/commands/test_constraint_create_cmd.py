import pytest

from sketcher.core import Sketch
from sketcher.core.commands import CreateOrEditConstraintCommand
from sketcher.core.constraints import (
    DiameterConstraint,
    DistanceConstraint,
    RadiusConstraint,
)


@pytest.fixture
def sketch():
    return Sketch()


def test_create_distance_constraint_for_line(sketch):
    p1_id = sketch.add_point(0, 0)
    p2_id = sketch.add_point(30, 40)
    line_id = sketch.add_line(p1_id, p2_id)
    line = sketch.registry.get_entity(line_id)

    cmd = CreateOrEditConstraintCommand(sketch, line)
    cmd.execute()

    assert cmd.is_new_constraint
    assert cmd.constraint is not None
    assert isinstance(cmd.constraint, DistanceConstraint)
    assert cmd.constraint.value == pytest.approx(50.0)
    assert len(sketch.constraints) == 1


def test_create_diameter_constraint_for_circle(sketch):
    center_id = sketch.add_point(0, 0)
    radius_id = sketch.add_point(10, 0)
    circle_id = sketch.add_circle(center_id, radius_id)
    circle = sketch.registry.get_entity(circle_id)

    cmd = CreateOrEditConstraintCommand(sketch, circle)
    cmd.execute()

    assert cmd.is_new_constraint
    assert cmd.constraint is not None
    assert isinstance(cmd.constraint, DiameterConstraint)
    assert cmd.constraint.value == pytest.approx(20.0)
    assert len(sketch.constraints) == 1


def test_create_radius_constraint_for_arc(sketch):
    center_id = sketch.add_point(0, 0)
    start_id = sketch.add_point(10, 0)
    end_id = sketch.add_point(0, 10)
    arc_id = sketch.add_arc(center_id, start_id, end_id)
    arc = sketch.registry.get_entity(arc_id)

    cmd = CreateOrEditConstraintCommand(sketch, arc)
    cmd.execute()

    assert cmd.is_new_constraint
    assert cmd.constraint is not None
    assert isinstance(cmd.constraint, RadiusConstraint)
    assert cmd.constraint.value == pytest.approx(10.0)
    assert len(sketch.constraints) == 1


def test_returns_existing_constraint_for_line(sketch):
    p1_id = sketch.add_point(0, 0)
    p2_id = sketch.add_point(30, 40)
    line_id = sketch.add_line(p1_id, p2_id)
    line = sketch.registry.get_entity(line_id)

    existing = DistanceConstraint(p1_id, p2_id, 50.0)
    sketch.constraints.append(existing)

    cmd = CreateOrEditConstraintCommand(sketch, line)
    cmd.execute()

    assert not cmd.is_new_constraint
    assert cmd.constraint is existing
    assert len(sketch.constraints) == 1


def test_returns_existing_constraint_for_circle(sketch):
    center_id = sketch.add_point(0, 0)
    radius_id = sketch.add_point(10, 0)
    circle_id = sketch.add_circle(center_id, radius_id)
    circle = sketch.registry.get_entity(circle_id)

    existing = DiameterConstraint(circle_id, 20.0)
    sketch.constraints.append(existing)

    cmd = CreateOrEditConstraintCommand(sketch, circle)
    cmd.execute()

    assert not cmd.is_new_constraint
    assert cmd.constraint is existing
    assert len(sketch.constraints) == 1


def test_returns_existing_constraint_for_arc(sketch):
    center_id = sketch.add_point(0, 0)
    start_id = sketch.add_point(10, 0)
    end_id = sketch.add_point(0, 10)
    arc_id = sketch.add_arc(center_id, start_id, end_id)
    arc = sketch.registry.get_entity(arc_id)

    existing = RadiusConstraint(arc_id, 10.0)
    sketch.constraints.append(existing)

    cmd = CreateOrEditConstraintCommand(sketch, arc)
    cmd.execute()

    assert not cmd.is_new_constraint
    assert cmd.constraint is existing
    assert len(sketch.constraints) == 1


def test_undo_removes_created_constraint(sketch):
    p1_id = sketch.add_point(0, 0)
    p2_id = sketch.add_point(30, 40)
    line_id = sketch.add_line(p1_id, p2_id)
    line = sketch.registry.get_entity(line_id)

    cmd = CreateOrEditConstraintCommand(sketch, line)
    cmd.execute()

    assert len(sketch.constraints) == 1

    cmd.undo()

    assert len(sketch.constraints) == 0


def test_undo_does_not_affect_existing_constraint(sketch):
    p1_id = sketch.add_point(0, 0)
    p2_id = sketch.add_point(30, 40)
    line_id = sketch.add_line(p1_id, p2_id)
    line = sketch.registry.get_entity(line_id)

    existing = DistanceConstraint(p1_id, p2_id, 50.0)
    sketch.constraints.append(existing)

    cmd = CreateOrEditConstraintCommand(sketch, line)
    cmd.execute()
    cmd.undo()

    assert len(sketch.constraints) == 1
    assert sketch.constraints[0] is existing


def test_redo_recreates_constraint(sketch):
    p1_id = sketch.add_point(0, 0)
    p2_id = sketch.add_point(30, 40)
    line_id = sketch.add_line(p1_id, p2_id)
    line = sketch.registry.get_entity(line_id)

    cmd = CreateOrEditConstraintCommand(sketch, line)
    cmd.execute()
    cmd.undo()
    cmd.execute()

    assert len(sketch.constraints) == 1
    assert isinstance(sketch.constraints[0], DistanceConstraint)


def test_command_label_uses_type_name(sketch):
    p1_id = sketch.add_point(0, 0)
    p2_id = sketch.add_point(30, 40)
    line_id = sketch.add_line(p1_id, p2_id)
    line = sketch.registry.get_entity(line_id)

    cmd = CreateOrEditConstraintCommand(sketch, line)
    label = cmd._get_command_label(DistanceConstraint(p1_id, p2_id, 50.0))

    assert label == "Add Distance"


def test_command_label_for_radius(sketch):
    center_id = sketch.add_point(0, 0)
    start_id = sketch.add_point(10, 0)
    end_id = sketch.add_point(0, 10)
    arc_id = sketch.add_arc(center_id, start_id, end_id)
    arc = sketch.registry.get_entity(arc_id)

    cmd = CreateOrEditConstraintCommand(sketch, arc)
    label = cmd._get_command_label(RadiusConstraint(arc_id, 10.0))

    assert label == "Add Radius"


def test_command_label_for_diameter(sketch):
    center_id = sketch.add_point(0, 0)
    radius_id = sketch.add_point(10, 0)
    circle_id = sketch.add_circle(center_id, radius_id)
    circle = sketch.registry.get_entity(circle_id)

    cmd = CreateOrEditConstraintCommand(sketch, circle)
    label = cmd._get_command_label(DiameterConstraint(circle_id, 20.0))

    assert label == "Add Diameter"


def test_get_constraint_for_entity_line(sketch):
    p1_id = sketch.add_point(0, 0)
    p2_id = sketch.add_point(30, 40)
    line_id = sketch.add_line(p1_id, p2_id)
    line = sketch.registry.get_entity(line_id)

    existing = DistanceConstraint(p1_id, p2_id, 50.0)
    sketch.constraints.append(existing)

    result = CreateOrEditConstraintCommand.get_constraint_for_entity(
        sketch, line
    )

    assert result is existing


def test_get_constraint_for_entity_circle(sketch):
    center_id = sketch.add_point(0, 0)
    radius_id = sketch.add_point(10, 0)
    circle_id = sketch.add_circle(center_id, radius_id)
    circle = sketch.registry.get_entity(circle_id)

    existing = DiameterConstraint(circle_id, 20.0)
    sketch.constraints.append(existing)

    result = CreateOrEditConstraintCommand.get_constraint_for_entity(
        sketch, circle
    )

    assert result is existing


def test_get_constraint_for_entity_arc(sketch):
    center_id = sketch.add_point(0, 0)
    start_id = sketch.add_point(10, 0)
    end_id = sketch.add_point(0, 10)
    arc_id = sketch.add_arc(center_id, start_id, end_id)
    arc = sketch.registry.get_entity(arc_id)

    existing = RadiusConstraint(arc_id, 10.0)
    sketch.constraints.append(existing)

    result = CreateOrEditConstraintCommand.get_constraint_for_entity(
        sketch, arc
    )

    assert result is existing


def test_get_constraint_for_entity_none(sketch):
    p1_id = sketch.add_point(0, 0)
    p2_id = sketch.add_point(30, 40)
    line_id = sketch.add_line(p1_id, p2_id)
    line = sketch.registry.get_entity(line_id)

    result = CreateOrEditConstraintCommand.get_constraint_for_entity(
        sketch, line
    )

    assert result is None


def test_create_constraint_for_entity_with_initial_value(sketch):
    p1_id = sketch.add_point(0, 0)
    p2_id = sketch.add_point(30, 40)
    line_id = sketch.add_line(p1_id, p2_id)
    line = sketch.registry.get_entity(line_id)

    result = CreateOrEditConstraintCommand.create_constraint_for_entity(
        sketch, line, initial_value=100.0
    )

    assert result is not None
    assert isinstance(result, DistanceConstraint)
    assert result.value == 100.0
