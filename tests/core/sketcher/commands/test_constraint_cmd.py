import pytest

from rayforge.core.sketcher import Sketch
from rayforge.core.sketcher.commands import ModifyConstraintCommand
from rayforge.core.sketcher.constraints import DistanceConstraint


@pytest.fixture
def sketch():
    """Create a basic sketch for testing."""
    return Sketch()


@pytest.fixture
def constraint(sketch):
    """Create a distance constraint for testing."""
    p1 = sketch.add_point(0.0, 0.0)
    p2 = sketch.add_point(10.0, 0.0)
    return DistanceConstraint(p1, p2, 10.0)


def test_modify_constraint_command_initialization(sketch, constraint):
    """Test that ModifyConstraintCommand initializes correctly."""
    cmd = ModifyConstraintCommand(
        sketch, constraint, 20.0, "20", "Edit Constraint"
    )

    assert cmd.sketch is sketch
    assert cmd.constraint is constraint
    assert cmd.new_value == 20.0
    assert cmd.new_expression == "20"
    assert cmd.old_value == 10.0
    assert cmd.old_expression is None


def test_modify_constraint_command_with_existing_expression(sketch):
    """Test initialization when constraint has an expression."""
    p1 = sketch.add_point(0.0, 0.0)
    p2 = sketch.add_point(10.0, 0.0)
    constraint = DistanceConstraint(p1, p2, 10.0)
    constraint.expression = "10"

    cmd = ModifyConstraintCommand(sketch, constraint, 20.0, "20")

    assert cmd.old_expression == "10"


def test_modify_constraint_command_execute(sketch, constraint):
    """Test that execute modifies the constraint value."""
    cmd = ModifyConstraintCommand(sketch, constraint, 20.0, "20")

    assert constraint.value == 10.0
    assert constraint.expression is None

    cmd.execute()

    assert constraint.value == 20.0
    assert constraint.expression == "20"


def test_modify_constraint_cmd_execute_without_expression(sketch, constraint):
    """Test execute when no new expression is provided."""
    cmd = ModifyConstraintCommand(sketch, constraint, 20.0)

    cmd.execute()

    assert constraint.value == 20.0
    assert constraint.expression is None


def test_modify_constraint_command_undo(sketch, constraint):
    """Test that undo restores the original constraint value."""
    cmd = ModifyConstraintCommand(sketch, constraint, 20.0, "20")

    cmd.execute()
    assert constraint.value == 20.0
    assert constraint.expression == "20"

    cmd.undo()

    assert constraint.value == 10.0
    assert constraint.expression is None


def test_modify_constraint_command_undo_with_expression(sketch):
    """Test undo when original constraint had an expression."""
    p1 = sketch.add_point(0.0, 0.0)
    p2 = sketch.add_point(10.0, 0.0)
    constraint = DistanceConstraint(p1, p2, 10.0)
    constraint.expression = "10"

    cmd = ModifyConstraintCommand(sketch, constraint, 20.0, "20")

    cmd.execute()
    cmd.undo()

    assert constraint.value == 10.0
    assert constraint.expression == "10"


def test_modify_constraint_command_execute_undo_cycle(sketch, constraint):
    """Test that execute and undo can be called multiple times."""
    cmd = ModifyConstraintCommand(sketch, constraint, 20.0, "20")

    for _ in range(3):
        cmd.execute()
        assert constraint.value == 20.0
        assert constraint.expression == "20"

        cmd.undo()
        assert constraint.value == 10.0
        assert constraint.expression is None
