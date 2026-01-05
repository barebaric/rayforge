from unittest.mock import MagicMock

import pytest

from rayforge.core.sketcher import Sketch
from rayforge.core.sketcher.commands import FilletCommand
from rayforge.core.sketcher.constraints import (
    CollinearConstraint,
    EqualDistanceConstraint,
    TangentConstraint,
)
from rayforge.core.sketcher.entities import Line, Arc
from rayforge.ui_gtk.sketcher.sketchelement import SketchElement

# Constant for consistency
DEFAULT_FILLET_RADIUS = 10.0


@pytest.fixture
def sketch_with_corner():
    """Creates a sketch with two lines forming a corner at (0,0)."""
    s = Sketch()
    # p1=(-100, 0), p2=(0, 0) [corner], p3=(0, 100)
    # IDs: origin=0, p1=1, corner=2, p3=3, line1=4, line2=5
    p1_id = s.add_point(-100, 0)
    corner_pid = s.add_point(0, 0)
    p3_id = s.add_point(0, 100)

    # line1 from p1 to corner, line2 from corner to p3
    line1_id = s.add_line(p1_id, corner_pid)
    line2_id = s.add_line(corner_pid, p3_id)

    return s, corner_pid, line1_id, line2_id


@pytest.fixture
def element_with_corner(sketch_with_corner):
    """Creates a SketchElement with a corner and mocked editor."""
    sketch, _, _, _ = sketch_with_corner
    element = SketchElement(sketch=sketch)
    element.editor = MagicMock()
    element.execute_command = element.editor.history_manager.execute
    return element


def test_is_action_supported_fillet_valid(
    element_with_corner, sketch_with_corner
):
    """Test fillet is supported when a valid corner junction is selected."""
    _, corner_pid, _, _ = sketch_with_corner
    element_with_corner.selection.select_junction(corner_pid, is_multi=False)

    assert element_with_corner.is_action_supported("fillet") is True


@pytest.mark.parametrize(
    "setup_selection",
    [
        "no_selection",
        "point_selection",
        "single_line_junction",
        "triple_line_junction",
    ],
)
def test_is_action_supported_fillet_invalid(setup_selection):
    """Test fillet is not supported for invalid selections."""
    s = Sketch()
    p1 = s.add_point(0, 0)
    p2 = s.add_point(10, 0)
    p3 = s.add_point(10, 10)
    p4 = s.add_point(0, 10)
    s.add_line(p1, p2)

    element = SketchElement(sketch=s)

    if setup_selection == "no_selection":
        pass  # Selection is already empty
    elif setup_selection == "point_selection":
        element.selection.select_point(p1, is_multi=False)
    elif setup_selection == "single_line_junction":
        # A 'junction' with only one line is just a point
        element.selection.select_junction(p1, is_multi=False)
    elif setup_selection == "triple_line_junction":
        s.add_line(p1, p3)
        s.add_line(p1, p4)
        element.selection.select_junction(p1, is_multi=False)

    assert element.is_action_supported("fillet") is False


def test_add_fillet_action_executes_command(
    element_with_corner, sketch_with_corner
):
    """Test that add_fillet_action creates and executes a FilletCommand."""
    _, corner_pid, _, _ = sketch_with_corner
    element_with_corner.selection.select_junction(corner_pid, is_multi=False)

    element_with_corner.add_fillet_action()

    element_with_corner.execute_command.assert_called_once()
    command_instance = element_with_corner.execute_command.call_args[0][0]
    assert isinstance(command_instance, FilletCommand)


def test_fillet_command_execute(sketch_with_corner):
    """Test the direct execution of FilletCommand."""
    sketch, corner_pid, line1_id, line2_id = sketch_with_corner

    initial_points_count = len(sketch.registry.points)
    initial_entities_count = len(sketch.registry.entities)
    initial_constraints_count = len(sketch.constraints)

    command = FilletCommand(
        sketch, corner_pid, line1_id, line2_id, DEFAULT_FILLET_RADIUS
    )
    command.execute()

    # Verify additions:
    # Points: +3 (Tangent1, Tangent2, Center)
    # Entities: +1 total (2 lines removed, 2 lines + 1 arc added)
    # Constraints: +5 (2 Tangent, 2 Collinear, 1 EqualDistance)
    assert len(sketch.registry.points) == initial_points_count + 3
    assert len(sketch.registry.entities) == initial_entities_count + 1
    assert len(sketch.constraints) == initial_constraints_count + 5

    # Verify new geometry types
    assert len(command.added_entities) == 3
    # First two are lines connecting original points to tangents
    assert isinstance(command.added_entities[0], Line)
    assert isinstance(command.added_entities[1], Line)
    # Third is the Arc
    fillet_arc = command.added_entities[2]
    assert isinstance(fillet_arc, Arc)

    # Verify constraints
    new_constraints = sketch.constraints[-5:]

    # We expect 2 Tangent, 2 Collinear, 1 EqualDistance.
    # We check presence using sets or types.
    constraint_types = [type(c) for c in new_constraints]
    assert constraint_types.count(TangentConstraint) == 2
    assert constraint_types.count(CollinearConstraint) == 2
    assert constraint_types.count(EqualDistanceConstraint) == 1

    # Verify original lines were removed
    assert sketch.registry.get_entity(line1_id) is None
    assert sketch.registry.get_entity(line2_id) is None


def test_fillet_command_undo(sketch_with_corner):
    """Test that undoing a FilletCommand restores the original state."""
    sketch, corner_pid, line1_id, line2_id = sketch_with_corner

    line1 = sketch.registry.get_entity(line1_id)
    line2 = sketch.registry.get_entity(line2_id)
    assert isinstance(line1, Line)
    assert isinstance(line2, Line)

    # Store initial state
    initial_state = {
        "points_count": len(sketch.registry.points),
        "entities_count": len(sketch.registry.entities),
        "constraints_count": len(sketch.constraints),
        "line1_p1": line1.p1_idx,
        "line1_p2": line1.p2_idx,
        "line2_p1": line2.p1_idx,
        "line2_p2": line2.p2_idx,
    }

    command = FilletCommand(
        sketch, corner_pid, line1_id, line2_id, DEFAULT_FILLET_RADIUS
    )
    command.execute()

    # Sanity check that something changed
    assert len(sketch.registry.points) != initial_state["points_count"]

    command.undo()

    # Verify state is restored
    assert len(sketch.registry.points) == initial_state["points_count"]
    assert len(sketch.registry.entities) == initial_state["entities_count"]
    assert len(sketch.constraints) == initial_state["constraints_count"]

    restored_line1 = sketch.registry.get_entity(line1_id)
    restored_line2 = sketch.registry.get_entity(line2_id)
    assert isinstance(restored_line1, Line)
    assert isinstance(restored_line2, Line)
    assert restored_line1.p1_idx == initial_state["line1_p1"]
    assert restored_line1.p2_idx == initial_state["line1_p2"]
    assert restored_line2.p1_idx == initial_state["line2_p1"]
    assert restored_line2.p2_idx == initial_state["line2_p2"]


def test_add_fillet_action_on_short_lines(
    element_with_corner, sketch_with_corner
):
    """Test that fillet action is aborted if lines are too short."""
    sketch, corner_pid, line1_id, _ = sketch_with_corner
    element = element_with_corner

    # Modify sketch to make one line very short
    line1 = sketch.registry.get_entity(line1_id)
    assert isinstance(line1, Line)
    p1 = sketch.registry.get_point(line1.p1_idx)
    p1.x = -1e-7  # Line length is now 1e-7, likely smaller than needed radius

    element.selection.select_junction(corner_pid, is_multi=False)
    element.add_fillet_action()

    # Execute command should not have been called because the action should
    # abort due to geometry validation
    element.execute_command.assert_not_called()
