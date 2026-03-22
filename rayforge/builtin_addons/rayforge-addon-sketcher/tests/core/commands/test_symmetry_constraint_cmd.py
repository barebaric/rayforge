import pytest

from sketcher.core import Sketch
from sketcher.core.commands import (
    SymmetryConstraintCommand,
    SymmetryConstraintParams,
)


@pytest.fixture
def sketch():
    return Sketch()


def test_determine_params_three_points(sketch):
    p1_id = sketch.add_point(0, 0)
    p2_id = sketch.add_point(10, 0)
    center_id = sketch.add_point(5, 0)

    result = SymmetryConstraintCommand.determine_constraint_params(
        [p1_id, p2_id, center_id], []
    )

    assert result is not None
    assert isinstance(result, SymmetryConstraintParams)
    assert result.p1_id == p1_id
    assert result.p2_id == p2_id
    assert result.center_id == center_id
    assert result.axis_id is None


def test_determine_params_two_points_one_line(sketch):
    p1_id = sketch.add_point(0, 10)
    p2_id = sketch.add_point(0, -10)

    axis_p1 = sketch.add_point(-5, 0)
    axis_p2 = sketch.add_point(5, 0)
    axis_line_id = sketch.add_line(axis_p1, axis_p2)

    result = SymmetryConstraintCommand.determine_constraint_params(
        [p1_id, p2_id], [axis_line_id]
    )

    assert result is not None
    assert isinstance(result, SymmetryConstraintParams)
    assert result.p1_id == p1_id
    assert result.p2_id == p2_id
    assert result.center_id is None
    assert result.axis_id == axis_line_id


def test_determine_params_invalid_two_points_no_entity(sketch):
    p1_id = sketch.add_point(0, 0)
    p2_id = sketch.add_point(10, 0)

    result = SymmetryConstraintCommand.determine_constraint_params(
        [p1_id, p2_id], []
    )

    assert result is None


def test_determine_params_invalid_one_point(sketch):
    p1_id = sketch.add_point(0, 0)

    result = SymmetryConstraintCommand.determine_constraint_params([p1_id], [])

    assert result is None


def test_determine_params_invalid_four_points(sketch):
    p1_id = sketch.add_point(0, 0)
    p2_id = sketch.add_point(10, 0)
    p3_id = sketch.add_point(20, 0)
    p4_id = sketch.add_point(30, 0)

    result = SymmetryConstraintCommand.determine_constraint_params(
        [p1_id, p2_id, p3_id, p4_id], []
    )

    assert result is None


def test_determine_params_invalid_two_points_two_entities(sketch):
    p1_id = sketch.add_point(0, 0)
    p2_id = sketch.add_point(10, 0)

    line1_p1 = sketch.add_point(0, 5)
    line1_p2 = sketch.add_point(10, 5)
    line1_id = sketch.add_line(line1_p1, line1_p2)

    line2_p1 = sketch.add_point(0, -5)
    line2_p2 = sketch.add_point(10, -5)
    line2_id = sketch.add_line(line2_p1, line2_p2)

    result = SymmetryConstraintCommand.determine_constraint_params(
        [p1_id, p2_id], [line1_id, line2_id]
    )

    assert result is None


def test_determine_params_empty_selection():
    result = SymmetryConstraintCommand.determine_constraint_params([], [])

    assert result is None


def test_determine_params_three_points_with_entity_ignored(sketch):
    p1_id = sketch.add_point(0, 0)
    p2_id = sketch.add_point(10, 0)
    center_id = sketch.add_point(5, 0)

    axis_p1 = sketch.add_point(0, 5)
    axis_p2 = sketch.add_point(10, 5)
    axis_line_id = sketch.add_line(axis_p1, axis_p2)

    result = SymmetryConstraintCommand.determine_constraint_params(
        [p1_id, p2_id, center_id], [axis_line_id]
    )

    assert result is None
