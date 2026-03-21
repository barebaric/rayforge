import pytest

from rayforge.core.sketcher import Sketch
from rayforge.core.sketcher.commands import (
    EqualConstraintCommand,
    EqualConstraintMergeResult,
)
from rayforge.core.sketcher.constraints import EqualLengthConstraint


@pytest.fixture
def sketch():
    return Sketch()


@pytest.fixture
def lines_without_constraints(sketch):
    p1 = sketch.add_point(0, 0)
    p2 = sketch.add_point(10, 0)
    p3 = sketch.add_point(20, 0)
    p4 = sketch.add_point(30, 0)

    line1_id = sketch.add_line(p1, p2)
    line2_id = sketch.add_line(p3, p4)

    return sketch, line1_id, line2_id


def test_find_and_merge_no_existing_constraints(lines_without_constraints):
    sketch, line1_id, line2_id = lines_without_constraints

    result = EqualConstraintCommand.find_and_merge_constraints(
        sketch, [line1_id, line2_id]
    )

    assert result is not None
    assert isinstance(result, EqualConstraintMergeResult)
    assert set(result.final_entity_ids) == {line1_id, line2_id}
    assert result.constraints_to_remove == []


def test_find_and_merge_with_existing_constraint(sketch):
    p1 = sketch.add_point(0, 0)
    p2 = sketch.add_point(10, 0)
    p3 = sketch.add_point(20, 0)
    p4 = sketch.add_point(30, 0)
    p5 = sketch.add_point(40, 0)
    p6 = sketch.add_point(50, 0)

    line1_id = sketch.add_line(p1, p2)
    line2_id = sketch.add_line(p3, p4)
    line3_id = sketch.add_line(p5, p6)

    existing = EqualLengthConstraint([line1_id, line2_id])
    sketch.constraints.append(existing)

    result = EqualConstraintCommand.find_and_merge_constraints(
        sketch, [line2_id, line3_id]
    )

    assert result is not None
    assert set(result.final_entity_ids) == {line1_id, line2_id, line3_id}
    assert existing in result.constraints_to_remove


def test_find_and_merge_all_new_entities(lines_without_constraints):
    sketch, line1_id, line2_id = lines_without_constraints

    result = EqualConstraintCommand.find_and_merge_constraints(
        sketch, [line1_id, line2_id]
    )

    assert result is not None
    assert len(result.final_entity_ids) == 2
    assert len(result.constraints_to_remove) == 0


def test_find_and_merge_single_entity(sketch):
    p1 = sketch.add_point(0, 0)
    p2 = sketch.add_point(10, 0)
    line_id = sketch.add_line(p1, p2)

    result = EqualConstraintCommand.find_and_merge_constraints(
        sketch, [line_id]
    )

    assert result is not None
    assert result.final_entity_ids == [line_id]
    assert result.constraints_to_remove == []


def test_find_and_merge_multiple_constraints(sketch):
    p1 = sketch.add_point(0, 0)
    p2 = sketch.add_point(10, 0)
    p3 = sketch.add_point(20, 0)
    p4 = sketch.add_point(30, 0)
    p5 = sketch.add_point(40, 0)
    p6 = sketch.add_point(50, 0)
    p7 = sketch.add_point(60, 0)
    p8 = sketch.add_point(70, 0)

    line1_id = sketch.add_line(p1, p2)
    line2_id = sketch.add_line(p3, p4)
    line3_id = sketch.add_line(p5, p6)
    line4_id = sketch.add_line(p7, p8)

    constr1 = EqualLengthConstraint([line1_id, line2_id])
    constr2 = EqualLengthConstraint([line3_id, line4_id])
    sketch.constraints.append(constr1)
    sketch.constraints.append(constr2)

    result = EqualConstraintCommand.find_and_merge_constraints(
        sketch, [line2_id, line3_id]
    )

    assert result is not None
    assert set(result.final_entity_ids) == {
        line1_id,
        line2_id,
        line3_id,
        line4_id,
    }
    assert constr1 in result.constraints_to_remove
    assert constr2 in result.constraints_to_remove


def test_find_and_merge_empty_selection(sketch):
    result = EqualConstraintCommand.find_and_merge_constraints(sketch, [])

    assert result is not None
    assert result.final_entity_ids == []
    assert result.constraints_to_remove == []


def test_find_and_merge_no_overlap_with_existing(sketch):
    p1 = sketch.add_point(0, 0)
    p2 = sketch.add_point(10, 0)
    p3 = sketch.add_point(20, 0)
    p4 = sketch.add_point(30, 0)
    p5 = sketch.add_point(40, 0)
    p6 = sketch.add_point(50, 0)

    line1_id = sketch.add_line(p1, p2)
    line2_id = sketch.add_line(p3, p4)
    line3_id = sketch.add_line(p5, p6)

    existing = EqualLengthConstraint([line1_id, line2_id])
    sketch.constraints.append(existing)

    result = EqualConstraintCommand.find_and_merge_constraints(
        sketch, [line3_id]
    )

    assert result is not None
    assert result.final_entity_ids == [line3_id]
    assert result.constraints_to_remove == []
