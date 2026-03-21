import pytest

from rayforge.core.sketcher import Sketch
from rayforge.core.sketcher.commands import (
    AngleConstraintCommand,
    AngleConstraintParams,
)


@pytest.fixture
def sketch():
    return Sketch()


@pytest.fixture
def intersecting_lines(sketch):
    p1 = sketch.add_point(0, 0)
    p2 = sketch.add_point(100, 0)
    line1_id = sketch.add_line(p1, p2)

    p3 = sketch.add_point(50, -50)
    p4 = sketch.add_point(50, 50)
    line2_id = sketch.add_line(p3, p4)

    return sketch, line1_id, line2_id


def test_calculate_constraint_params_intersecting_lines(intersecting_lines):
    sketch, line1_id, line2_id = intersecting_lines
    result = AngleConstraintCommand.calculate_constraint_params(
        sketch.registry, line1_id, line2_id
    )

    assert result is not None
    assert isinstance(result, AngleConstraintParams)
    assert result.value_deg == pytest.approx(90.0, abs=0.1)


def test_calculate_constraint_params_45_degree_angle(sketch):
    p1 = sketch.add_point(0, 0)
    p2 = sketch.add_point(100, 0)
    line1_id = sketch.add_line(p1, p2)

    p3 = sketch.add_point(0, 0)
    p4 = sketch.add_point(100, 100)
    line2_id = sketch.add_line(p3, p4)

    result = AngleConstraintCommand.calculate_constraint_params(
        sketch.registry, line1_id, line2_id
    )

    assert result is not None
    assert result.value_deg == pytest.approx(45.0, abs=0.1)


def test_calculate_constraint_params_parallel_lines(sketch):
    p1 = sketch.add_point(0, 0)
    p2 = sketch.add_point(100, 0)
    line1_id = sketch.add_line(p1, p2)

    p3 = sketch.add_point(0, 50)
    p4 = sketch.add_point(100, 50)
    line2_id = sketch.add_line(p3, p4)

    result = AngleConstraintCommand.calculate_constraint_params(
        sketch.registry, line1_id, line2_id
    )

    assert result is None


def test_calculate_constraint_params_non_line_entity(sketch):
    p1 = sketch.add_point(0, 0)
    p2 = sketch.add_point(100, 0)
    line_id = sketch.add_line(p1, p2)

    center = sketch.add_point(50, 50)
    radius_p = sketch.add_point(60, 50)
    arc_id = sketch.add_arc(center, radius_p, 0, 90)

    result = AngleConstraintCommand.calculate_constraint_params(
        sketch.registry, line_id, arc_id
    )

    assert result is None


def test_calculate_constraint_params_two_non_lines(sketch):
    center1 = sketch.add_point(0, 0)
    r1 = sketch.add_point(10, 0)
    arc1_id = sketch.add_arc(center1, r1, 0, 90)

    center2 = sketch.add_point(50, 0)
    r2 = sketch.add_point(60, 0)
    arc2_id = sketch.add_arc(center2, r2, 0, 90)

    result = AngleConstraintCommand.calculate_constraint_params(
        sketch.registry, arc1_id, arc2_id
    )

    assert result is None


def test_calculate_constraint_params_returns_correct_ids(intersecting_lines):
    sketch, line1_id, line2_id = intersecting_lines
    result = AngleConstraintCommand.calculate_constraint_params(
        sketch.registry, line1_id, line2_id
    )

    assert result is not None
    assert result.anchor_id in (line1_id, line2_id)
    assert result.other_id in (line1_id, line2_id)
    assert result.anchor_id != result.other_id


def test_calculate_constraint_params_obtuse_angle(sketch):
    p1 = sketch.add_point(0, 0)
    p2 = sketch.add_point(100, 0)
    line1_id = sketch.add_line(p1, p2)

    p3 = sketch.add_point(0, 0)
    p4 = sketch.add_point(50, -100)
    line2_id = sketch.add_line(p3, p4)

    result = AngleConstraintCommand.calculate_constraint_params(
        sketch.registry, line1_id, line2_id
    )

    assert result is not None
    assert result.value_deg <= 180.0
    assert result.value_deg > 0.0
