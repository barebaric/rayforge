import pytest

from sketcher.core import Sketch
from sketcher.core.commands import (
    DistanceConstraintCommand,
    DistanceConstraintParams,
)


@pytest.fixture
def sketch():
    return Sketch()


def test_calculate_distance_two_points(sketch):
    p1_id = sketch.add_point(0, 0)
    p2_id = sketch.add_point(30, 40)

    result = DistanceConstraintCommand.calculate_distance(
        sketch.registry, [p1_id, p2_id], []
    )

    assert result is not None
    assert isinstance(result, DistanceConstraintParams)
    assert result.distance == pytest.approx(50.0)
    assert result.p1_id == p1_id
    assert result.p2_id == p2_id


def test_calculate_distance_single_line(sketch):
    p1_id = sketch.add_point(0, 0)
    p2_id = sketch.add_point(60, 80)
    line_id = sketch.add_line(p1_id, p2_id)

    result = DistanceConstraintCommand.calculate_distance(
        sketch.registry, [], [line_id]
    )

    assert result is not None
    assert result.distance == pytest.approx(100.0)
    assert result.p1_id == p1_id
    assert result.p2_id == p2_id


def test_calculate_distance_no_selection(sketch):
    result = DistanceConstraintCommand.calculate_distance(
        sketch.registry, [], []
    )

    assert result is None


def test_calculate_distance_one_point(sketch):
    p1_id = sketch.add_point(0, 0)

    result = DistanceConstraintCommand.calculate_distance(
        sketch.registry, [p1_id], []
    )

    assert result is None


def test_calculate_distance_three_points(sketch):
    p1_id = sketch.add_point(0, 0)
    p2_id = sketch.add_point(10, 0)
    p3_id = sketch.add_point(20, 0)

    result = DistanceConstraintCommand.calculate_distance(
        sketch.registry, [p1_id, p2_id, p3_id], []
    )

    assert result is None


def test_calculate_distance_multiple_entities(sketch):
    p1_id = sketch.add_point(0, 0)
    p2_id = sketch.add_point(10, 0)
    p3_id = sketch.add_point(20, 0)
    p4_id = sketch.add_point(30, 0)
    line1_id = sketch.add_line(p1_id, p2_id)
    line2_id = sketch.add_line(p3_id, p4_id)

    result = DistanceConstraintCommand.calculate_distance(
        sketch.registry, [], [line1_id, line2_id]
    )

    assert result is None


def test_calculate_distance_non_line_entity(sketch):
    center_id = sketch.add_point(0, 0)
    radius_id = sketch.add_point(10, 0)
    circle_id = sketch.add_circle(center_id, radius_id)

    result = DistanceConstraintCommand.calculate_distance(
        sketch.registry, [], [circle_id]
    )

    assert result is None


def test_calculate_distance_zero_distance(sketch):
    p1_id = sketch.add_point(5, 5)
    p2_id = sketch.add_point(5, 5)

    result = DistanceConstraintCommand.calculate_distance(
        sketch.registry, [p1_id, p2_id], []
    )

    assert result is not None
    assert result.distance == pytest.approx(0.0)


def test_calculate_distance_from_points(sketch):
    p1 = sketch.registry.get_point(sketch.add_point(0, 0))
    p2 = sketch.registry.get_point(sketch.add_point(3, 4))

    result = DistanceConstraintCommand.calculate_distance_from_points(p1, p2)

    assert result == pytest.approx(5.0)


def test_calculate_distance_from_points_negative_coords(sketch):
    p1 = sketch.registry.get_point(sketch.add_point(-10, -10))
    p2 = sketch.registry.get_point(sketch.add_point(-13, -14))

    result = DistanceConstraintCommand.calculate_distance_from_points(p1, p2)

    assert result == pytest.approx(5.0)


def test_calculate_distance_prefers_points_over_entity(sketch):
    p1_id = sketch.add_point(0, 0)
    p2_id = sketch.add_point(30, 40)
    p3_id = sketch.add_point(0, 0)
    p4_id = sketch.add_point(60, 80)
    line_id = sketch.add_line(p3_id, p4_id)

    result = DistanceConstraintCommand.calculate_distance(
        sketch.registry, [p1_id, p2_id], [line_id]
    )

    assert result is not None
    assert result.distance == pytest.approx(50.0)
    assert result.p1_id == p1_id
    assert result.p2_id == p2_id
