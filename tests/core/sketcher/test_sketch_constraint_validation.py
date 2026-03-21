import pytest

from rayforge.core.sketcher import Sketch
from rayforge.core.sketcher.constraints import CoincidentConstraint


@pytest.fixture
def sketch():
    return Sketch()


class TestSupportsConstraint:
    def test_distance_two_points(self, sketch):
        p1 = sketch.add_point(0, 0)
        p2 = sketch.add_point(10, 0)

        assert sketch.supports_constraint("dist", [p1, p2], []) is True

    def test_distance_single_line(self, sketch):
        p1 = sketch.add_point(0, 0)
        p2 = sketch.add_point(10, 0)
        line_id = sketch.add_line(p1, p2)

        assert sketch.supports_constraint("dist", [], [line_id]) is True

    def test_distance_multiple_lines(self, sketch):
        p1 = sketch.add_point(0, 0)
        p2 = sketch.add_point(10, 0)
        p3 = sketch.add_point(20, 0)
        p4 = sketch.add_point(30, 0)
        line1_id = sketch.add_line(p1, p2)
        line2_id = sketch.add_line(p3, p4)

        assert (
            sketch.supports_constraint("dist", [], [line1_id, line2_id])
            is False
        )

    def test_horizontal_two_points(self, sketch):
        p1 = sketch.add_point(0, 0)
        p2 = sketch.add_point(10, 0)

        assert sketch.supports_constraint("horiz", [p1, p2], []) is True

    def test_horizontal_single_line(self, sketch):
        p1 = sketch.add_point(0, 0)
        p2 = sketch.add_point(10, 10)
        line_id = sketch.add_line(p1, p2)

        assert sketch.supports_constraint("horiz", [], [line_id]) is True

    def test_vertical_two_points(self, sketch):
        p1 = sketch.add_point(0, 0)
        p2 = sketch.add_point(0, 10)

        assert sketch.supports_constraint("vert", [p1, p2], []) is True

    def test_vertical_single_line(self, sketch):
        p1 = sketch.add_point(0, 0)
        p2 = sketch.add_point(10, 10)
        line_id = sketch.add_line(p1, p2)

        assert sketch.supports_constraint("vert", [], [line_id]) is True

    def test_radius_arc(self, sketch):
        start = sketch.add_point(10, 0)
        end = sketch.add_point(0, 10)
        center = sketch.add_point(0, 0)
        arc_id = sketch.add_arc(start, end, center)

        assert sketch.supports_constraint("radius", [], [arc_id]) is True

    def test_radius_circle(self, sketch):
        center = sketch.add_point(0, 0)
        radius = sketch.add_point(10, 0)
        circle_id = sketch.add_circle(center, radius)

        assert sketch.supports_constraint("radius", [], [circle_id]) is True

    def test_diameter_circle(self, sketch):
        center = sketch.add_point(0, 0)
        radius = sketch.add_point(10, 0)
        circle_id = sketch.add_circle(center, radius)

        assert sketch.supports_constraint("diameter", [], [circle_id]) is True

    def test_diameter_arc(self, sketch):
        start = sketch.add_point(10, 0)
        end = sketch.add_point(0, 10)
        center = sketch.add_point(0, 0)
        arc_id = sketch.add_arc(start, end, center)

        assert sketch.supports_constraint("diameter", [], [arc_id]) is False

    def test_perpendicular_two_lines(self, sketch):
        p1 = sketch.add_point(0, 0)
        p2 = sketch.add_point(10, 0)
        p3 = sketch.add_point(0, 10)
        line1_id = sketch.add_line(p1, p2)
        line2_id = sketch.add_line(p1, p3)

        assert (
            sketch.supports_constraint("perp", [], [line1_id, line2_id])
            is True
        )

    def test_tangent_line_and_arc(self, sketch):
        p1 = sketch.add_point(0, 0)
        p2 = sketch.add_point(10, 0)
        line_id = sketch.add_line(p1, p2)

        start = sketch.add_point(50, 0)
        end = sketch.add_point(0, 50)
        center = sketch.add_point(0, 0)
        arc_id = sketch.add_arc(start, end, center)

        assert (
            sketch.supports_constraint("tangent", [], [line_id, arc_id])
            is True
        )

    def test_tangent_line_and_circle(self, sketch):
        p1 = sketch.add_point(0, 0)
        p2 = sketch.add_point(10, 0)
        line_id = sketch.add_line(p1, p2)

        center = sketch.add_point(50, 0)
        radius = sketch.add_point(60, 0)
        circle_id = sketch.add_circle(center, radius)

        assert (
            sketch.supports_constraint("tangent", [], [line_id, circle_id])
            is True
        )

    def test_equal_two_lines(self, sketch):
        p1 = sketch.add_point(0, 0)
        p2 = sketch.add_point(10, 0)
        p3 = sketch.add_point(20, 0)
        p4 = sketch.add_point(30, 0)
        line1_id = sketch.add_line(p1, p2)
        line2_id = sketch.add_line(p3, p4)

        assert (
            sketch.supports_constraint("equal", [], [line1_id, line2_id])
            is True
        )

    def test_equal_line_and_circle(self, sketch):
        p1 = sketch.add_point(0, 0)
        p2 = sketch.add_point(10, 0)
        line_id = sketch.add_line(p1, p2)

        center = sketch.add_point(50, 0)
        radius = sketch.add_point(60, 0)
        circle_id = sketch.add_circle(center, radius)

        assert (
            sketch.supports_constraint("equal", [], [line_id, circle_id])
            is True
        )

    def test_align_coincident(self, sketch):
        p1 = sketch.add_point(0, 0)
        p2 = sketch.add_point(10, 0)

        assert sketch.supports_constraint("align", [p1, p2], []) is True

    def test_point_on_line_valid(self, sketch):
        p1 = sketch.add_point(0, 0)
        p2 = sketch.add_point(10, 0)
        line_id = sketch.add_line(p1, p2)

        external_point = sketch.add_point(5, 5)

        assert (
            sketch.supports_constraint(
                "point_on_line", [external_point], [line_id]
            )
            is True
        )

    def test_point_on_line_endpoint_invalid(self, sketch):
        p1 = sketch.add_point(0, 0)
        p2 = sketch.add_point(10, 0)
        line_id = sketch.add_line(p1, p2)

        assert (
            sketch.supports_constraint("point_on_line", [p1], [line_id])
            is False
        )
        assert (
            sketch.supports_constraint("point_on_line", [p2], [line_id])
            is False
        )

    def test_symmetry_three_points(self, sketch):
        p1 = sketch.add_point(0, 0)
        p2 = sketch.add_point(10, 0)
        center = sketch.add_point(5, 0)

        assert (
            sketch.supports_constraint("symmetry", [p1, p2, center], [])
            is True
        )

    def test_symmetry_two_points_and_line(self, sketch):
        p1 = sketch.add_point(0, 10)
        p2 = sketch.add_point(0, -10)

        axis_p1 = sketch.add_point(-5, 0)
        axis_p2 = sketch.add_point(5, 0)
        axis_line_id = sketch.add_line(axis_p1, axis_p2)

        assert (
            sketch.supports_constraint("symmetry", [p1, p2], [axis_line_id])
            is True
        )

    def test_angle_two_lines(self, sketch):
        p1 = sketch.add_point(0, 0)
        p2 = sketch.add_point(10, 0)
        p3 = sketch.add_point(0, 10)
        line1_id = sketch.add_line(p1, p2)
        line2_id = sketch.add_line(p1, p3)

        assert (
            sketch.supports_constraint("angle", [], [line1_id, line2_id])
            is True
        )

    def test_angle_invalid_one_line(self, sketch):
        p1 = sketch.add_point(0, 0)
        p2 = sketch.add_point(10, 0)
        line_id = sketch.add_line(p1, p2)

        assert sketch.supports_constraint("angle", [], [line_id]) is False

    def test_aspect_ratio_two_lines(self, sketch):
        p1 = sketch.add_point(0, 0)
        p2 = sketch.add_point(10, 0)
        p3 = sketch.add_point(0, 10)
        p4 = sketch.add_point(0, 20)
        line1_id = sketch.add_line(p1, p2)
        line2_id = sketch.add_line(p3, p4)

        assert (
            sketch.supports_constraint(
                "aspect_ratio", [], [line1_id, line2_id]
            )
            is True
        )


class TestGetCoincidentPoints:
    def test_no_coincident_constraints(self, sketch):
        p1 = sketch.add_point(0, 0)
        sketch.add_point(10, 0)

        result = sketch.get_coincident_points(p1)

        assert result == {p1}

    def test_single_coincident_pair(self, sketch):
        p1 = sketch.add_point(0, 0)
        p2 = sketch.add_point(10, 0)

        sketch.constraints.append(CoincidentConstraint(p1, p2))

        result = sketch.get_coincident_points(p1)

        assert result == {p1, p2}

    def test_transitive_coincident(self, sketch):
        p1 = sketch.add_point(0, 0)
        p2 = sketch.add_point(10, 0)
        p3 = sketch.add_point(20, 0)

        sketch.constraints.append(CoincidentConstraint(p1, p2))
        sketch.constraints.append(CoincidentConstraint(p2, p3))

        result = sketch.get_coincident_points(p1)

        assert result == {p1, p2, p3}

    def test_chain_from_middle(self, sketch):
        p1 = sketch.add_point(0, 0)
        p2 = sketch.add_point(10, 0)
        p3 = sketch.add_point(20, 0)

        sketch.constraints.append(CoincidentConstraint(p1, p2))
        sketch.constraints.append(CoincidentConstraint(p2, p3))

        result = sketch.get_coincident_points(p2)

        assert result == {p1, p2, p3}

    def test_separate_groups(self, sketch):
        p1 = sketch.add_point(0, 0)
        p2 = sketch.add_point(10, 0)
        p3 = sketch.add_point(20, 0)
        p4 = sketch.add_point(30, 0)

        sketch.constraints.append(CoincidentConstraint(p1, p2))
        sketch.constraints.append(CoincidentConstraint(p3, p4))

        result1 = sketch.get_coincident_points(p1)
        result3 = sketch.get_coincident_points(p3)

        assert result1 == {p1, p2}
        assert result3 == {p3, p4}
