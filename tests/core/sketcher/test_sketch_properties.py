import pytest

from rayforge.core.sketcher import Sketch
from rayforge.core.sketcher.constraints import ConstraintStatus


@pytest.fixture
def sketch():
    return Sketch()


class TestIsFullyConstrained:
    def test_empty_sketch(self, sketch):
        origin = sketch.registry.get_point(sketch.origin_id)
        origin.constrained = True
        assert sketch.is_fully_constrained is True

    def test_unconstrained_line(self, sketch):
        p1 = sketch.add_point(0, 0)
        p2 = sketch.add_point(10, 0)
        sketch.add_line(p1, p2)

        assert sketch.is_fully_constrained is False

    def test_fixed_points(self, sketch):
        p1 = sketch.add_point(0, 0, fixed=True)
        p2 = sketch.add_point(10, 0, fixed=True)
        sketch.add_line(p1, p2)

        for pt in sketch.registry.points:
            pt.constrained = True
        for ent in sketch.registry.entities:
            ent.constrained = True

        assert sketch.is_fully_constrained is True

    def test_partially_constrained(self, sketch):
        p1 = sketch.add_point(0, 0, fixed=True)
        p2 = sketch.add_point(10, 0)
        sketch.add_line(p1, p2)

        origin = sketch.registry.get_point(sketch.origin_id)
        origin.constrained = True

        pt1 = sketch.registry.get_point(p1)
        pt1.constrained = True

        for ent in sketch.registry.entities:
            ent.constrained = True

        assert sketch.is_fully_constrained is False

    def test_circle_with_unconstrained_radius_point(self, sketch):
        center = sketch.add_point(50, 50, fixed=True)
        radius = sketch.add_point(60, 50)
        circle_id = sketch.add_circle(center, radius)

        center_pt = sketch.registry.get_point(center)
        center_pt.constrained = True

        circle = sketch.registry.get_entity(circle_id)
        circle.constrained = True

        origin = sketch.registry.get_point(sketch.origin_id)
        origin.constrained = True

        assert sketch.is_fully_constrained is True

    def test_circle_with_shared_radius_point(self, sketch):
        center = sketch.add_point(50, 50, fixed=True)
        radius = sketch.add_point(60, 50)
        circle_id = sketch.add_circle(center, radius)

        p1 = sketch.add_point(60, 50)
        sketch.add_line(radius, p1)

        center_pt = sketch.registry.get_point(center)
        center_pt.constrained = True
        radius_pt = sketch.registry.get_point(radius)
        radius_pt.constrained = False

        circle = sketch.registry.get_entity(circle_id)
        circle.constrained = True

        origin = sketch.registry.get_point(sketch.origin_id)
        origin.constrained = True

        assert sketch.is_fully_constrained is False


class TestConflictingConstraints:
    def test_no_conflicts(self, sketch):
        assert sketch.conflicting_constraints == []
        assert sketch.has_conflicts is False

    def test_with_conflicting_constraint(self, sketch):
        p1 = sketch.add_point(0, 0)
        p2 = sketch.add_point(10, 0)
        sketch.add_line(p1, p2)

        from rayforge.core.sketcher.constraints import DistanceConstraint

        constr = DistanceConstraint(p1, p2, 10.0)
        constr.status = ConstraintStatus.CONFLICTING
        sketch.constraints.append(constr)

        assert len(sketch.conflicting_constraints) == 1
        assert sketch.has_conflicts is True

    def test_mixed_constraints(self, sketch):
        p1 = sketch.add_point(0, 0)
        p2 = sketch.add_point(10, 0)
        sketch.add_line(p1, p2)

        from rayforge.core.sketcher.constraints import DistanceConstraint

        constr1 = DistanceConstraint(p1, p2, 10.0)
        constr1.status = ConstraintStatus.VALID
        sketch.constraints.append(constr1)

        constr2 = DistanceConstraint(p1, p2, 20.0)
        constr2.status = ConstraintStatus.CONFLICTING
        sketch.constraints.append(constr2)

        assert len(sketch.conflicting_constraints) == 1
        assert sketch.conflicting_constraints[0] == constr2
        assert sketch.has_conflicts is True

    def test_multiple_conflicts(self, sketch):
        p1 = sketch.add_point(0, 0)
        p2 = sketch.add_point(10, 0)
        p3 = sketch.add_point(20, 0)
        sketch.add_line(p1, p2)
        sketch.add_line(p2, p3)

        from rayforge.core.sketcher.constraints import DistanceConstraint

        constr1 = DistanceConstraint(p1, p2, 10.0)
        constr1.status = ConstraintStatus.CONFLICTING
        sketch.constraints.append(constr1)

        constr2 = DistanceConstraint(p2, p3, 10.0)
        constr2.status = ConstraintStatus.CONFLICTING
        sketch.constraints.append(constr2)

        assert len(sketch.conflicting_constraints) == 2
        assert sketch.has_conflicts is True
