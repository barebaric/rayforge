"""
Tests for constraint conflict detection in the sketcher.
"""

from sketcher.core.arrays import CurveAlongArrayStrategy
from sketcher.core.commands import CreateArrayCommand
from sketcher.core.constraints import (
    ConstraintStatus,
    DistanceConstraint,
    HorizontalConstraint,
)
from sketcher.core.params import ParameterContext
from sketcher.core.registry import EntityRegistry
from sketcher.core.sketch import Sketch
from sketcher.core.solver import CONFLICT_ERROR_THRESHOLD, Solver


class TestSolverConflictDetection:
    """Tests for the Solver.get_conflicting_constraints method."""

    def test_no_conflicts_satisfied_constraints(self):
        """When constraints are satisfied, no conflicts should be reported."""
        reg = EntityRegistry()
        params = ParameterContext()

        p1 = reg.add_point(0, 0, fixed=True)
        p2 = reg.add_point(10, 0, fixed=False)

        constraints = [
            HorizontalConstraint(p1, p2),
            DistanceConstraint(p1, p2, 10.0),
        ]

        solver = Solver(reg, params, constraints)
        solver.solve()

        conflicting = solver.get_conflicting_constraints()
        assert conflicting == set()

    def test_conflicting_distance_constraints(self):
        """
        Two distance constraints on the same points with incompatible values
        should both be detected as conflicting.
        """
        reg = EntityRegistry()
        params = ParameterContext()

        p1 = reg.add_point(0, 0, fixed=True)
        p2 = reg.add_point(15, 0, fixed=False)

        constraints = [
            DistanceConstraint(p1, p2, 10.0),
            DistanceConstraint(p1, p2, 20.0),
        ]

        solver = Solver(reg, params, constraints)
        solver.solve()

        conflicting = solver.get_conflicting_constraints()
        assert 0 in conflicting
        assert 1 in conflicting

    def test_partial_conflict_some_satisfied(self):
        """
        When some constraints can be satisfied but others cannot,
        only the unsatisfied ones should be marked as conflicting.
        """
        reg = EntityRegistry()
        params = ParameterContext()

        p1 = reg.add_point(0, 0, fixed=True)
        p2 = reg.add_point(10, 0, fixed=True)
        p3 = reg.add_point(5, 0, fixed=False)

        constraints = [
            HorizontalConstraint(p1, p2),
            DistanceConstraint(p1, p3, 10.0),
            DistanceConstraint(p2, p3, 10.0),
        ]

        solver = Solver(reg, params, constraints)
        solver.solve()

        conflicting = solver.get_conflicting_constraints()
        assert 0 not in conflicting
        assert 1 in conflicting
        assert 2 in conflicting

    def test_fixed_points_impossible_constraint(self):
        """
        An impossible constraint between two fixed points should be detected.
        """
        reg = EntityRegistry()
        params = ParameterContext()

        p1 = reg.add_point(0, 0, fixed=True)
        p2 = reg.add_point(10, 0, fixed=True)

        constraints = [DistanceConstraint(p1, p2, 5.0)]

        solver = Solver(reg, params, constraints)
        solver.solve()

        conflicting = solver.get_conflicting_constraints()
        assert 0 in conflicting

    def test_conflict_threshold(self):
        """
        Test that the threshold parameter works correctly.
        Constraints with error just below threshold should not be reported.
        """
        reg = EntityRegistry()
        params = ParameterContext()

        p1 = reg.add_point(0, 0, fixed=True)
        p2 = reg.add_point(10.0001, 0, fixed=True)

        constraints = [DistanceConstraint(p1, p2, 10.0)]

        solver = Solver(reg, params, constraints)
        solver.solve()

        conflicting = solver.get_conflicting_constraints(
            threshold=CONFLICT_ERROR_THRESHOLD
        )
        assert 0 not in conflicting

        conflicting_strict = solver.get_conflicting_constraints(threshold=1e-6)
        assert 0 in conflicting_strict


class TestSketchConflictTracking:
    """Tests for the Sketch class conflict tracking functionality."""

    def test_sketch_updates_constraint_status(self):
        """
        After solving, conflicting constraints should have CONFLICTING status.
        """
        sketch = Sketch()
        p1 = sketch.add_point(0, 0, fixed=True)
        p2 = sketch.add_point(10, 0)

        sketch.constrain_distance(p1, p2, 5.0)
        sketch.constrain_distance(p1, p2, 15.0)

        sketch.solve()

        assert sketch.constraints[0].status == ConstraintStatus.CONFLICTING
        assert sketch.constraints[1].status == ConstraintStatus.CONFLICTING

    def test_sketch_has_conflicts_property(self):
        """The has_conflicts property should return True when conflicts."""
        sketch = Sketch()
        p1 = sketch.add_point(0, 0, fixed=True)
        p2 = sketch.add_point(10, 0)

        sketch.constrain_distance(p1, p2, 5.0)
        sketch.constrain_distance(p1, p2, 15.0)

        sketch.solve()

        assert sketch.has_conflicts is True

    def test_sketch_no_conflicts_property(self):
        """The has_conflicts property should return False when no conflicts."""
        sketch = Sketch()
        p1 = sketch.add_point(0, 0, fixed=True)
        p2 = sketch.add_point(10, 0)

        sketch.constrain_horizontal(p1, p2)
        sketch.constrain_distance(p1, p2, 10.0)

        sketch.solve()

        assert sketch.has_conflicts is False

    def test_sketch_conflicting_constraints_property(self):
        """
        The conflicting_constraints property should return only conflicting
        constraints.
        """
        sketch = Sketch()
        p1 = sketch.add_point(0, 0, fixed=True)
        p2 = sketch.add_point(10, 0)

        sketch.constrain_horizontal(p1, p2)
        sketch.constrain_distance(p1, p2, 5.0)
        sketch.constrain_distance(p1, p2, 15.0)

        sketch.solve()

        conflicting = sketch.conflicting_constraints
        assert len(conflicting) == 2
        assert all(
            c.status == ConstraintStatus.CONFLICTING for c in conflicting
        )

    def test_conflict_cleared_when_resolved(self):
        """
        When a conflict is resolved (e.g., constraint removed or value
        changed), the CONFLICTING status should be cleared.
        """
        sketch = Sketch()
        p1 = sketch.add_point(0, 0, fixed=True)
        p2 = sketch.add_point(10, 0)

        c1 = sketch.constrain_distance(p1, p2, 5.0)
        c2 = sketch.constrain_distance(p1, p2, 15.0)

        sketch.solve()
        assert sketch.has_conflicts is True

        sketch.constraints.remove(c1)
        sketch.solve()

        assert sketch.has_conflicts is False
        assert c2.status == ConstraintStatus.VALID

    def test_error_status_preserved(self):
        """
        Constraints with ERROR status should not be changed to CONFLICTING.
        """
        sketch = Sketch()
        p1 = sketch.add_point(0, 0, fixed=True)
        p2 = sketch.add_point(10, 0)

        c = sketch.constrain_distance(p1, p2, "invalid_var_name")
        c.status = ConstraintStatus.ERROR

        sketch.solve()

        assert c.status == ConstraintStatus.ERROR


class TestConflictingWithEntities:
    """Test conflict detection with actual sketch entities."""

    def test_line_with_conflicting_length_constraints(self):
        """
        Test conflict detection on a line with conflicting length constraints.
        """
        sketch = Sketch()
        p1 = sketch.add_point(0, 0, fixed=True)
        p2 = sketch.add_point(10, 0)
        sketch.add_line(p1, p2)

        sketch.constrain_distance(p1, p2, 10.0)
        sketch.constrain_distance(p1, p2, 20.0)

        sketch.solve()

        assert sketch.has_conflicts is True

    def test_multiple_lines_partial_conflict(self):
        """
        Test a scenario where some constraints conflict but others don't.
        """
        sketch = Sketch()

        p1 = sketch.add_point(0, 0, fixed=True)
        p2 = sketch.add_point(10, 0)
        p3 = sketch.add_point(20, 0)

        sketch.add_line(p1, p2)
        sketch.add_line(p2, p3)

        sketch.constrain_distance(p1, p2, 10.0)
        sketch.constrain_distance(p2, p3, 10.0)

        sketch.solve()

        assert sketch.has_conflicts is False

        sketch.constrain_distance(p1, p2, 50.0)
        sketch.solve()

        assert sketch.has_conflicts is True


class TestConflictingWithArrayTemplate:
    """
    An array template's placement is owned by the array: sync_arrays
    re-anchors the template onto the guide after every solve. A
    template constraint the solver had satisfied can therefore be
    violated by the sync, and the conflict status must reflect the
    final (post-sync) geometry.
    """

    def _build_array_sketch(self):
        """Horizontal guide line plus a horizontal line template,
        turned into a tangent-aligned array through the create flow."""
        sketch = Sketch()
        g0 = sketch.registry.add_point(0.0, 0.0)
        g1 = sketch.registry.add_point(40.0, 0.0)
        guide = sketch.registry.add_line(g0, g1)
        t0 = sketch.registry.add_point(-6.0, -2.0)
        t1 = sketch.registry.add_point(6.0, -2.0)
        template = sketch.registry.add_line(t0, t1)
        strategy = CurveAlongArrayStrategy(
            count=3, path_entity_id=guide, align_to_tangent=True
        )
        CreateArrayCommand(sketch, strategy, [template]).execute()
        sketch.solve()
        return sketch, g1

    def _template_point_ids(self, sketch):
        registry = sketch.registry
        _slot, eids = sketch.arrays[0].living_members(registry)[0]
        line = registry.get_entity(eids[0])
        return line.p1_idx, line.p2_idx

    def test_orientation_constraint_conflicts_after_guide_edit(self):
        """Rotating the guide re-anchors the template, breaking a
        horizontal constraint that the solver itself had satisfied.
        The constraint must be flagged as conflicting instead of
        silently staying valid while solver and sync fight."""
        sketch, g1 = self._build_array_sketch()
        p1, p2 = self._template_point_ids(sketch)
        sketch.constrain_horizontal(p1, p2)
        horizontal = sketch.constraints[-1]

        sketch.solve()
        assert horizontal.status == ConstraintStatus.VALID

        sketch.registry.get_point(g1).y = 30.0
        sketch.solve()

        assert horizontal.status == ConstraintStatus.CONFLICTING
        assert sketch.has_conflicts is True

    def test_conflict_clears_when_guide_tangent_matches_again(self):
        """Rotating the guide back restores the tangent the template
        was anchored at, so the constraint becomes satisfiable again
        and the conflict status must be cleared. The re-anchor applies
        its stored delta once more after the solver re-satisfies the
        constraint, so the stable state is reached on the next solve
        cycle."""
        sketch, g1 = self._build_array_sketch()
        p1, p2 = self._template_point_ids(sketch)
        sketch.constrain_horizontal(p1, p2)
        horizontal = sketch.constraints[-1]

        sketch.registry.get_point(g1).y = 30.0
        sketch.solve()
        assert horizontal.status == ConstraintStatus.CONFLICTING

        sketch.registry.get_point(g1).y = 0.0
        sketch.solve()
        sketch.solve()

        assert horizontal.status == ConstraintStatus.VALID
        assert sketch.has_conflicts is False

    def test_shape_constraint_survives_reanchor(self):
        """Shape constraints (e.g. a distance) are placement-invariant:
        the rigid re-anchor keeps them satisfied and they must not be
        flagged."""
        sketch, g1 = self._build_array_sketch()
        p1, p2 = self._template_point_ids(sketch)
        distance = sketch.constrain_distance(p1, p2, 12.0)

        sketch.registry.get_point(g1).y = 30.0
        sketch.solve()

        assert distance.status == ConstraintStatus.VALID
        assert sketch.has_conflicts is False
