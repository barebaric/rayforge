import logging
import time
from collections.abc import Sequence

from raygeo.sketch import PointDef
from raygeo.sketch import Solver as GeoSolver

from .constraints import Constraint
from .params import ParameterContext
from .registry import EntityRegistry
from .types import EntityID

CONFLICT_ERROR_THRESHOLD = 1e-3

logger = logging.getLogger(__name__)


class Solver:
    def __init__(
        self,
        registry: EntityRegistry,
        params: ParameterContext,
        constraints: Sequence[Constraint],
        auxiliary_constraints: Sequence[Constraint] = (),
        point_filter: set[EntityID] | None = None,
    ):
        self.registry = registry
        self.params = params
        self.constraints = list(constraints)
        self.auxiliary_constraints = list(auxiliary_constraints)
        self.point_filter = point_filter

    def _build_problem(
        self,
    ) -> tuple[GeoSolver, list[int]]:
        """
        Flattens the registry and constraints into a raygeo solver
        problem.  Constraints without a solver counterpart are
        skipped; the returned index list maps raygeo constraint
        positions back to self.constraints positions.
        """
        points = [
            PointDef(p.id, p.x, p.y, p.fixed) for p in self.registry.points
        ]
        hard = []
        kept_indices = []
        for index, constraint in enumerate(self.constraints):
            converted = constraint.to_solver_constraint(self.registry)
            if converted is not None:
                hard.append(converted)
                kept_indices.append(index)
        auxiliary = []
        for constraint in self.auxiliary_constraints:
            converted = constraint.to_solver_constraint(self.registry)
            if converted is not None:
                auxiliary.append(converted)
        point_filter = (
            sorted(self.point_filter)
            if self.point_filter is not None
            else None
        )
        return (
            GeoSolver(points, hard, auxiliary, point_filter),
            kept_indices,
        )

    def solve(self, tolerance: float = 1e-5, update_dof: bool = True) -> bool:
        """
        Runs the raygeo least-squares solver to satisfy constraints.
        Returns True if successful.

        A global solve (no point_filter) partitions the constraints
        into independent connected components and solves each one
        separately, so the work and the DOF analysis scale with the
        constraint islands instead of the whole sketch.  Success is
        the conjunction of the component results, and the DOF
        analysis runs per component on its own Jacobian.

        If update_dof is False, the constrained status of points and
        entities is left untouched, which is useful for interactive
        updates.

        If point_filter is set, only the listed points are optimized
        and the DOF analysis is restricted to them; all other points
        keep their positions.
        """
        solve_started = time.perf_counter()

        if update_dof:
            for p in self.registry.points:
                p.constrained = p.fixed

        geo_solver, _ = self._build_problem()
        success = geo_solver.solve(tolerance, update_dof)
        self._last_status = geo_solver.status()
        self._last_nfev = geo_solver.nfev()

        positions = geo_solver.positions()
        for point, (x, y) in zip(self.registry.points, positions):
            point.x = x
            point.y = y
        if update_dof:
            constrained_flags = geo_solver.constrained()
            for point, constrained in zip(
                self.registry.points, constrained_flags
            ):
                point.constrained = constrained

        if update_dof and success and self.point_filter is None:
            self._update_entity_constraints()

        logger.debug(
            "Solver.solve: points=%d hard=%d aux=%d nfev=%d status=%d "
            "ms=%.1f success=%s",
            len(self.registry.points),
            len(self.constraints),
            len(self.auxiliary_constraints),
            self._last_nfev,
            self._last_status,
            (time.perf_counter() - solve_started) * 1000.0,
            success,
        )

        return success

    def _update_entity_constraints(self):
        """
        Updates the constrained status of Entities based on their points.
        An entity is constrained only if all its defining points are
        constrained.
        """
        for entity in self.registry.entities:
            entity.update_constrained_status(self.registry, self.constraints)

    def get_conflicting_constraints(
        self, threshold: float = CONFLICT_ERROR_THRESHOLD
    ) -> set[int]:
        """
        Identifies constraints that have significant residual error,
        indicating they conflict with other constraints or cannot be satisfied.

        Returns:
            Set of constraint indices in self.constraints that are conflicting.
        """
        geo_solver, kept_indices = self._build_problem()
        conflicting = geo_solver.conflicting(threshold)
        return {kept_indices[index] for index in conflicting}
