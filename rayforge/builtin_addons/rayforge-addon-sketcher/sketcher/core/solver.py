import logging
import time
from collections.abc import Sequence

from raygeo.sketch import PointDef
from raygeo.sketch import Solver as GeoSolver

from .types import EntityID

CONFLICT_ERROR_THRESHOLD = 1e-3

logger = logging.getLogger(__name__)


class Solver:
    """Drives the raygeo least-squares solver over the sketch's
    constraint objects."""

    def __init__(
        self,
        registry,
        constraints: Sequence,
        auxiliary_constraints: Sequence = (),
        point_filter: set[EntityID] | None = None,
    ):
        self.registry = registry
        self.constraints = list(constraints)
        self.auxiliary_constraints = list(auxiliary_constraints)
        self.point_filter = point_filter

    def _build_problem(self) -> GeoSolver:
        """Flattens the registry into a raygeo solver problem."""
        points = [
            PointDef(point.id, point.x, point.y, point.fixed)
            for point in self.registry.points()
        ]
        point_filter = (
            sorted(self.point_filter)
            if self.point_filter is not None
            else None
        )
        return GeoSolver(
            points,
            list(self.constraints),
            list(self.auxiliary_constraints),
            point_filter,
        )

    def solve(self, tolerance: float = 1e-5, update_dof: bool = True) -> bool:
        """
        Runs the raygeo least-squares solver to satisfy constraints.
        Returns True if successful.

        If update_dof is False, the constrained status of points and
        entities is left untouched, which is useful for interactive
        updates.

        If point_filter is set, only the listed points are optimized
        and the DOF analysis is restricted to them; all other points
        keep their positions.
        """
        solve_started = time.perf_counter()

        if update_dof:
            for point in self.registry.points():
                self.registry.set_point_constrained(point.id, point.fixed)

        geo_solver = self._build_problem()
        success = geo_solver.solve(tolerance, update_dof)
        self._last_status = geo_solver.status()
        self._last_nfev = geo_solver.nfev()

        positions = geo_solver.positions()
        for point, (x, y) in zip(self.registry.points(), positions):
            self.registry.set_point_position(point.id, x, y)
        if update_dof:
            constrained_flags = geo_solver.constrained()
            for point, constrained in zip(
                self.registry.points(), constrained_flags
            ):
                self.registry.set_point_constrained(point.id, constrained)

        if update_dof and success and self.point_filter is None:
            self._update_entity_constraints()

        logger.debug(
            "Solver.solve: points=%d hard=%d aux=%d nfev=%d status=%d "
            "ms=%.1f success=%s",
            len(self.registry.points()),
            len(self.constraints),
            len(self.auxiliary_constraints),
            self._last_nfev,
            self._last_status,
            (time.perf_counter() - solve_started) * 1000.0,
            success,
        )

        return success

    def _radius_constrained_entity_ids(self) -> list[int]:
        """Entities whose radius/length is explicitly dimensioned by a
        constraint."""
        constrained: list[int] = []
        for eid in self.registry.entity_ids():
            if not self.registry.is_radius_entity(eid):
                continue
            for constraint in self.constraints:
                constrains = getattr(constraint, "constrains_radius", None)
                if constrains is not None and constrains(
                    self.registry, eid
                ):
                    constrained.append(eid)
                    break
        return constrained

    def _update_entity_constraints(self):
        """
        Updates the constrained status of entities from their defining
        points, honoring constraints that explicitly dimension an
        entity's radius or length.
        """
        self.registry.update_constrained_status(
            self._radius_constrained_entity_ids()
        )

    def get_conflicting_constraints(
        self, threshold: float = CONFLICT_ERROR_THRESHOLD
    ) -> set[int]:
        """
        Identifies constraints that have significant residual error,
        indicating they conflict with other constraints or cannot be
        satisfied.

        Returns:
            Set of constraint indices in self.constraints that are
            conflicting.
        """
        geo_solver = self._build_problem()
        conflicting = geo_solver.conflicting(threshold)
        return set(conflicting)
