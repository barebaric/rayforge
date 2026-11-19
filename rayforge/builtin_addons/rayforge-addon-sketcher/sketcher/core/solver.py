import logging
import time
from collections.abc import Sequence

from raygeo.sketch import PointDef
from raygeo.sketch import Solver as RustSolver
from raygeo.sketch.constraints import Constraint
from raygeo.sketch.entities import SketchPoint as Point
from raygeo.sketch.params import ParameterContext
from raygeo.sketch.registry import EntityRegistry

from .types import EntityID

CONFLICT_ERROR_THRESHOLD = 1e-3

logger = logging.getLogger(__name__)


class Solver:
    """Thin adapter over the Rust sketch solver.

    Snapshots the registry's points into point definitions, runs the
    Rust solver, and writes solved positions and degree-of-freedom
    flags back into the registry. All solver math lives in raygeo.
    """

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

    def _build(self) -> tuple[RustSolver, list[Point]]:
        points = [
            PointDef(p.id, p.x, p.y, p.fixed) for p in self.registry.points()
        ]
        solver = RustSolver(
            points,
            self.registry,
            self.constraints,
            self.auxiliary_constraints or None,
            list(self.point_filter) if self.point_filter is not None else None,
        )
        return solver, self.registry.points()

    def solve(self, tolerance: float = 1e-5, update_dof: bool = True) -> bool:
        """Runs the Rust solver and writes its result into the registry.

        Returns True if every component converged.
        """
        started = time.perf_counter()
        solver, points = self._build()
        success = solver.solve(tolerance, update_dof)

        positions = solver.positions()
        constrained = solver.constrained()
        for point, (x, y), is_constrained in zip(
            points, positions, constrained
        ):
            point.x = x
            point.y = y
            if update_dof:
                point.constrained = is_constrained

        if update_dof and success and self.point_filter is None:
            self._update_entity_constraints()

        logger.debug(
            "Solver.solve: points=%d hard=%d aux=%d iters=%d ms=%.1f "
            "success=%s",
            len(points),
            len(self.constraints),
            len(self.auxiliary_constraints),
            solver.nfev(),
            (time.perf_counter() - started) * 1000.0,
            success,
        )
        return success

    def _update_entity_constraints(self) -> None:
        """Recomputes entity constrained flags from the point flags."""
        radius_ids: list[EntityID] = []
        for entity_id in self.registry.entity_ids():
            for constraint in self.constraints:
                constrains_radius = getattr(
                    constraint, "constrains_radius", None
                )
                if constrains_radius is not None and constrains_radius(
                    self.registry, entity_id
                ):
                    radius_ids.append(entity_id)
                    break
        self.registry.update_constrained_status(radius_ids)

    def get_conflicting_constraints(
        self, threshold: float = CONFLICT_ERROR_THRESHOLD
    ) -> set[int]:
        """Returns indices of hard constraints whose residual exceeds
        the threshold at the registry's current positions."""
        solver, _points = self._build()
        return set(solver.conflicting(threshold))
