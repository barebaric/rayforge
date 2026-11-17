import logging
import time
from collections.abc import Sequence

import numpy as np
import scipy.linalg
from scipy.optimize import least_squares
from threadpoolctl import ThreadpoolController

from .components import compute_constraint_components, get_referenced_points
from .constraints import Constraint
from .entities import Point
from .params import ParameterContext
from .registry import EntityRegistry
from .types import EntityID

CONFLICT_ERROR_THRESHOLD = 1e-3

logger = logging.getLogger(__name__)

_blas_controller = ThreadpoolController()


def _blas_limiter():
    """Returns a context manager that runs BLAS-backed code
    single-threaded.

    The solver's matrices are small; multithreaded OpenBLAS spends
    far more on thread synchronization around their factorizations
    than on the factorizations themselves. The controller is scanned
    once and reused — constructing a limiter per call rescans the
    process maps, which costs more than the solves."""
    return _blas_controller.limit(limits=1)


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
        self.constraints = constraints
        self.auxiliary_constraints = auxiliary_constraints
        self.point_filter = point_filter

    def solve(self, tolerance: float = 1e-5, update_dof: bool = True) -> bool:
        """
        Runs the least_squares optimizer to satisfy constraints.
        Returns True if successful.

        A global solve (no point_filter) partitions the constraints
        into independent connected components and solves each one
        separately, so least_squares work and the DOF analysis scale
        with the constraint islands instead of the whole sketch.
        Success is the conjunction of the component results, and the
        DOF analysis runs per component on its own small Jacobian.

        If update_dof is False, the constrained status of points and
        entities is left untouched, which is useful for interactive
        updates.

        If point_filter is set, only the listed points are optimized
        and the DOF analysis is restricted to them; all other points
        keep their positions.
        """
        if update_dof:
            for p in self.registry.points:
                p.constrained = p.fixed

        if self.point_filter is None:
            return self._solve_decomposed(tolerance, update_dof)
        return self._solve_scoped(tolerance, update_dof)

    def _solve_decomposed(self, tolerance: float, update_dof: bool) -> bool:
        """Solves each connected constraint component independently.

        The constrained-status reset already ran in solve(); each part
        analyzes its own DOF, and the entity statuses are updated once
        for the whole sketch after every component succeeded."""
        parts = self._partition_constraints()
        if not parts:
            if update_dof:
                self._update_entity_constraints()
            return True

        success = True
        for constraints, auxiliary, point_filter in parts:
            part_solver = Solver(
                self.registry,
                self.params,
                constraints,
                auxiliary_constraints=auxiliary,
                point_filter=point_filter,
            )
            if not part_solver._solve_scoped(tolerance, update_dof):
                success = False

        if update_dof and success:
            self._update_entity_constraints()
        return success

    def _partition_constraints(
        self,
    ) -> list[tuple[list[Constraint], list[Constraint], set[EntityID]]]:
        """Splits the hard and auxiliary constraints into one
        sub-problem per connected component of the constraint graph.
        The components are the graph's connected components, so every
        constraint's referenced points lie in exactly one of them.

        Returns (constraints, auxiliary_constraints, point_filter)
        tuples, one per non-empty part."""
        all_constraints = list(self.constraints) + list(
            self.auxiliary_constraints
        )
        components = compute_constraint_components(
            self.registry, all_constraints
        )
        component_of: dict[EntityID, int] = {}
        for index, component in enumerate(components):
            for pid in component:
                component_of[pid] = index

        hard_parts: list[list[Constraint]] = [[] for _ in components]
        aux_parts: list[list[Constraint]] = [[] for _ in components]

        def assign(constraint: Constraint, parts: list[list[Constraint]]):
            for pid in get_referenced_points(self.registry, constraint):
                index = component_of.get(pid)
                if index is not None:
                    parts[index].append(constraint)
                    return

        for constraint in self.constraints:
            assign(constraint, hard_parts)
        for constraint in self.auxiliary_constraints:
            assign(constraint, aux_parts)

        return [
            (hard, aux, component)
            for hard, aux, component in zip(hard_parts, aux_parts, components)
            if hard or aux
        ]

    def _solve_scoped(self, tolerance: float, update_dof: bool) -> bool:
        """Solves this solver's own constraint set.

        The constrained-status reset and the entity status update are
        the caller's responsibility: a decomposed run resets once up
        front, analyzes each component's DOF on its own small
        Jacobian, and updates the entity statuses once at the end."""
        solve_started = time.perf_counter()

        # 1. Identify mutable points (degrees of freedom). Points that
        # no constraint references cannot move; they stay out of the
        # state vector entirely.
        touched: set[EntityID] = set()
        for const in self.constraints:
            touched |= get_referenced_points(self.registry, const)
        for const in self.auxiliary_constraints:
            touched |= get_referenced_points(self.registry, const)

        mutable_points: list[Point] = [
            p
            for p in self.registry.points
            if not p.fixed
            and p.id in touched
            and (self.point_filter is None or p.id in self.point_filter)
        ]

        if not mutable_points:
            return True  # Nothing to solve

        # Map point_id -> index in state vector (0, 2, 4...)
        point_indices = {p.id: i * 2 for i, p in enumerate(mutable_points)}

        all_constraints = list(self.constraints) + list(
            self.auxiliary_constraints
        )

        # 2. Extract initial state vector [x0, y0, x1, y1, ...]
        x0_list = []
        for p in mutable_points:
            x0_list.extend([p.x, p.y])

        x0 = np.array(x0_list)

        def update_registry(x_state):
            """Updates registry points directly from vector."""
            ptr = 0
            for p in mutable_points:
                p.x = x_state[ptr]
                p.y = x_state[ptr + 1]
                ptr += 2

        # 3. Define the objective function (residuals)
        def objective(x_state):
            update_registry(x_state)

            # Calculate errors
            residuals = []
            for const in all_constraints:
                err = const.error(self.registry, self.params)
                # Flatten the error result into the residuals list
                if isinstance(err, (tuple, list)):
                    residuals.extend(err)
                else:
                    residuals.append(err)

            # If there are no constraints but we have mutable points,
            # we need at least one residual for least_squares.
            if not residuals:
                return np.array([0.0])

            return np.array(residuals)

        # Helper to build Jacobian rows for a specific set of constraints
        def build_jacobian_rows(constraint_list, n_vars):
            rows = []
            for const in constraint_list:
                grad_map = const.gradient(self.registry, self.params)

                # Determine how many residuals this constraint produces
                # We can infer this from the gradient map lists
                num_residuals = 0
                if grad_map:
                    first_val = next(iter(grad_map.values()))
                    num_residuals = len(first_val)
                else:
                    # Fallback check if gradient not implemented but error
                    # exists
                    err = const.error(self.registry, self.params)
                    if isinstance(err, (tuple, list)):
                        num_residuals = len(err)
                    else:
                        num_residuals = 1

                # Create zero rows for these residuals
                for _ in range(num_residuals):
                    rows.append(np.zeros(n_vars))

                start_row = len(rows) - num_residuals

                # Fill in the gradients
                for pid, grads in grad_map.items():
                    if pid in point_indices:
                        idx = point_indices[pid]
                        for i, (dx, dy) in enumerate(grads):
                            current_row = rows[start_row + i]
                            current_row[idx] = dx
                            current_row[idx + 1] = dy

            return rows

        # 4. Define the Jacobian function
        def jacobian(x_state):
            # Ensure registry is up-to-date
            update_registry(x_state)
            n_vars = len(x0)

            rows = build_jacobian_rows(all_constraints, n_vars)

            if not rows:
                return np.zeros((1, n_vars))

            return np.vstack(rows)

        # 5. Solve
        # 'trf' is robust for under-constrained problems (m < n)
        # We pass the analytical jacobian. An iteration limit protects
        # against pathological constraint graphs that would stall the
        # solver indefinitely.
        # 5. Solve
        # 'trf' is robust for under-constrained problems (m < n).
        # We pass the analytical jacobian. An iteration limit protects
        # against pathological constraint graphs that would stall the
        # solver indefinitely. The optimizer runs single-threaded
        # (see _blas_limiter).
        with _blas_limiter():
            result = least_squares(
                objective,
                x0,
                jac=jacobian,  # type: ignore
                method="trf",
                ftol=tolerance,
                xtol=1e-8,
                max_nfev=5000,
            )

        # 6. Final Update to ensure registry matches result
        update_registry(result.x)

        success = bool(result.success and result.cost <= tolerance)

        # 7. Analyze Degrees of Freedom (DOF) - CONDITIONALLY, scoped
        # to this problem's own points and constraints.
        if success and update_dof:
            # Re-compute Jacobian using ONLY hard constraints for DOF
            # analysis. Stabilizer constraints (auxiliary) should not count
            # towards DOF. A component without hard constraints keeps its
            # points unconstrained from the reset; no null space analysis
            # is needed for that.
            hard_rows = build_jacobian_rows(self.constraints, len(x0))
            if hard_rows:
                with _blas_limiter():
                    self._analyze_dof(np.vstack(hard_rows), mutable_points)

        logger.debug(
            "Solver.solve: mutable=%d hard=%d aux=%d iters=%d ms=%.1f "
            "success=%s",
            len(mutable_points),
            len(self.constraints),
            len(self.auxiliary_constraints),
            result.nfev,
            (time.perf_counter() - solve_started) * 1000.0,
            success,
        )

        return success

    def _analyze_dof(self, jacobian: np.ndarray, mutable_points: list[Point]):
        """
        Determines which points are fully constrained by analyzing the
        Null Space of the Jacobian matrix.
        """
        # If Jacobian is (n_constraints, n_vars), the Null Space represents
        # directions in which variables can move without changing residuals.
        # If the Null Space is empty, the system is fully constrained.

        # Get the null space basis ( orthonormal columns )
        # Using a tighter tolerance (1e-9) prevents false positives for DOF
        # when the system is actually rigid but has scaling differences.
        null_space = scipy.linalg.null_space(jacobian, rcond=1e-9)

        # null_space shape is (n_vars, n_dof)
        # If n_dof == 0, everything is constrained.

        if null_space.size == 0:
            for p in mutable_points:
                p.constrained = True
            return

        # If we have DOFs, we need to see which variables participate in them.
        # Rows of null_space correspond to [x0, y0, x1, y1, ...]
        n_vars = null_space.shape[0]

        for i, p in enumerate(mutable_points):
            idx_x = i * 2
            idx_y = i * 2 + 1

            if idx_x >= n_vars:
                break

            # Check magnitude of the point's contribution to the null space.
            # If row vectors in null space are zero (or near zero),
            # this variable cannot move effectively.
            x_mobility = np.sum(np.abs(null_space[idx_x, :]))
            y_mobility = np.sum(np.abs(null_space[idx_y, :]))

            # If mobility is negligible, the point is constrained.
            p.constrained = (x_mobility < 1e-4) and (y_mobility < 1e-4)

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
        conflicting: set[int] = set()

        for idx, const in enumerate(self.constraints):
            err = const.error(self.registry, self.params)
            if isinstance(err, (tuple, list)):
                max_err = max(abs(e) for e in err)
            else:
                max_err = abs(err)

            if max_err > threshold:
                conflicting.add(idx)

        return conflicting
