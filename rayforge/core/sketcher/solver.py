import numpy as np
import scipy.linalg
from scipy.optimize import least_squares
from typing import Sequence, List
from .entities import EntityRegistry, Point
from .params import ParameterContext
from .constraints import Constraint


class Solver:
    def __init__(
        self,
        registry: EntityRegistry,
        params: ParameterContext,
        constraints: Sequence[Constraint],
    ):
        self.registry = registry
        self.params = params
        self.constraints = constraints

    def solve(self, tolerance: float = 1e-6, update_dof: bool = True) -> bool:
        """
        Runs the least_squares optimizer to satisfy constraints.
        Returns True if successful.
        If update_dof is False, it will skip re-calculating the constrained
        status of points and entities, which is useful for interactive updates.
        """
        # 1. Identify mutable points (degrees of freedom)
        mutable_points: List[Point] = [
            p for p in self.registry.points if not p.fixed
        ]

        # Map point_id -> index in state vector (0, 2, 4...)
        point_indices = {p.id: i * 2 for i, p in enumerate(mutable_points)}

        # Only reset if we are doing a full DOF update
        if update_dof:
            for p in self.registry.points:
                p.constrained = p.fixed

        if not mutable_points:
            if update_dof:
                self._update_entity_constraints()
            return True  # Nothing to solve

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
            for const in self.constraints:
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

        # 4. Define the Jacobian function
        def jacobian(x_state):
            # Ensure registry is up-to-date (least_squares might call jac with
            # the same x as obj, or different)
            update_registry(x_state)

            n_vars = len(x0)
            rows = []

            for const in self.constraints:
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

            if not rows:
                return np.zeros((1, n_vars))

            return np.vstack(rows)

        # 5. Solve
        # 'trf' is robust for under-constrained problems (m < n)
        # We pass the analytical jacobian
        result = least_squares(
            objective,
            x0,
            jac=jacobian,  # type: ignore
            method="trf",
            ftol=tolerance,
        )

        # 6. Final Update to ensure registry matches result
        update_registry(result.x)

        success = bool(result.success and result.cost < tolerance)

        # 7. Analyze Degrees of Freedom (DOF) - CONDITIONALLY
        if success and update_dof:
            self._analyze_dof(result.jac, mutable_points)
            self._update_entity_constraints()

        return success

    def _analyze_dof(self, jacobian: np.ndarray, mutable_points: List[Point]):
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
