import numpy as np
import scipy.linalg
from scipy.optimize import least_squares
from typing import Sequence, List
from .entities import EntityRegistry, Point, Line, Arc, Circle
from .params import ParameterContext
from .constraints import Constraint, RadiusConstraint, DiameterConstraint


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

        # 3. Define the objective function (residuals)
        def objective(x_state):
            # Update registry points directly from vector
            ptr = 0
            for p in mutable_points:
                p.x = x_state[ptr]
                p.y = x_state[ptr + 1]
                ptr += 2

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

        # 4. Solve
        # 'trf' is robust for under-constrained problems (m < n)
        result = least_squares(objective, x0, method="trf", ftol=tolerance)

        # 5. Final Update to ensure registry matches result
        objective(result.x)

        success = bool(result.success and result.cost < tolerance)

        # 6. Analyze Degrees of Freedom (DOF) - CONDITIONALLY
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
        # We use a slightly loose tolerance to account for floating point drift
        null_space = scipy.linalg.null_space(jacobian, rcond=1e-5)

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
        constrained. For circles, it is constrained if its center and radius
        are defined.
        """
        registry = self.registry
        for entity in registry.entities:
            is_fully_constrained = False

            if isinstance(entity, Line):
                p1 = registry.get_point(entity.p1_idx)
                p2 = registry.get_point(entity.p2_idx)
                is_fully_constrained = p1.constrained and p2.constrained

            elif isinstance(entity, Arc):
                s = registry.get_point(entity.start_idx)
                e = registry.get_point(entity.end_idx)
                c = registry.get_point(entity.center_idx)
                is_fully_constrained = (
                    s.constrained and e.constrained and c.constrained
                )

            elif isinstance(entity, Circle):
                center_pt = registry.get_point(entity.center_idx)
                radius_pt = registry.get_point(entity.radius_pt_idx)

                # A circle's geometry is defined by its center and radius.
                center_is_constrained = center_pt.constrained

                # The radius is defined if:
                # 1. An explicit Radius or Diameter constraint exists.
                # 2. Or, the radius point itself is fully constrained.
                radius_is_defined = radius_pt.constrained
                if not radius_is_defined:
                    for constr in self.constraints:
                        if (
                            isinstance(constr, RadiusConstraint)
                            and constr.entity_id == entity.id
                        ):
                            radius_is_defined = True
                            break
                        if (
                            isinstance(constr, DiameterConstraint)
                            and constr.circle_id == entity.id
                        ):
                            radius_is_defined = True
                            break

                is_fully_constrained = (
                    center_is_constrained and radius_is_defined
                )

            entity.constrained = is_fully_constrained
