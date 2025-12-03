import json
import uuid
from pathlib import Path
from typing import Union, List, Optional, Set, Dict, Any, Sequence
from blinker import Signal
from ..geo import Geometry
from ..varset import VarSet
from ..asset import IAsset
from .constraints import (
    Constraint,
    DistanceConstraint,
    EqualDistanceConstraint,
    HorizontalConstraint,
    VerticalConstraint,
    CoincidentConstraint,
    RadiusConstraint,
    DiameterConstraint,
    PointOnLineConstraint,
    PerpendicularConstraint,
    TangentConstraint,
    EqualLengthConstraint,
    SymmetryConstraint,
)
from .entities import EntityRegistry, Line, Arc, Circle
from .params import ParameterContext
from .solver import Solver


_CONSTRAINT_CLASSES = {
    "DistanceConstraint": DistanceConstraint,
    "EqualDistanceConstraint": EqualDistanceConstraint,
    "HorizontalConstraint": HorizontalConstraint,
    "VerticalConstraint": VerticalConstraint,
    "CoincidentConstraint": CoincidentConstraint,
    "RadiusConstraint": RadiusConstraint,
    "DiameterConstraint": DiameterConstraint,
    "PointOnLineConstraint": PointOnLineConstraint,
    "PerpendicularConstraint": PerpendicularConstraint,
    "TangentConstraint": TangentConstraint,
    "EqualLengthConstraint": EqualLengthConstraint,
    "SymmetryConstraint": SymmetryConstraint,
}


class Sketch(IAsset):
    """
    A parametric sketcher that allows defining geometry via constraints
    and expressions.
    """

    def __init__(self, name: str = "New Sketch") -> None:
        self.uid: str = str(uuid.uuid4())
        self._name = name
        self.params = ParameterContext()
        self.registry = EntityRegistry()
        self.constraints: List[Constraint] = []
        self.input_parameters = VarSet(
            title="Input Parameters",
            description="Parameters that control this sketch's geometry.",
        )
        self.updated = Signal()

        # Initialize the Origin Point (Fixed Anchor)
        self.origin_id = self.registry.add_point(0.0, 0.0, fixed=True)

    @property
    def name(self) -> str:
        """The user-facing name of the asset."""
        return self._name

    @name.setter
    def name(self, value: str):
        """Sets the asset name and sends an update signal if changed."""
        if self._name != value:
            self._name = value
            self.updated.send(self)

    @property
    def asset_type_name(self) -> str:
        """The machine-readable type name for the asset list."""
        return "sketch"

    @property
    def display_icon_name(self) -> str:
        """The icon name for the asset list."""
        return "sketch-edit-symbolic"

    @property
    def is_reorderable(self) -> bool:
        """Whether this asset type supports reordering in the asset list."""
        return False

    @property
    def is_draggable_to_canvas(self) -> bool:
        """Whether this asset can be dragged from the list onto the canvas."""
        return True

    @property
    def is_empty(self) -> bool:
        """Returns True if the sketch has no drawable entities."""
        # We check entities rather than points, because an empty sketch
        # always contains at least one point (the origin).
        return len(self.registry.entities) == 0

    @property
    def is_fully_constrained(self) -> bool:
        """
        Returns True if every point and every entity in the sketch
        is fully constrained.

        Exception: Points that serve solely as internal handles for fully
        constrained entities (e.g., Circle radius point) are ignored if they
        are not constrained, provided they are not used by any other entity.
        """
        # An empty sketch (just origin) is considered fully constrained
        if not self.registry.points:
            return True

        # 1. All entities must be constrained
        if not all(e.constrained for e in self.registry.entities):
            return False

        # 2. Calculate point usage counts to ensure exclusive ownership
        usage_count: Dict[int, int] = {}
        for e in self.registry.entities:
            for pid in e.get_point_ids():
                usage_count[pid] = usage_count.get(pid, 0) + 1

        # 3. Collect allowed exemptions polymorphically
        allowed_unconstrained_ids = set()
        for e in self.registry.entities:
            candidates = e.get_ignorable_unconstrained_points()
            for pid in candidates:
                # Only allow exemption if the point is used exclusively by this
                # entity (usage count == 1)
                if usage_count.get(pid, 0) == 1:
                    allowed_unconstrained_ids.add(pid)

        # 4. Check all points
        for p in self.registry.points:
            if not p.constrained:
                # If point is unconstrained, it must be in the exempt list
                if p.id not in allowed_unconstrained_ids:
                    return False

        return True

    def to_dict(self, include_input_values: bool = True) -> Dict[str, Any]:
        """Serializes the Sketch to a dictionary."""
        return {
            "uid": self.uid,
            "type": self.asset_type_name,
            "name": self.name,
            "input_parameters": self.input_parameters.to_dict(
                include_value=include_input_values
            ),
            "params": self.params.to_dict(),
            "registry": self.registry.to_dict(),
            "constraints": [c.to_dict() for c in self.constraints],
            "origin_id": self.origin_id,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Sketch":
        """Deserializes a dictionary into a Sketch instance."""
        required_keys = ["params", "registry", "constraints", "origin_id"]
        if not all(key in data for key in required_keys):
            raise KeyError(
                "Sketch data is missing one of the required keys: "
                f"{required_keys}."
            )

        new_sketch = cls()
        new_sketch.uid = data.get("uid", str(uuid.uuid4()))
        new_sketch.name = data.get("name", "")

        # Handle backward compatibility for input_parameters
        if "input_parameters" in data:
            new_sketch.input_parameters = VarSet.from_dict(
                data["input_parameters"]
            )

        new_sketch.params = ParameterContext.from_dict(data["params"])
        new_sketch.registry = EntityRegistry.from_dict(data["registry"])
        new_sketch.origin_id = data["origin_id"]
        new_sketch.constraints = []
        for c_data in data["constraints"]:
            c_type = c_data.get("type")
            c_cls = _CONSTRAINT_CLASSES.get(c_type)
            if c_cls:
                new_sketch.constraints.append(c_cls.from_dict(c_data))

        return new_sketch

    @classmethod
    def from_file(cls, file_path: Union[str, Path]) -> "Sketch":
        """Deserializes a sketch from a JSON file (.rfs)."""
        with open(file_path, "r") as f:
            data = json.load(f)
        return cls.from_dict(data)

    def set_param(self, name: str, value: Union[str, float]) -> None:
        """Define a parameter like 'width'=100 or 'height'='width/2'."""
        self.params.set(name, value)

    def add_point(self, x: float, y: float, fixed: bool = False) -> int:
        """Adds a point. Returns its ID."""
        return self.registry.add_point(x, y, fixed)

    def add_line(self, p1: int, p2: int, construction: bool = False) -> int:
        """Adds a line segment between two point IDs."""
        return self.registry.add_line(p1, p2, construction)

    def add_arc(
        self,
        start: int,
        end: int,
        center: int,
        clockwise: bool = False,
        construction: bool = False,
    ) -> int:
        """Adds an arc defined by start, end, and center point IDs."""
        return self.registry.add_arc(
            start, end, center, clockwise, construction
        )

    def add_circle(
        self, center: int, radius_pt: int, construction: bool = False
    ) -> int:
        """Adds a circle defined by a center and a point on its radius."""
        return self.registry.add_circle(center, radius_pt, construction)

    # --- Validation ---

    def supports_constraint(
        self,
        constraint_type: str,
        point_ids: Sequence[int],
        entity_ids: Sequence[int],
    ) -> bool:
        """
        Determines if a constraint type is valid for the given selection of
        points and entities.
        """
        # Resolve entities
        entities = []
        for eid in entity_ids:
            e = self.registry.get_entity(eid)
            if e:
                entities.append(e)

        n_pts = len(point_ids)
        n_ents = len(entities)

        lines = [e for e in entities if isinstance(e, Line)]
        arcs = [e for e in entities if isinstance(e, Arc)]
        circles = [e for e in entities if isinstance(e, Circle)]
        n_lines = len(lines)
        n_arcs = len(arcs)
        n_circles = len(circles)

        # 1. Linear/Distance Constraints (Horizontal, Vertical, Distance)
        if constraint_type in ("dist", "horiz", "vert"):
            # Case A: Two Points (valid for all)
            if n_pts == 2 and n_ents == 0:
                return True

            # Case B: Line(s) (check if all entities are lines)
            if n_pts == 0 and n_lines > 0 and n_ents == n_lines:
                if constraint_type == "dist":
                    return n_lines == 1  # Distance is only for a single line
                else:  # horiz, vert
                    return n_lines >= 1  # 1 or more lines are valid
            return False

        # 2. Radius / Diameter
        if constraint_type == "radius":
            return (
                (n_arcs == 1 or n_circles == 1) and n_ents == 1 and n_pts == 0
            )
        if constraint_type == "diameter":
            return n_circles == 1 and n_ents == 1 and n_pts == 0

        # 3. Perpendicular
        if constraint_type == "perp":
            # Valid combinations:
            # - 2 Lines
            # - 1 Line and 1 Arc/Circle
            # - 2 Arcs/Circles
            def is_shape(e):
                return isinstance(e, (Arc, Circle))

            shapes = [e for e in entities if is_shape(e)]
            n_shapes = len(shapes)

            if n_ents != 2 or n_pts != 0:
                return False

            if n_lines == 2:
                return True
            if n_lines == 1 and n_shapes == 1:
                return True
            if n_shapes == 2:
                return True

            return False

        # 4. Tangent
        if constraint_type == "tangent":
            return (
                n_lines == 1
                and (n_arcs == 1 or n_circles == 1)
                and n_ents == 2
                and n_pts == 0
            )

        # 5. Equal
        if constraint_type == "equal":
            # Two or more entities that have a length/radius property
            return (
                n_ents >= 2
                and n_pts == 0
                and all(isinstance(e, (Line, Arc, Circle)) for e in entities)
            )

        # 6. Align (Coincident or Point-on-Line)
        if constraint_type == "align":
            # Coincident: Two points
            supports_coincident = n_pts == 2 and n_ents == 0
            # Point on Shape: One Point and One Shape (Line/Arc/Circle)
            supports_pos = False
            if n_pts == 1 and n_ents == 1:
                # Reuse the more specific check's logic
                supports_pos = self.supports_constraint(
                    "point_on_line", point_ids, entity_ids
                )
            return supports_coincident or supports_pos

        # Internal keys used by add_alignment_constraint
        if constraint_type == "coincident":
            # Two points
            return n_pts == 2 and n_ents == 0

        # 7. Point On Line (now Point On Shape)
        if constraint_type == "point_on_line":
            # One Point and One Shape (Line, Arc, or Circle)
            if n_pts == 1 and n_ents == 1:
                entity = entities[0]
                pid = point_ids[0]

                # Ensure point is not one of the shape's control points
                control_points = set()
                if isinstance(entity, Line):
                    control_points = {entity.p1_idx, entity.p2_idx}
                elif isinstance(entity, Arc):
                    control_points = {
                        entity.start_idx,
                        entity.end_idx,
                        entity.center_idx,
                    }
                elif isinstance(entity, Circle):
                    control_points = {
                        entity.center_idx,
                        entity.radius_pt_idx,
                    }

                if pid not in control_points:
                    return True
            return False

        # 8. Symmetry
        if constraint_type == "symmetry":
            # Case A: Three points (1 center + 2 symmetric)
            if n_pts == 3 and n_ents == 0:
                return True
            # Case B: Two points + 1 Line (Axis)
            if n_pts == 2 and n_lines == 1 and n_ents == 1:
                return True
            return False

        return False

    # --- Constraint Shortcuts ---

    def get_coincident_points(self, start_pid: int) -> Set[int]:
        """
        Finds all points transitively connected to start_pid via
        CoincidentConstraints.
        Returns a set including the starting point itself.
        """
        coincident_group = {start_pid}

        # Use a list as a queue for a breadth-first search
        queue = [start_pid]
        visited = {start_pid}

        head = 0
        while head < len(queue):
            current_pid = queue[head]
            head += 1

            for constr in self.constraints:
                if not isinstance(constr, CoincidentConstraint):
                    continue

                # Find the other point in the constraint
                other_pid = -1
                if constr.p1 == current_pid:
                    other_pid = constr.p2
                elif constr.p2 == current_pid:
                    other_pid = constr.p1

                if other_pid != -1 and other_pid not in visited:
                    visited.add(other_pid)
                    coincident_group.add(other_pid)
                    queue.append(other_pid)

        return coincident_group

    def constrain_distance(
        self, p1: int, p2: int, dist: Union[str, float]
    ) -> DistanceConstraint:
        constr = DistanceConstraint(p1, p2, dist)
        self.constraints.append(constr)
        return constr

    def constrain_equal_distance(
        self, p1: int, p2: int, p3: int, p4: int
    ) -> None:
        """Enforces dist(p1, p2) == dist(p3, p4)."""
        self.constraints.append(EqualDistanceConstraint(p1, p2, p3, p4))

    def constrain_horizontal(self, p1: int, p2: int) -> None:
        self.constraints.append(HorizontalConstraint(p1, p2))

    def constrain_vertical(self, p1: int, p2: int) -> None:
        self.constraints.append(VerticalConstraint(p1, p2))

    def constrain_coincident(self, p1: int, p2: int) -> None:
        self.constraints.append(CoincidentConstraint(p1, p2))

    def constrain_point_on_line(self, point_id: int, shape_id: int) -> None:
        self.constraints.append(PointOnLineConstraint(point_id, shape_id))

    def constrain_radius(
        self, entity_id: int, radius: Union[str, float]
    ) -> RadiusConstraint:
        constr = RadiusConstraint(entity_id, radius)
        self.constraints.append(constr)
        return constr

    def constrain_diameter(
        self, circle_id: int, diameter: Union[str, float]
    ) -> DiameterConstraint:
        constr = DiameterConstraint(circle_id, diameter)
        self.constraints.append(constr)
        return constr

    def constrain_perpendicular(self, l1: int, l2: int) -> None:
        self.constraints.append(PerpendicularConstraint(l1, l2))

    def constrain_tangent(self, line: int, shape: int) -> None:
        self.constraints.append(TangentConstraint(line, shape))

    def constrain_equal_length(self, entity_ids: List[int]) -> None:
        """Enforces equal length/radius between two or more entities."""
        if len(entity_ids) < 2:
            return
        self.constraints.append(EqualLengthConstraint(entity_ids))

    def constrain_symmetry(
        self, point_ids: List[int], entity_ids: List[int]
    ) -> None:
        """
        Enforces symmetry.
        - If 3 points: The first in point_ids is treated as the center.
        - If 2 points + 1 Line: The line is the axis.
        """
        if len(point_ids) == 3 and not entity_ids:
            # 3 Points: First is Center, other two are symmetric
            center = point_ids[0]
            p1 = point_ids[1]
            p2 = point_ids[2]
            self.constraints.append(SymmetryConstraint(p1, p2, center=center))

        elif len(point_ids) == 2 and len(entity_ids) == 1:
            # 2 Points + 1 Line: Line is Axis
            p1 = point_ids[0]
            p2 = point_ids[1]
            axis = entity_ids[0]
            self.constraints.append(SymmetryConstraint(p1, p2, axis=axis))

    # --- Manipulation & Processing ---

    def move_point(self, pid: int, x: float, y: float) -> bool:
        """
        Attempts to move a point to a new location and resolve constraints.
        Returns True if the point was moved, False if it is locked/constrained.
        """
        try:
            p = self.registry.get_point(pid)
        except IndexError:
            return False

        if p.fixed:
            return False

        # Backend Logic: If the solver has determined this point has 0 degrees
        # of freedom (fully constrained), we reject kinematic movement.
        if p.constrained:
            return False

        # Perturbation Strategy: Update initial guess, then solve.
        p.x = x
        p.y = y

        return self.solve()

    def solve(
        self,
        extra_constraints: Optional[List[Constraint]] = None,
        update_constraint_status: bool = True,
        variable_overrides: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """
        Resolves all constraints.

        Args:
            extra_constraints: A list of temporary constraints to add for this
                solve, e.g., for dragging.
            update_constraint_status: If True, re-calculates the degrees of
                freedom for all points and entities after a successful solve.
            variable_overrides: A dictionary of parameter values to use for
                this solve only, without permanently changing the sketch's
                parameters. e.g., `{'width': 150.0}`.

        Returns:
            True if the solver converged successfully.
        """
        success = False
        try:
            # Step 1: Build the evaluation context with correct precedence.
            # a) Start with legacy parameters.
            ctx = {}
            if self.params:
                ctx.update(self.params.get_all_values())

            # b) Add/overwrite with values from the new VarSet system.
            if self.input_parameters is not None:
                ctx.update(self.input_parameters.get_values())

            # c) Add/overwrite with runtime overrides (highest precedence).
            if variable_overrides:
                ctx.update(variable_overrides)

            # Step 2: Update constraints with the final context.
            all_constraints = self.constraints
            if extra_constraints:
                all_constraints = self.constraints + extra_constraints

            for c in all_constraints:
                if hasattr(c, "update_from_context"):
                    c.update_from_context(ctx)

            # Step 3: Run the solver.
            solver = Solver(self.registry, self.params, all_constraints)
            success = solver.solve(update_dof=update_constraint_status)

        except Exception as e:
            import logging

            logging.getLogger(__name__).error(
                f"Sketch solve failed: {e}", exc_info=True
            )
            success = False

        return success

    def to_geometry(self) -> Geometry:
        """
        Converts the solved sketch into a Geometry object.
        Links separate entities into continuous paths where possible.
        """
        geo = Geometry()

        # Simple export: MoveTo -> LineTo/ArcTo for every entity.
        for entity in self.registry.entities:
            # Skip construction geometry (helper lines)
            if entity.construction:
                continue

            if isinstance(entity, Line):
                p1 = self.registry.get_point(entity.p1_idx)
                p2 = self.registry.get_point(entity.p2_idx)

                geo.move_to(p1.x, p1.y)
                geo.line_to(p2.x, p2.y)

            elif isinstance(entity, Arc):
                start = self.registry.get_point(entity.start_idx)
                end = self.registry.get_point(entity.end_idx)
                center = self.registry.get_point(entity.center_idx)

                # Geometry.arc_to requires center_offset (i, j) relative
                # to the current point (start)
                i = center.x - start.x
                j = center.y - start.y

                geo.move_to(start.x, start.y)
                geo.arc_to(end.x, end.y, i, j, clockwise=entity.clockwise)

            elif isinstance(entity, Circle):
                center = self.registry.get_point(entity.center_idx)
                radius_pt = self.registry.get_point(entity.radius_pt_idx)

                # Draw as two semi-circles
                dx = radius_pt.x - center.x
                dy = radius_pt.y - center.y
                opposite_pt_x = center.x - dx
                opposite_pt_y = center.y - dy

                # Center offset relative to start point
                i1, j1 = -dx, -dy
                # Center offset relative to mid-point
                i2, j2 = dx, dy

                geo.move_to(radius_pt.x, radius_pt.y)
                # First semi-circle
                geo.arc_to(
                    opposite_pt_x, opposite_pt_y, i1, j1, clockwise=False
                )
                # Second semi-circle
                geo.arc_to(radius_pt.x, radius_pt.y, i2, j2, clockwise=False)

        return geo
