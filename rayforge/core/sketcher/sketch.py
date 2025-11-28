from typing import Union, List, Optional, Set, Dict, Any, Sequence
from ..geo import Geometry
from .params import ParameterContext
from .entities import EntityRegistry, Line, Arc, Circle
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
)
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
}


class Sketch:
    """
    A parametric sketcher that allows defining geometry via constraints
    and expressions.
    """

    def __init__(self) -> None:
        self.params = ParameterContext()
        self.registry = EntityRegistry()
        self.constraints: List[Constraint] = []

        # Initialize the Origin Point (Fixed Anchor)
        self.origin_id = self.registry.add_point(0.0, 0.0, fixed=True)

    def to_dict(self) -> Dict[str, Any]:
        """Serializes the Sketch to a dictionary."""
        return {
            "params": self.params.to_dict(),
            "registry": self.registry.to_dict(),
            "constraints": [c.to_dict() for c in self.constraints],
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Sketch":
        """Deserializes a dictionary into a Sketch instance."""
        # Check for required keys to prevent creating an empty sketch from
        # invalid data.
        if not all(
            key in data for key in ["params", "registry", "constraints"]
        ):
            raise KeyError(
                "Sketch data is missing one of the required keys: "
                "'params', 'registry', 'constraints'."
            )

        new_sketch = cls.__new__(
            cls
        )  # Create instance without calling __init__

        new_sketch.params = ParameterContext.from_dict(data["params"])
        new_sketch.registry = EntityRegistry.from_dict(data["registry"])
        new_sketch.constraints = []

        # Find the origin_id from the loaded registry points
        origin_point = next(
            (p for p in new_sketch.registry.points if p.fixed), None
        )
        new_sketch.origin_id = origin_point.id if origin_point else -1

        for c_data in data["constraints"]:
            c_type = c_data.get("type")
            c_cls = _CONSTRAINT_CLASSES.get(c_type)
            if c_cls:
                new_sketch.constraints.append(c_cls.from_dict(c_data))

        return new_sketch

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
            # Case A: Two Points
            if n_pts == 2 and n_ents == 0:
                return True
            # Case B: One Line
            if n_pts == 0 and n_lines == 1 and n_ents == 1:
                return True
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
            # Exactly two Lines
            return n_lines == 2 and n_ents == 2 and n_pts == 0

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
    ) -> bool:
        """Resolves all constraints."""
        all_constraints = self.constraints
        if extra_constraints:
            all_constraints = self.constraints + extra_constraints

        solver = Solver(self.registry, self.params, all_constraints)
        return solver.solve(update_dof=update_constraint_status)

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
