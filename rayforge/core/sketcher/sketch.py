import json
import uuid
from pathlib import Path
from typing import Union, List, Optional, Set, Dict, Any, Sequence, Tuple
from blinker import Signal
from collections import defaultdict
import math
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
    CollinearConstraint,
)
from .entities import Line, Arc, Circle, Entity
from .params import ParameterContext
from .registry import EntityRegistry
from .solver import Solver


_DEFAULT_VARSET_TITLE = _("Sketch Parameters")
_DEFAULT_VARSET_DESCRIPTION = _(
    "Parameters that control this sketch's geometry"
)


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
    "CollinearConstraint": CollinearConstraint,
}


class Fill:
    """Represents a filled area bounded by sketch entities."""

    def __init__(self, uid: str, boundary: List[Tuple[int, bool]]):
        self.uid = uid
        self.boundary = boundary  # List of (entity_id, is_forward_traversal)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "uid": self.uid,
            "boundary": [list(item) for item in self.boundary],
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Fill":
        # JSON deserializes tuples as lists. Convert back to tuples so they
        # are hashable (required for FillTool set operations).
        boundary = [tuple(item) for item in data["boundary"]]
        return cls(
            uid=data.get("uid", str(uuid.uuid4())),
            boundary=boundary,
        )


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
        self.fills: List[Fill] = []
        self.input_parameters = VarSet(
            title=_DEFAULT_VARSET_TITLE,
            description=_DEFAULT_VARSET_DESCRIPTION,
        )
        self.updated = Signal()

        # Initialize the Origin Point (Fixed Anchor)
        self.origin_id = self.registry.add_point(0.0, 0.0, fixed=True)

    def notify_update(self):
        """Public method to signal that the sketch has been modified."""
        self.updated.send(self)

    def _validate_and_cleanup_fills(self):
        """
        Removes any Fill objects whose boundary entities no longer form a
        valid, closed loop (e.g., if an entity was deleted).
        """
        valid_fills = []
        # Find all currently valid loops to check against
        current_loops = self._find_all_closed_loops()
        # For efficient lookup, convert lists to sets of tuples
        current_loop_sets = {frozenset(loop) for loop in current_loops}

        for fill in self.fills:
            fill_boundary_set = frozenset(fill.boundary)
            if fill_boundary_set in current_loop_sets:
                valid_fills.append(fill)

        self.fills = valid_fills

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
                include_value=include_input_values, include_metadata=False
            ),
            "params": self.params.to_dict(),
            "registry": self.registry.to_dict(),
            "constraints": [c.to_dict() for c in self.constraints],
            "fills": [f.to_dict() for f in self.fills],
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
            # Re-apply the default title and description, as they are not
            # serialized in the file.
            new_sketch.input_parameters.title = _DEFAULT_VARSET_TITLE
            new_sketch.input_parameters.description = (
                _DEFAULT_VARSET_DESCRIPTION
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

        new_sketch.fills = []
        for f_data in data.get("fills", []):
            new_sketch.fills.append(Fill.from_dict(f_data))

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

    def remove_entities(self, entities_to_remove: List[Entity]):
        """
        Removes entities from the sketch and automatically cleans up any
        dependent fills.
        """
        if not entities_to_remove:
            return
        ids_to_remove = [e.id for e in entities_to_remove]
        self.registry.remove_entities_by_id(ids_to_remove)
        self._validate_and_cleanup_fills()

    def remove_point_if_unused(self, pid: Optional[int]) -> bool:
        """
        Removes a point from the registry if it's not part of any entity.

        Args:
            pid: The point ID to remove. If None, returns False.

        Returns:
            True if the point was removed, False otherwise.
        """
        if pid is None:
            return False
        if not self.registry.is_point_used(pid):
            self.registry.points = [
                p for p in self.registry.points if p.id != pid
            ]
            return True
        return False

    def _get_edge_tangent_at_start(
        self, entity: Any, start_pid: int
    ) -> Tuple[float, float]:
        """Helper to get the tangent vector for an entity at a given point."""
        if isinstance(entity, Line):
            p1 = self.registry.get_point(entity.p1_idx)
            p2 = self.registry.get_point(entity.p2_idx)
            if start_pid == p1.id:
                return (p2.x - p1.x, p2.y - p1.y)
            else:
                return (p1.x - p2.x, p1.y - p2.y)

        elif isinstance(entity, Arc):
            start = self.registry.get_point(entity.start_idx)
            center = self.registry.get_point(entity.center_idx)
            if start_pid == start.id:
                # Traversing forward from the arc's start point
                # Tangent of circle at P is perp to Radius CP.
                # If CCW: (-dy, dx). If CW: (dy, -dx).
                dx, dy = start.x - center.x, start.y - center.y
                return (dy, -dx) if entity.clockwise else (-dy, dx)
            else:
                # Traversing backward from the arc's end point
                end = self.registry.get_point(entity.end_idx)
                dx, dy = end.x - center.x, end.y - center.y
                # Tangent of curve at End is T. Traversal is -T.
                # T_ccw = (-dy, dx). Traversal = (dy, -dx).
                # T_cw = (dy, -dx). Traversal = (-dy, dx).
                return (-dy, dx) if entity.clockwise else (dy, -dx)
        return (1.0, 0.0)

    def _build_adjacency_list(self) -> Dict[int, List[Dict[str, Any]]]:
        """
        Builds a map of point_id -> list of outgoing edges.
        Each edge dict contains: {'to': point_id, 'id': entity_id, 'fwd': bool}

        Coincident points are treated as the same node in the graph, so edges
        are added for all points in a coincident group.
        """
        adj = defaultdict(list)

        # Build a mapping from each point to its coincident group
        point_to_group = {}
        for p in self.registry.points:
            if p.id not in point_to_group:
                coincident_group = self.get_coincident_points(p.id)
                for pid in coincident_group:
                    point_to_group[pid] = coincident_group

        for e in self.registry.entities:
            # Skip circles in graph traversal (handled separately)
            if isinstance(e, Circle):
                continue
            if isinstance(e, (Line, Arc)):
                p_ids = e.get_point_ids()
                # Lines/Arcs have 2 endpoints for traversal
                p1_id, p2_id = p_ids[0], p_ids[1]

                # Get the coincident groups for both endpoints
                group1 = point_to_group.get(p1_id, {p1_id})
                group2 = point_to_group.get(p2_id, {p2_id})

                # Add edges from all points in group1 to all points in group2
                for src in group1:
                    for dst in group2:
                        if src != dst:
                            adj[src].append(
                                {"to": dst, "id": e.id, "fwd": True}
                            )

                # Add edges from all points in group2 to all points in group1
                for src in group2:
                    for dst in group1:
                        if src != dst:
                            adj[src].append(
                                {"to": dst, "id": e.id, "fwd": False}
                            )
        return adj

    def _sort_edges_by_angle(
        self, adj: Dict[int, List[Dict[str, Any]]]
    ) -> Dict[int, List[Dict[str, Any]]]:
        """
        Sorts the outgoing edges at each node by angle (CCW).
        """
        sorted_adj = {}
        for p_id, edges in adj.items():
            edges_with_angle = []
            for edge in edges:
                entity = self.registry.get_entity(edge["id"])
                if not entity:
                    continue
                tangent_vec = self._get_edge_tangent_at_start(entity, p_id)
                angle = math.atan2(tangent_vec[1], tangent_vec[0])
                edges_with_angle.append({"angle": angle, **edge})

            # Sort by angle [-pi, pi]
            edges_with_angle.sort(key=lambda x: x["angle"])
            sorted_adj[p_id] = edges_with_angle
        return sorted_adj

    def _get_next_edge_ccw(
        self,
        current_p_id: int,
        incoming_entity_id: int,
        incoming_fwd: bool,
        sorted_adj: Dict[int, List[Dict[str, Any]]],
    ) -> Optional[Dict[str, Any]]:
        """
        Given an incoming edge to a node, picks the next edge in CCW order
        (left-most turn) to traverse faces.
        """
        outgoing_edges = sorted_adj.get(current_p_id, [])
        if not outgoing_edges:
            return None

        # If we arrived via `incoming_entity_id` traveling `incoming_fwd`,
        # then looking back from the current node, that edge is the reverse.
        rev_fwd = not incoming_fwd

        try:
            # Find the edge entry in the current node's list that corresponds
            # to where we came from.
            idx = next(
                i
                for i, e in enumerate(outgoing_edges)
                if e["id"] == incoming_entity_id and e["fwd"] == rev_fwd
            )
            # Pick the previous edge in the sorted list (CCW rotation)
            next_idx = (idx - 1) % len(outgoing_edges)
            return outgoing_edges[next_idx]
        except StopIteration:
            return None

    def _calculate_loop_signed_area(
        self, loop: List[Tuple[int, bool]]
    ) -> float:
        """Calculates signed area of the loop using Shoelace formula."""
        if not loop:
            return 0.0

        # Special case for circles
        if len(loop) == 1:
            entity = self.registry.get_entity(loop[0][0])
            if isinstance(entity, Circle):
                center = self.registry.get_point(entity.center_idx)
                radius_pt = self.registry.get_point(entity.radius_pt_idx)
                radius = math.hypot(
                    radius_pt.x - center.x, radius_pt.y - center.y
                )
                # By convention, a single circle loop is CCW -> positive area
                return math.pi * radius**2

        points = []
        # Calculate polygon area (straight chords)
        first_ent = self.registry.get_entity(loop[0][0])
        if not first_ent:
            return 0.0
        first_fwd = loop[0][1]
        p_ids = first_ent.get_point_ids()
        curr_p_id = p_ids[0] if first_fwd else p_ids[1]

        for eid, fwd in loop:
            try:
                pt = self.registry.get_point(curr_p_id)
                points.append((pt.x, pt.y))
                ent = self.registry.get_entity(eid)
                if not ent:
                    return 0.0
                p_ids = ent.get_point_ids()
                curr_p_id = p_ids[1] if curr_p_id == p_ids[0] else p_ids[0]
            except IndexError:
                return 0.0

        area = 0.0
        for i in range(len(points)):
            p1 = points[i]
            p2 = points[(i + 1) % len(points)]
            area += p1[0] * p2[1] - p2[0] * p1[1]
        area *= 0.5

        # Add contributions from Arcs (area between chord and arc)
        for eid, fwd in loop:
            ent = self.registry.get_entity(eid)
            if isinstance(ent, Arc):
                # Calculate area of the circular segment
                start = self.registry.get_point(ent.start_idx)
                end = self.registry.get_point(ent.end_idx)
                center = self.registry.get_point(ent.center_idx)

                # Vectors from center
                r_vec_start = (start.x - center.x, start.y - center.y)
                r_vec_end = (end.x - center.x, end.y - center.y)
                radius_sq = r_vec_start[0] ** 2 + r_vec_start[1] ** 2

                # Calculate sweep angle of the arc definition
                ang_start = math.atan2(r_vec_start[1], r_vec_start[0])
                ang_end = math.atan2(r_vec_end[1], r_vec_end[0])

                if ent.clockwise:
                    # CW: Start -> End decreases angle
                    diff = ang_start - ang_end
                else:
                    # CCW: Start -> End increases angle
                    diff = ang_end - ang_start

                # Normalize to [0, 2pi)
                while diff < 0:
                    diff += 2 * math.pi
                while diff >= 2 * math.pi:
                    diff -= 2 * math.pi

                # Area of segment = 0.5 * r^2 * (theta - sin(theta))
                # This area is always positive.
                seg_area = 0.5 * radius_sq * (diff - math.sin(diff))

                # Determine sign contribution to the loop area (assumed
                # CCW positive).
                # If Arc is CCW and we traverse Fwd: Left turn. Add.
                # If Arc is CW and we traverse Fwd: Right turn. Subtract.
                # If Arc is CCW and we traverse Rev: Right turn. Subtract.
                # If Arc is CW and we traverse Rev: Left turn. Add.

                is_ccw_arc = not ent.clockwise
                is_left_turn = is_ccw_arc == fwd

                if is_left_turn:
                    area += seg_area
                else:
                    area -= seg_area

        return area

    def _find_all_closed_loops(self) -> List[List[Tuple[int, bool]]]:
        """
        Finds all closed loops (faces) in the sketch graph.
        """
        adj = self._build_adjacency_list()
        sorted_adj = self._sort_edges_by_angle(adj)

        loops = []
        visited_half_edges: Set[Tuple[int, int, bool]] = set()

        for p_start, edges in sorted_adj.items():
            for start_edge in edges:
                half_edge_key = (p_start, start_edge["id"], start_edge["fwd"])
                if half_edge_key in visited_half_edges:
                    continue

                loop: List[Tuple[int, bool]] = []
                loop_half_edges: List[Tuple[int, int, bool]] = []
                curr_p = p_start
                curr_edge = start_edge

                for _ in range(len(self.registry.entities) + 1):
                    current_half_edge = (
                        curr_p,
                        curr_edge["id"],
                        curr_edge["fwd"],
                    )
                    if current_half_edge in visited_half_edges:
                        loop = []
                        break

                    loop.append((curr_edge["id"], curr_edge["fwd"]))
                    loop_half_edges.append(current_half_edge)

                    next_p = curr_edge["to"]

                    next_edge_info = self._get_next_edge_ccw(
                        next_p, curr_edge["id"], curr_edge["fwd"], sorted_adj
                    )

                    if not next_edge_info:
                        loop = []
                        break

                    next_key = (
                        next_p,
                        next_edge_info["id"],
                        next_edge_info["fwd"],
                    )
                    if next_key == half_edge_key:
                        break  # Loop closed

                    curr_p = next_p
                    curr_edge = next_edge_info
                else:
                    loop = []  # Loop did not close

                if loop:
                    if self._calculate_loop_signed_area(loop) > 1e-6:
                        loops.append(loop)
                        # Mark all half-edges from the valid loop as visited
                        visited_half_edges.update(loop_half_edges)

        # Add circles as single-entity loops
        for e in self.registry.entities:
            if isinstance(e, Circle):
                loops.append([(e.id, True)])

        return loops

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
            # Step 1: Create a disposable ParameterContext clone for this
            # solve.
            solve_params = ParameterContext.from_dict(self.params.to_dict())

            # Step 2: Build the seed dictionary, starting with defaults from
            # the VarSet, then applying instance-specific overrides.
            initial_values = {}
            if self.input_parameters:
                initial_values.update(self.input_parameters.get_values())
            if variable_overrides:
                initial_values.update(variable_overrides)

            # Step 3: Evaluate all expressions from scratch using the temporary
            # context, seeded with the combined values.
            solve_params.evaluate_all(initial_values=initial_values)
            ctx = solve_params.get_all_values()

            # Step 4: Update constraints with the final, resolved values.
            all_constraints = self.constraints
            if extra_constraints:
                all_constraints = self.constraints + extra_constraints
            for c in all_constraints:
                if hasattr(c, "update_from_context"):
                    c.update_from_context(ctx)

            # Step 5: Run the solver with the disposable, correctly evaluated
            # context.
            solver = Solver(self.registry, solve_params, all_constraints)
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

    def get_fill_geometries(self) -> List[Geometry]:
        """
        Generates Geometry objects for all defined fills.
        Each geometry object represents a single closed filled region.
        """
        fill_geometries = []
        for fill in self.fills:
            if not fill.boundary:
                continue

            geo = Geometry()

            # Case 1: Single entity loop (Circle)
            if len(fill.boundary) == 1:
                eid, _ = fill.boundary[0]
                entity = self.registry.get_entity(eid)
                if isinstance(entity, Circle):
                    center = self.registry.get_point(entity.center_idx)
                    radius_pt = self.registry.get_point(entity.radius_pt_idx)

                    # Draw as two semi-circles to form a closed loop
                    dx = radius_pt.x - center.x
                    dy = radius_pt.y - center.y

                    # Start at radius point
                    geo.move_to(radius_pt.x, radius_pt.y)

                    # To opposite point
                    opposite_x = center.x - dx
                    opposite_y = center.y - dy

                    # Offset from start (radius_pt) to center is (-dx, -dy)
                    geo.arc_to(
                        opposite_x, opposite_y, -dx, -dy, clockwise=False
                    )

                    # Back to start. Offset from opposite to center is (dx, dy)
                    geo.arc_to(
                        radius_pt.x, radius_pt.y, dx, dy, clockwise=False
                    )

                    fill_geometries.append(geo)
                continue

            # Case 2: Multi-segment loop
            try:
                # 1. Start point
                first_eid, first_fwd = fill.boundary[0]
                first_ent = self.registry.get_entity(first_eid)
                if not first_ent:
                    continue

                p_ids = first_ent.get_point_ids()
                # If forward: Start -> End. Start is p_ids[0].
                # If backward: End -> Start. Start is p_ids[1].
                start_pid = p_ids[0] if first_fwd else p_ids[1]
                start_pt = self.registry.get_point(start_pid)

                geo.move_to(start_pt.x, start_pt.y)

                valid_loop = True

                for eid, fwd in fill.boundary:
                    entity = self.registry.get_entity(eid)
                    if not entity:
                        valid_loop = False
                        break

                    if isinstance(entity, Line):
                        p_ids = entity.get_point_ids()
                        # If fwd, end is p2 (index 1).
                        # If back, end is p1 (index 0).
                        end_pid = p_ids[1] if fwd else p_ids[0]
                        end_pt = self.registry.get_point(end_pid)
                        geo.line_to(end_pt.x, end_pt.y)

                    elif isinstance(entity, Arc):
                        arc_start_pt = self.registry.get_point(
                            entity.start_idx
                        )
                        arc_end_pt = self.registry.get_point(entity.end_idx)
                        center_pt = self.registry.get_point(entity.center_idx)

                        # Target point of this segment
                        target_pt = arc_end_pt if fwd else arc_start_pt

                        # Current point (start of this segment)
                        current_pt = arc_start_pt if fwd else arc_end_pt

                        # Center offset relative to current point
                        offset_x = center_pt.x - current_pt.x
                        offset_y = center_pt.y - current_pt.y

                        # Determine direction for geometry
                        # If traversing fwd (S->E), use entity direction.
                        # If traversing back (E->S), invert entity direction.
                        is_cw = (
                            entity.clockwise if fwd else not entity.clockwise
                        )

                        geo.arc_to(
                            target_pt.x,
                            target_pt.y,
                            offset_x,
                            offset_y,
                            clockwise=is_cw,
                        )

                if valid_loop:
                    fill_geometries.append(geo)

            except (IndexError, AttributeError):
                continue

        return fill_geometries
