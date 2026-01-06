from __future__ import annotations
import logging
from typing import TYPE_CHECKING, List, Optional, Sequence, Dict, Tuple
from ..entities import Arc
from .base import SketchChangeCommand

if TYPE_CHECKING:
    from ..constraints import Constraint, EqualDistanceConstraint
    from ..entities import Entity, Point
    from ..sketch import Sketch

logger = logging.getLogger(__name__)


class AddItemsCommand(SketchChangeCommand):
    """Command to add points, entities, and constraints to a sketch."""

    def __init__(
        self,
        sketch: "Sketch",
        name: str,
        points: Optional[Sequence["Point"]] = None,
        entities: Optional[Sequence["Entity"]] = None,
        constraints: Optional[Sequence["Constraint"]] = None,
    ):
        super().__init__(sketch, name)
        self.points = list(points) if points else []
        self.entities = list(entities) if entities else []
        self.constraints = list(constraints) if constraints else []

    def _do_execute(self) -> None:
        registry = self.sketch.registry
        new_points = []
        id_map: Dict[int, int] = {}  # Map old temp IDs to new final IDs

        for p in self.points:
            old_id = p.id
            # Assign a real ID if it's a temp ID (negative or >= counter)
            if p.id < 0 or p.id >= registry._id_counter:
                p.id = registry._id_counter
                registry._id_counter += 1
            if old_id != p.id:
                id_map[old_id] = p.id
            new_points.append(p)
        registry.points.extend(new_points)

        new_entities = []
        for e in self.entities:
            old_id = e.id
            if e.id < 0 or e.id >= registry._id_counter:
                e.id = registry._id_counter
                registry._id_counter += 1
            if old_id != e.id:
                id_map[old_id] = e.id

            # Update point references within the entity
            for attr, value in vars(e).items():
                if isinstance(value, int) and value in id_map:
                    setattr(e, attr, id_map[value])
            new_entities.append(e)
        registry.entities.extend(new_entities)

        # Update point and entity references within constraints
        for c in self.constraints:
            for attr, value in vars(c).items():
                if isinstance(value, int) and value in id_map:
                    setattr(c, attr, id_map[value])
                elif isinstance(value, list):
                    # Handle lists of IDs, like in EqualLengthConstraint
                    new_ids = [id_map.get(old, old) for old in value]
                    setattr(c, attr, new_ids)

        # Rebuild entity map after adding
        registry._entity_map = {e.id: e for e in registry.entities}
        self.sketch.constraints.extend(self.constraints)

    def _do_undo(self) -> None:
        registry = self.sketch.registry
        point_ids = {p.id for p in self.points}
        entity_ids = {e.id for e in self.entities}

        registry.points = [p for p in registry.points if p.id not in point_ids]
        registry.entities = [
            e for e in registry.entities if e.id not in entity_ids
        ]
        registry._entity_map = {e.id: e for e in registry.entities}
        for c in self.constraints:
            if c in self.sketch.constraints:
                self.sketch.constraints.remove(c)


class RemoveItemsCommand(SketchChangeCommand):
    """Command to remove points, entities, and constraints from a sketch."""

    def __init__(
        self,
        sketch: "Sketch",
        name: str,
        points: Optional[List["Point"]] = None,
        entities: Optional[Sequence["Entity"]] = None,
        constraints: Optional[List["Constraint"]] = None,
    ):
        super().__init__(sketch, name)
        self.points = points or []
        self.entities = list(entities) if entities else []
        self.constraints = constraints or []

    @staticmethod
    def calculate_dependencies(
        sketch: Sketch, selection
    ) -> Tuple[List["Point"], List["Entity"], List["Constraint"]]:
        """
        Calculates the full set of items to be deleted based on the current
        selection, including dependent items.
        """
        to_delete_constraints: List[Constraint] = []
        to_delete_entity_ids = set(selection.entity_ids)
        to_delete_point_ids = set(selection.point_ids)

        # 1. Selected Constraints
        if selection.constraint_idx is not None:
            if sketch.constraints and (
                0 <= selection.constraint_idx < len(sketch.constraints)
            ):
                to_delete_constraints.append(
                    sketch.constraints[selection.constraint_idx]
                )

        registry_entities = sketch.registry.entities or []
        entity_map = {e.id: e for e in registry_entities}

        # 2. Cascading: If points are deleted, find entities that use them
        for e in registry_entities:
            if e.id in to_delete_entity_ids:
                continue
            p_ids: List[int] = e.get_point_ids()
            # If any control point is marked for deletion, the entity must go
            if any(pid in to_delete_point_ids for pid in p_ids):
                to_delete_entity_ids.add(e.id)

        # 2.5. Cleanup Implicit Constraints for Deleted Entities (Arc geometry)
        # Arcs rely on EqualDistanceConstraint(c, s, c, e) not linked by ID.
        current_constraints = sketch.constraints or []
        for eid in to_delete_entity_ids:
            e = entity_map.get(eid)
            if isinstance(e, Arc):
                c, s, end = e.center_idx, e.start_idx, e.end_idx
                # Find constraints matching this geometry
                for constr in current_constraints:
                    if isinstance(constr, EqualDistanceConstraint):
                        set1 = {constr.p1, constr.p2}
                        set2 = {constr.p3, constr.p4}
                        target1, target2 = {c, s}, {c, end}
                        if (set1 == target1 and set2 == target2) or (
                            set1 == target2 and set2 == target1
                        ):
                            if constr not in to_delete_constraints:
                                to_delete_constraints.append(constr)

        # 3. Orphan Points
        if to_delete_entity_ids:
            used_points_by_remaining = set()
            points_of_deleted_entities = set()

            for e in registry_entities:
                p_ids = e.get_point_ids()
                if e.id in to_delete_entity_ids:
                    points_of_deleted_entities.update(p_ids)
                else:
                    used_points_by_remaining.update(p_ids)

            orphans = points_of_deleted_entities - used_points_by_remaining
            to_delete_point_ids.update(orphans)

        # 4. Cleanup Constraints (Dependencies)
        for constr in current_constraints:
            if constr in to_delete_constraints:
                continue
            if constr.depends_on_points(
                to_delete_point_ids
            ) or constr.depends_on_entities(to_delete_entity_ids):
                if constr not in to_delete_constraints:
                    to_delete_constraints.append(constr)

        # 5. Get actual objects from IDs
        final_points = [
            p
            for p in sketch.registry.points
            if p.id in to_delete_point_ids and not p.fixed
        ]
        final_entities = [
            e for e in registry_entities if e.id in to_delete_entity_ids
        ]

        return final_points, final_entities, to_delete_constraints

    def _do_execute(self) -> None:
        registry = self.sketch.registry
        point_ids = {p.id for p in self.points}
        entity_ids = {e.id for e in self.entities}

        registry.points = [p for p in registry.points if p.id not in point_ids]
        registry.entities = [
            e for e in registry.entities if e.id not in entity_ids
        ]
        registry._entity_map = {e.id: e for e in registry.entities}
        for c in self.constraints:
            if c in self.sketch.constraints:
                self.sketch.constraints.remove(c)

    def _do_undo(self) -> None:
        registry = self.sketch.registry
        registry.points.extend(self.points)
        registry.entities.extend(self.entities)
        registry._entity_map = {e.id: e for e in registry.entities}
        self.sketch.constraints.extend(self.constraints)
