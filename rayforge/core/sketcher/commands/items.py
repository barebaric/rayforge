from __future__ import annotations
import logging
from typing import TYPE_CHECKING, List, Optional, Sequence, Dict

from .base import SketchChangeCommand

if TYPE_CHECKING:
    from ..entities import Entity, Point
    from ..constraints import Constraint
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
