from __future__ import annotations

import copy
import logging
from gettext import gettext as _
from typing import TYPE_CHECKING

from ..entities import Ellipse, TextBoxEntity
from ..entities.point import Point
from ..entity_group import EntityGroup
from ..types import EntityID
from .base import SketchChangeCommand

if TYPE_CHECKING:
    from ..constraints import Constraint
    from ..entities import Entity
    from ..selection import SketchSelection
    from ..sketch import Sketch

logger = logging.getLogger(__name__)


def _allocate_id(registry) -> EntityID:
    new_id = registry._id_counter
    registry._id_counter += 1
    return new_id


def _remap_id_refs(obj: object, id_map: dict[EntityID, EntityID]) -> None:
    """Rewrites integer ID references on a copied point/entity/constraint."""
    for attr, value in vars(obj).items():
        if isinstance(value, bool):
            continue
        if isinstance(value, int) and value in id_map:
            setattr(obj, attr, id_map[value])
        elif isinstance(value, list) and attr.endswith("_ids"):
            setattr(obj, attr, [id_map.get(v, v) for v in value])


class DuplicateCommand(SketchChangeCommand):
    """
    Duplicates the current selection in-place.

    Selected entities and their points are copied with fresh IDs; internal
    constraints (all references within the selection) are copied and
    remapped to the duplicates as well. Constraints that reference
    geometry outside the selection are not copied. Fixed points (e.g. the
    origin) are duplicated as regular movable points so the copy can be
    moved independently of the original.
    """

    def __init__(self, sketch: Sketch, selection: SketchSelection):
        super().__init__(sketch, _("Duplicate Selection"))
        self._selection = selection.copy()
        self.created_points: list[Point] = []
        self.created_entities: list[Entity] = []
        self.created_constraints: list[Constraint] = []
        self.new_entity_ids: list[EntityID] = []
        self.new_point_ids: list[EntityID] = []

    @staticmethod
    def prepare(
        sketch: Sketch,
        selection: SketchSelection,
    ) -> (
        tuple[
            list[EntityID],
            set[EntityID],
            list[Constraint],
        ]
        | None
    ):
        """
        Pure function that resolves the duplicate operation parameters.

        Returns:
            A tuple of (entity_ids_to_duplicate,
            point_ids_to_duplicate, constraints_to_duplicate), or None
            if the selection contains no duplicatable geometry.
        """
        registry = sketch.registry

        # 1. Resolve entity set (including compound helpers)
        entity_id_set: set[EntityID] = set(selection.entity_ids)
        point_id_set: set[EntityID] = set(selection.point_ids)

        changed = True
        while changed:
            changed = False
            for e in registry.entities:
                helper_ids = None
                if isinstance(e, TextBoxEntity):
                    helper_ids = e.construction_line_ids
                elif isinstance(e, Ellipse):
                    helper_ids = e.helper_line_ids

                if helper_ids and (
                    e.id in entity_id_set
                    or not entity_id_set.isdisjoint(helper_ids)
                ):
                    for hid in helper_ids:
                        if hid not in entity_id_set:
                            entity_id_set.add(hid)
                            changed = True

        # 2. Resolve point set from entities
        point_id_set.update(
            EntityGroup(registry, sorted(entity_id_set)).point_ids()
        )

        if not entity_id_set and not point_id_set:
            return None

        # 3. Find internal constraints to duplicate. Constraints that
        #    reference geometry outside the selection are not copied.
        constraints_to_duplicate: list[Constraint] = []
        for constr in sketch.constraints:
            ref_points = constr.get_referenced_point_ids()
            ref_entities = constr.get_referenced_entity_ids()
            has_refs = bool(ref_points or ref_entities)
            is_internal = ref_points <= point_id_set and (
                ref_entities <= entity_id_set
            )
            if has_refs and is_internal:
                constraints_to_duplicate.append(constr)

        return sorted(entity_id_set), point_id_set, constraints_to_duplicate

    def _do_execute(self) -> None:
        result = self.prepare(self.sketch, self._selection)
        if result is None:
            return

        entity_ids, point_ids, constraints = result
        registry = self.sketch.registry
        id_map: dict[EntityID, EntityID] = {}

        for pid in sorted(point_ids):
            p = registry.get_point(pid)
            if not p:
                continue
            clone = Point.from_dict(p.to_dict())
            clone.fixed = False
            id_map[pid] = _allocate_id(registry)
            clone.id = id_map[pid]
            self.created_points.append(clone)

        for eid in entity_ids:
            e = registry.get_entity(eid)
            if not e:
                continue
            clone = copy.deepcopy(e)
            id_map[eid] = _allocate_id(registry)
            clone.id = id_map[eid]
            self.created_entities.append(clone)

        for constr in constraints:
            self.created_constraints.append(copy.deepcopy(constr))

        for obj in (*self.created_entities, *self.created_constraints):
            _remap_id_refs(obj, id_map)

        registry.points.extend(self.created_points)
        registry.entities.extend(self.created_entities)
        registry._entity_map = {e.id: e for e in registry.entities}
        self.sketch.constraints.extend(self.created_constraints)

        self.new_entity_ids = [e.id for e in self.created_entities]
        self.new_point_ids = [
            id_map[pid] for pid in self._selection.point_ids if pid in id_map
        ]

    def _do_undo(self) -> None:
        registry = self.sketch.registry
        point_ids = {p.id for p in self.created_points}
        entity_ids = {e.id for e in self.created_entities}

        registry.points = [p for p in registry.points if p.id not in point_ids]
        registry.entities = [
            e for e in registry.entities if e.id not in entity_ids
        ]
        registry._entity_map = {e.id: e for e in registry.entities}
        for c in self.created_constraints:
            if c in self.sketch.constraints:
                self.sketch.constraints.remove(c)
