from __future__ import annotations

import logging
from gettext import gettext as _
from typing import TYPE_CHECKING, Any

from .. import constraint_info
from ..constraint_ops import rebuild
from ..entity_group import EntityGroup
from ..entity_ops import clone_entity
from ..types import EntityID
from .base import SketchChangeCommand
from .items import AddItemsCommand

if TYPE_CHECKING:
    from ..selection import SketchSelection
    from ..sketch import Sketch

logger = logging.getLogger(__name__)


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
        self.add_cmd: AddItemsCommand | None = None
        self.created_constraints: list[Any] = []
        self.new_entity_ids: list[EntityID] = []
        self.new_point_ids: list[EntityID] = []
        self._entity_id_map: dict[EntityID, EntityID] = {}
        self._point_id_map: dict[EntityID, EntityID] = {}

    @property
    def created_points(self) -> list[EntityID]:
        """The point ids created by the duplicate."""
        return self.new_point_ids

    @property
    def created_entities(self) -> list[EntityID]:
        """The entity ids created by the duplicate."""
        return self.new_entity_ids

    @staticmethod
    def prepare(
        sketch: Sketch,
        selection: SketchSelection,
    ) -> (
        tuple[
            list[EntityID],
            set[EntityID],
            list[Any],
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
            for eid in registry.entity_ids():
                helper_ids = registry.helper_ids(eid)

                if helper_ids and (
                    eid in entity_id_set
                    or not entity_id_set.isdisjoint(helper_ids)
                ):
                    for hid in helper_ids:
                        if hid not in entity_id_set:
                            entity_id_set.add(hid)
                            changed = True

        # 2. Resolve point set from entities
        point_id_set.update(
            EntityGroup(
                registry, sorted(entity_id_set), flags=sketch.flags
            ).point_ids()
        )

        if not entity_id_set and not point_id_set:
            return None

        # 3. Find internal constraints to duplicate. Constraints that
        #    reference geometry outside the selection are not copied.
        constraints_to_duplicate: list[Any] = []
        for constr in sketch.constraints:
            ref_points = constr.get_referenced_point_ids()
            ref_entities = constraint_info.referenced_entity_ids(
                registry, constr
            )
            has_refs = bool(ref_points or ref_entities)
            is_internal = ref_points <= point_id_set and (
                ref_entities <= entity_id_set
            )
            if has_refs and is_internal:
                constraints_to_duplicate.append(constr)

        return sorted(entity_id_set), point_id_set, constraints_to_duplicate

    def _do_execute(self) -> None:
        if self.add_cmd:
            return self.add_cmd._redo()

        result = self.prepare(self.sketch, self._selection)
        if result is None:
            return

        entity_ids, point_ids, constraints = result
        registry = self.sketch.registry
        add_cmd = AddItemsCommand(self.sketch, "")
        point_map: dict[EntityID, EntityID] = {}

        for pid in sorted(point_ids):
            position = registry.get_point(pid)
            if position is None:
                continue
            # Copies are regular movable points so the duplicate can be
            # moved independently of the original.
            point_map[pid] = add_cmd.add_point(*position)

        entity_map: dict[EntityID, EntityID] = {}
        for eid in entity_ids:
            entity = registry.get_entity_object(eid)
            if entity is None:
                continue
            clone = clone_entity(registry, entity, point_map)
            clone_dict = registry.entity_to_dict(registry.add_entity(clone))
            registry.remove_entities([clone_dict["id"]])
            helper_ids = registry.helper_ids(eid)
            if helper_ids:
                mapped_helpers = [
                    entity_map[hid] for hid in helper_ids if hid in entity_map
                ]
                if clone_dict.get("type") == "ellipse":
                    clone_dict["helper_line_ids"] = mapped_helpers
                elif clone_dict.get("type") == "text_box":
                    clone_dict["construction_line_ids"] = mapped_helpers
            entity_map[eid] = add_cmd.add_entity_from_dict(clone_dict)
            self.sketch.copy_entity_flags(eid, entity_map[eid])

        for constr in constraints:
            self.created_constraints.append(rebuild(constr, point_map))
            add_cmd.add_constraint(self.created_constraints[-1])

        self.add_cmd = add_cmd
        self._entity_id_map = entity_map
        self._point_id_map = point_map
        self.new_entity_ids = [
            entity_map[eid] for eid in entity_ids if eid in entity_map
        ]
        self.new_point_ids = list(point_map.values())

    def _do_undo(self) -> None:
        if self.add_cmd:
            self.add_cmd._do_undo()
        self.created_constraints.clear()
