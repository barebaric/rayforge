from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from .. import constraint_info
from .base import SketchChangeCommand

if TYPE_CHECKING:
    from raygeo.sketch.registry import EntityRegistry

    from ..sketch import Sketch
    from ..selection import SketchSelection

logger = logging.getLogger(__name__)


class AddItemsCommand(SketchChangeCommand):
    """Command that adds points, entities and constraints to the
    sketch.

    Items are added through the command's ``add_point``/``add_entity``
    /``add_constraint`` builder methods, which register them in the
    sketch immediately and record them, so undo removes them and redo
    restores them with their original ids."""

    def __init__(
        self,
        sketch: Sketch,
        name: str,
        constraints: list[Any] | None = None,
    ):
        super().__init__(sketch, name)
        self.point_ids: list[int] = []
        self.entity_ids: list[int] = []
        self.constraints: list[Any] = []
        self._point_dicts: list[dict[str, Any]] = []
        self._entity_dicts: list[dict[str, Any]] = []
        for constraint in constraints or []:
            self.add_constraint(constraint)

    def entities(self) -> list:
        return [
            self.sketch.registry.get_entity_object(eid)
            for eid in self.entity_ids
        ]

    def add_point(self, x: float, y: float, fixed: bool = False) -> int:
        pid = self.sketch.add_point(x, y, fixed)
        self.point_ids.append(pid)
        self._point_dicts.append(
            {"id": pid, "x": x, "y": y, "fixed": fixed}
        )
        return pid

    def add_entity(self, entity: Any) -> int:
        eid = self.sketch.registry.add_entity(entity)
        self.entity_ids.append(eid)
        self._entity_dicts.append(
            self.sketch.registry.entity_to_dict(eid)
        )
        return eid

    def add_entity_from_dict(self, entity_dict: dict[str, Any]) -> int:
        """Adds an entity from its reference dictionary, preserving
        its helper references and dictionary-only state (e.g. an
        ellipse's helper line ids, which plain entity construction
        cannot carry)."""
        from ..entity_ops import entity_from_dict

        registry = self.sketch.registry
        temp_id = registry.add_entity(entity_from_dict(entity_dict))
        final_dict = dict(entity_dict)
        final_dict["id"] = temp_id
        registry.remove_entities([temp_id])
        self.sketch.restore_structure([], [final_dict])
        self.entity_ids.append(temp_id)
        self._entity_dicts.append(final_dict)
        return temp_id

    def adopt_point(self, pid: int) -> None:
        """Records an already-existing point with the command, so undo
        removes it (used for preview points the command keeps)."""
        position = self.sketch.registry.get_point(pid)
        if position is None:
            return
        self.point_ids.append(pid)
        self._point_dicts.append(
            {
                "id": pid,
                "x": position[0],
                "y": position[1],
                "fixed": False,
            }
        )

    def add_constraint(self, constraint: Any) -> Any:
        if isinstance(constraint, dict):
            added = self.sketch.add_constraint_from_dict(constraint)
        else:
            added = self.sketch.add_constraint(constraint)
        self.constraints.append(added)
        return added

    def _do_execute(self) -> None:
        # Items were registered through the builder methods.
        pass

    def _do_undo(self) -> None:
        sketch = self.sketch
        for constraint in reversed(self.constraints):
            sketch.remove_constraint_object(constraint)
        sketch.registry.remove_entities(self.entity_ids)
        sketch.flags.drop_ids(set(self.entity_ids), set())
        sketch.registry.remove_points(list(self.point_ids))
        sketch.flags.drop_ids(set(), set(self.point_ids))

    def _redo(self) -> None:
        self.sketch.restore_structure(
            self._point_dicts, self._entity_dicts
        )


class RemoveItemsCommand(SketchChangeCommand):
    """Command that removes points, entities and constraints from the
    sketch. Removed items are serialized before removal so undo can
    restore them with their original ids."""

    def __init__(
        self,
        sketch: Sketch,
        name: str,
        points: list[dict[str, Any]] | None = None,
        entity_ids: list[int] | None = None,
        constraints: list[Any] | None = None,
        entities: list[int] | None = None,
    ):
        super().__init__(sketch, name)
        self.point_dicts: list[dict[str, Any]] = points or []
        self.entity_ids: list[int] = entity_ids or entities or []
        self.constraints: list[Any] = constraints or []
        self._entity_dicts: list[dict[str, Any]] = []
        self._point_dicts: list[dict[str, Any]] = []

    @staticmethod
    def calculate_dependencies(
        sketch: Sketch, selection: SketchSelection
    ) -> tuple[list[dict[str, Any]], list[int], list[Any]]:
        """
        Calculates the full set of items to be deleted based on the current
        selection, including dependent items. Returns (point_dicts,
        entity_ids, constraints).
        """
        return RemoveItemsCommand.calculate_dependencies_for_ids(
            sketch, set(selection.entity_ids), set(selection.point_ids)
        )

    @classmethod
    def calculate_dependencies_for_ids(
        cls,
        sketch: Sketch,
        entity_ids: set[int],
        point_ids: set[int] | None = None,
    ) -> tuple[list[dict[str, Any]], list[int], list[Any]]:
        """
        Calculates the removal dependencies for a plain set of entity
        (and optionally point) IDs, without going through a selection.
        """
        registry = sketch.registry
        to_delete_constraints: list[Any] = []
        to_delete_entity_ids = set(entity_ids)
        to_delete_point_ids = set(point_ids or set())

        # Iteratively find all dependencies until no new items are added
        while True:
            num_points_before = len(to_delete_point_ids)
            num_entities_before = len(to_delete_entity_ids)

            # A. Unity Logic for compound objects like text boxes
            for eid in registry.entity_ids():
                helper_ids = registry.helper_ids(eid)
                if not helper_ids:
                    continue
                is_part_of_delete_set = (
                    eid in to_delete_entity_ids
                    or not to_delete_entity_ids.isdisjoint(helper_ids)
                    or not to_delete_point_ids.isdisjoint(
                        registry.entity_point_ids(eid) or []
                    )
                )
                if is_part_of_delete_set:
                    to_delete_entity_ids.add(eid)
                    to_delete_entity_ids.update(helper_ids)
                    to_delete_point_ids.update(
                        registry.entity_point_ids(eid) or []
                    )

            # B. Cascading: If points are deleted, find entities that use them
            for eid in registry.entity_ids():
                if eid in to_delete_entity_ids:
                    continue
                p_ids: list[int] = registry.entity_point_ids(eid) or []
                if any(pid in to_delete_point_ids for pid in p_ids):
                    to_delete_entity_ids.add(eid)

            # C. Orphan Points
            if to_delete_entity_ids:
                used_points_by_remaining = set()
                points_of_deleted_entities = set()

                for eid in registry.entity_ids():
                    p_ids = registry.entity_point_ids(eid) or []
                    if eid in to_delete_entity_ids:
                        points_of_deleted_entities.update(p_ids)
                    else:
                        used_points_by_remaining.update(p_ids)

                orphans = (
                    points_of_deleted_entities - used_points_by_remaining
                )
                to_delete_point_ids.update(orphans)

            if (
                len(to_delete_point_ids) == num_points_before
                and len(to_delete_entity_ids) == num_entities_before
            ):
                break  # Stable state reached

        # 2.5. Cleanup Implicit Constraints for Deleted Entities
        # (equal-distance arc geometry pairs)
        for eid in to_delete_entity_ids:
            if registry.entity_type(eid) != "arc":
                continue
            entity = registry.get_entity_object(eid)
            if entity is None:
                continue
            c, s, end = (
                entity.center_idx,
                entity.start_idx,
                entity.end_idx,
            )
            from raygeo.sketch.constraints.equal_distance import (
                EqualDistanceConstraint,
            )

            for constr in sketch.constraints:
                if not isinstance(constr, EqualDistanceConstraint):
                    continue
                set1 = {constr.p1, constr.p2}
                set2 = {constr.p3, constr.p4}
                target1, target2 = {c, s}, {c, end}
                if (
                    (set1 == target1 and set2 == target2)
                    or (set1 == target2 and set2 == target1)
                ) and constr not in to_delete_constraints:
                    to_delete_constraints.append(constr)

        # 4. Cleanup Constraints (Dependencies)
        for constr in sketch.constraints:
            if constr in to_delete_constraints:
                continue
            if (
                constraint_info.depends_on_points(
                    constr, to_delete_point_ids
                )
                or constraint_info.depends_on_entities(
                    sketch.registry, constr, to_delete_entity_ids
                )
            ) and constr not in to_delete_constraints:
                to_delete_constraints.append(constr)

        # 4.5. Orphan Points Owned Only by Deleted Constraints
        # A point may be referenced solely by constraints (e.g. a
        # rectangle's auto-created center point) and belong to no
        # entity. Step C above only orphans points that belonged to
        # deleted entities, so handle constraint-only points here.
        to_delete_constraint_set = {id(c) for c in to_delete_constraints}
        points_of_deleted_constraints: set[int] = set()
        used_points_by_surviving: set[int] = set()

        for constr in sketch.constraints:
            pids = constr.referenced_points()
            if id(constr) in to_delete_constraint_set:
                points_of_deleted_constraints.update(pids)
            else:
                used_points_by_surviving.update(pids)

        for eid in registry.entity_ids():
            if eid not in to_delete_entity_ids:
                used_points_by_surviving.update(
                    registry.entity_point_ids(eid) or []
                )

        constraint_orphans = (
            points_of_deleted_constraints
            - used_points_by_surviving
            - to_delete_point_ids
        )
        to_delete_point_ids.update(constraint_orphans)

        # 5. Serialize the actual items. Fixed points are included
        # (array copies are fixed and must be deletable) except the
        # sketch origin, which survives entity cascades.
        point_dicts = []
        for point in registry.points():
            if point.id not in to_delete_point_ids:
                continue
            if point.fixed and point.id == sketch.origin_id:
                continue
            point_dicts.append(
                {
                    "id": point.id,
                    "x": point.x,
                    "y": point.y,
                    "fixed": point.fixed,
                }
            )
        return point_dicts, sorted(to_delete_entity_ids), to_delete_constraints

    def apply_direct(self) -> None:
        """
        Removes this command's items from the sketch directly,
        bypassing execute so prune_arrays() never sees arrays mid-edit
        (e.g. while an array command rebuilds its members).
        """
        sketch = self.sketch
        registry = sketch.registry
        if not self._entity_dicts:
            self._entity_dicts = [
                registry.entity_to_dict(eid)
                for eid in self.entity_ids
                if registry.get_entity_object(eid) is not None
            ]
        for constraint in self.constraints:
            sketch.remove_constraint_object(constraint)
        registry.remove_entities(self.entity_ids)
        sketch.flags.drop_ids(set(self.entity_ids), set())
        registry.remove_points([d["id"] for d in self.point_dicts])
        sketch.flags.drop_ids(
            set(), {d["id"] for d in self.point_dicts}
        )

    def _do_execute(self) -> None:
        self.apply_direct()
        self.sketch.prune_arrays()

    def _do_undo(self) -> None:
        sketch = self.sketch
        sketch.restore_structure(self.point_dicts, self._entity_dicts)
        for constraint in self.constraints:
            sketch.add_constraint(constraint)

    def _redo(self) -> None:
        self.apply_direct()

    @classmethod
    def calculate_dependencies_for_selection(
        cls,
        sketch: Sketch,
        selection: SketchSelection,
    ) -> tuple[list[dict[str, Any]], list[int], list[Any]]:
        """Removal dependencies including a selected constraint."""
        point_dicts, entity_ids, constraints = (
            cls.calculate_dependencies_for_ids(
                sketch, set(selection.entity_ids), set(selection.point_ids)
            )
        )
        if (
            selection.constraint_idx is not None
            and sketch.constraints
            and 0 <= selection.constraint_idx < len(sketch.constraints)
        ):
            selected = sketch.constraints[selection.constraint_idx]
            if selected not in constraints:
                constraints.append(selected)
        return point_dicts, entity_ids, constraints


