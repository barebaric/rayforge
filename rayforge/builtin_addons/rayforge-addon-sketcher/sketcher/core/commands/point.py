from __future__ import annotations

import logging
from gettext import gettext as _
from typing import TYPE_CHECKING

from raygeo.geo.types import Point as GeoPoint

from ..types import EntityID
from .base import SketchChangeCommand

if TYPE_CHECKING:
    from raygeo.sketch.registry import EntityRegistry

    from rayforge.core.undo.command import Command

    from ..sketch import Sketch
    from ..types import Constraint

logger = logging.getLogger(__name__)


class MovePointCommand(SketchChangeCommand):
    """An undoable command for moving a sketch point, with coalescing."""

    def __init__(
        self,
        sketch: Sketch,
        point_id: EntityID,
        start_pos: GeoPoint,
        end_pos: GeoPoint,
        snapshot: dict | None = None,
        snap_constraints: list[Constraint] | None = None,
    ):
        super().__init__(sketch, _("Move Point"))
        self.point_id = point_id
        self.start_pos = start_pos
        self.end_pos = end_pos
        self._snap_constraints: list[Constraint] = snap_constraints or []
        self._created_constraints: list[Constraint] = []

        # If we are provided with a snapshot (from the tool), use it.
        # This is critical because the drag operation changes coordinates
        # *before* the command is executed.
        if snapshot:
            self._snapshot = snapshot

    def _set_position(self, position: GeoPoint) -> None:
        self.sketch.registry.set_point_position(
            self.point_id, position[0], position[1]
        )

    def _do_execute(self) -> None:
        # Just ensure the specific point ends up where intended.
        # The base class's capture_snapshot logic handles the rest if needed.
        if self.sketch.registry.get_point(self.point_id) is not None:
            self._set_position(self.end_pos)

        for constraint in self._snap_constraints:
            self.sketch.add_constraint(constraint)
            self._created_constraints.append(constraint)

    def _do_undo(self) -> None:
        # Revert the specific point (though restore_snapshot does this for
        # all).
        if self.sketch.registry.get_point(self.point_id) is not None:
            self._set_position(self.start_pos)

        for constraint in self._created_constraints:
            self.sketch.remove_constraint_object(constraint)
        self._created_constraints.clear()

    def can_coalesce_with(self, next_command: Command) -> bool:
        return (
            isinstance(next_command, MovePointCommand)
            and self.point_id == next_command.point_id
        )

    def coalesce_with(self, next_command: Command) -> bool:
        if not self.can_coalesce_with(next_command):
            return False

        # Update our end position to the newest position
        self.end_pos = next_command.end_pos  # type: ignore
        self.timestamp = next_command.timestamp
        # We do NOT update self._snapshot; we keep the state from before
        # the FIRST move.
        return True


class MoveControlPointCommand(SketchChangeCommand):
    """An undoable command for moving a control point offset."""

    def __init__(
        self,
        sketch: Sketch,
        bezier_id: EntityID,
        cp_index: int,
        start_offset: GeoPoint | None,
        end_offset: GeoPoint | None,
        snapshot: dict | None = None,
    ):
        label = _("Move Control Point")
        super().__init__(sketch, label)
        self.bezier_id = bezier_id
        self.cp_index = cp_index
        self.start_offset = start_offset
        self.end_offset = end_offset
        # If we are provided with a snapshot (from the tool), use it.
        # This is critical because the drag operation changes coordinates
        # *before* the command is executed.
        if snapshot:
            self._snapshot = snapshot

    def _apply_offset(self, registry: EntityRegistry, offset) -> None:
        key = "cp1" if self.cp_index == 1 else "cp2"
        if offset is None:
            value = [0.0, 0.0]
        else:
            value = [offset[0], offset[1]]
        registry.set_state(self.bezier_id, {key: value})

    def _do_execute(self) -> None:
        self._apply_offset(self.sketch.registry, self.end_offset)

    def _do_undo(self) -> None:
        self._apply_offset(self.sketch.registry, self.start_offset)


class UnstickJunctionCommand(SketchChangeCommand):
    """Command to separate entities at a shared point."""

    def __init__(self, sketch: Sketch, junction_pid: EntityID):
        super().__init__(sketch, _("Unstick Junction"))
        self.junction_pid = junction_pid
        self.new_pid: EntityID | None = None
        # The entities moved onto the new point.
        self.modified_entity_ids: set[EntityID] = set()

    def _do_execute(self) -> None:
        registry = self.sketch.registry
        junction_pt = registry.get_point(self.junction_pid)
        if junction_pt is None:
            return

        junction_types = registry.junction_point_ids
        entities_at_junction = []
        for eid in registry.entity_ids():
            if self.junction_pid in junction_types(eid):
                entities_at_junction.append(eid)

        if len(entities_at_junction) < 2:
            return

        # Create a new point, add it to the registry, and store it
        new_pid = self.sketch.add_point(junction_pt[0], junction_pt[1])
        self.new_pid = new_pid

        # Keep the first entity on the junction, move the rest onto
        # the new point.
        is_first = True
        for eid in entities_at_junction:
            if is_first:
                is_first = False
                continue
            self.modified_entity_ids.add(eid)
        if self.modified_entity_ids:
            self.sketch.remap_entity_point_refs(
                {self.junction_pid: new_pid},
                entity_ids=self.modified_entity_ids,
            )

    def _do_undo(self) -> None:
        # Revert changes to entities
        if self.new_pid is not None and self.modified_entity_ids:
            self.sketch.remap_entity_point_refs(
                {self.new_pid: self.junction_pid},
                entity_ids=self.modified_entity_ids,
            )

        # Remove the added point
        if self.new_pid is not None:
            self.sketch.registry.remove_points([self.new_pid])


class MoveEntitiesCommand(SketchChangeCommand):
    """An undoable command for moving one or more entities as a group."""

    def __init__(
        self,
        sketch: Sketch,
        entity_ids: list[int],
        start_positions: dict[EntityID, GeoPoint],
        end_positions: dict[EntityID, GeoPoint],
        snapshot: dict | None = None,
    ):
        super().__init__(sketch, _("Move Entities"))
        self.entity_ids = list(entity_ids)
        self.start_positions = dict(start_positions)
        self.end_positions = dict(end_positions)
        if snapshot:
            self._snapshot = snapshot

    def _apply_positions(self, positions: dict[EntityID, GeoPoint]):
        registry = self.sketch.registry
        for pid, (x, y) in positions.items():
            if registry.get_point(pid) is not None:
                registry.set_point_position(pid, x, y)

    def _do_execute(self) -> None:
        self._apply_positions(self.end_positions)

    def _do_undo(self) -> None:
        self._apply_positions(self.start_positions)

    def can_coalesce_with(self, next_command: Command) -> bool:
        return (
            isinstance(next_command, MoveEntitiesCommand)
            and self.entity_ids == next_command.entity_ids
        )

    def coalesce_with(self, next_command: Command) -> bool:
        if not self.can_coalesce_with(next_command):
            return False
        assert isinstance(next_command, MoveEntitiesCommand)
        self.end_positions = next_command.end_positions
        self.timestamp = next_command.timestamp
        return True
