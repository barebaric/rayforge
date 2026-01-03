from __future__ import annotations
import logging
from typing import TYPE_CHECKING, Tuple, Optional, Dict

from .base import SketchChangeCommand
from ..entities import Line, Arc, Circle, Point

if TYPE_CHECKING:
    from ...undo.command import Command
    from ..sketch import Sketch

logger = logging.getLogger(__name__)


class MovePointCommand(SketchChangeCommand):
    """An undoable command for moving a sketch point, with coalescing."""

    def __init__(
        self,
        sketch: "Sketch",
        point_id: int,
        start_pos: Tuple[float, float],
        end_pos: Tuple[float, float],
        snapshot: Optional[Dict[int, Tuple[float, float]]] = None,
    ):
        super().__init__(sketch, _("Move Point"))
        self.point_id = point_id
        self.start_pos = start_pos
        self.end_pos = end_pos
        self._point_ref: Optional[Point] = None

        # If we are provided a snapshot (from the tool), use it.
        # This is critical because the drag operation changes coordinates
        # *before* the command is executed.
        if snapshot:
            self._state_snapshot = snapshot

    def _get_point(self) -> Optional["Point"]:
        """Gets a live reference to the point object."""
        # Check cache first
        if self._point_ref and self._point_ref.id == self.point_id:
            return self._point_ref
        # Find in registry if not cached or mismatched
        try:
            self._point_ref = self.sketch.registry.get_point(self.point_id)
            return self._point_ref
        except IndexError:
            return None

    def _do_execute(self) -> None:
        # Just ensure the specific point ends up where intended.
        # The base class's capture_snapshot logic handles the rest if needed.
        p = self._get_point()
        if p:
            p.x, p.y = self.end_pos

    def _do_undo(self) -> None:
        # Revert the specific point (though restore_snapshot does this for
        # all).
        p = self._get_point()
        if p:
            p.x, p.y = self.start_pos

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
        # Note: We do NOT update self._state_snapshot, because we want to
        # preserve the state from before the *first* move in the sequence.
        return True


class UnstickJunctionCommand(SketchChangeCommand):
    """Command to separate entities at a shared point."""

    def __init__(self, sketch: "Sketch", junction_pid: int):
        super().__init__(sketch, _("Unstick Junction"))
        self.junction_pid = junction_pid
        self.new_point: Optional[Point] = None
        # Stores {entity_id: (attribute_name, old_pid)}
        self.modified_map: Dict[int, Tuple[str, int]] = {}

    def _do_execute(self) -> None:
        try:
            junction_pt = self.sketch.registry.get_point(self.junction_pid)
        except IndexError:
            return

        entities_at_junction = []
        for e in self.sketch.registry.entities:
            if isinstance(e, Line):
                if self.junction_pid in [e.p1_idx, e.p2_idx]:
                    entities_at_junction.append(e)
            elif isinstance(e, Arc):
                if self.junction_pid in [
                    e.start_idx,
                    e.end_idx,
                    e.center_idx,
                ]:
                    entities_at_junction.append(e)
            elif isinstance(e, Circle):
                if self.junction_pid in [e.center_idx, e.radius_pt_idx]:
                    entities_at_junction.append(e)

        if len(entities_at_junction) < 2:
            return

        # Create a new point, add it to the registry, and store it
        new_pid = self.sketch.add_point(junction_pt.x, junction_pt.y)
        self.new_point = self.sketch.registry.get_point(new_pid)

        # Keep the first entity, modify the rest
        is_first = True
        for e in entities_at_junction:
            if is_first:
                is_first = False
                continue

            if isinstance(e, Line):
                if e.p1_idx == self.junction_pid:
                    self.modified_map[e.id] = ("p1_idx", e.p1_idx)
                    e.p1_idx = new_pid
                if e.p2_idx == self.junction_pid:
                    self.modified_map[e.id] = ("p2_idx", e.p2_idx)
                    e.p2_idx = new_pid
            elif isinstance(e, Arc):
                if e.start_idx == self.junction_pid:
                    self.modified_map[e.id] = ("start_idx", e.start_idx)
                    e.start_idx = new_pid
                if e.end_idx == self.junction_pid:
                    self.modified_map[e.id] = ("end_idx", e.end_idx)
                    e.end_idx = new_pid
                if e.center_idx == self.junction_pid:
                    self.modified_map[e.id] = ("center_idx", e.center_idx)
                    e.center_idx = new_pid
            elif isinstance(e, Circle):
                if e.center_idx == self.junction_pid:
                    self.modified_map[e.id] = ("center_idx", e.center_idx)
                    e.center_idx = new_pid
                if e.radius_pt_idx == self.junction_pid:
                    self.modified_map[e.id] = (
                        "radius_pt_idx",
                        e.radius_pt_idx,
                    )
                    e.radius_pt_idx = new_pid

    def _do_undo(self) -> None:
        # Revert changes to entities
        for eid, (attr, old_pid) in self.modified_map.items():
            e = self.sketch.registry.get_entity(eid)
            if e:
                setattr(e, attr, old_pid)

        # Remove the added point
        if self.new_point:
            registry = self.sketch.registry
            registry.points = [
                p for p in registry.points if p.id != self.new_point.id
            ]
