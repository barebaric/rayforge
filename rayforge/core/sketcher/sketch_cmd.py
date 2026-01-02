from __future__ import annotations
import logging
import uuid
from typing import TYPE_CHECKING, List, Tuple, Optional, Dict
from ..undo.command import Command
from .entities import Line, Arc, Circle
from .sketch import Fill, Sketch


if TYPE_CHECKING:
    from .entities import Point, Entity
    from .constraints import Constraint

logger = logging.getLogger(__name__)


class SketchChangeCommand(Command):
    """
    Base class for commands that modify a sketch and need to trigger a solve.
    Includes functionality to snapshot geometry state for precise undo.
    """

    def __init__(self, sketch: "Sketch", name: str):
        super().__init__(name)
        self.sketch = sketch
        # Stores {point_id: (x, y)} for all points in the sketch
        self._state_snapshot: Dict[int, Tuple[float, float]] = {}

    def capture_snapshot(self):
        """Captures the current coordinates of all points."""
        self._state_snapshot = {
            p.id: (p.x, p.y) for p in self.sketch.registry.points
        }

    def restore_snapshot(self):
        """Restores coordinates from the snapshot."""
        registry = self.sketch.registry
        for pid, (x, y) in self._state_snapshot.items():
            try:
                p = registry.get_point(pid)
                p.x = x
                p.y = y
            except IndexError:
                # Point might not exist if _do_undo didn't restore it,
                # but typically this shouldn't happen.
                pass

    def execute(self) -> None:
        # If a snapshot wasn't provided during initialization (e.g. by a tool
        # that moved points before creating the command), capture it now.
        if not self._state_snapshot:
            self.capture_snapshot()

        self._do_execute()
        self.sketch.notify_update()

    def undo(self) -> None:
        self._do_undo()
        # Restore the exact geometric positions from before the command.
        # This prevents the solver from jumping to an alternative solution
        # (e.g., triangle flip) when constraints are reapplied.
        self.restore_snapshot()
        self.sketch.notify_update()

    def _do_execute(self) -> None:
        raise NotImplementedError

    def _do_undo(self) -> None:
        raise NotImplementedError


class AddItemsCommand(SketchChangeCommand):
    """Command to add points, entities, and constraints to a sketch."""

    def __init__(
        self,
        sketch: "Sketch",
        name: str,
        points: Optional[List["Point"]] = None,
        entities: Optional[List["Entity"]] = None,
        constraints: Optional[List["Constraint"]] = None,
    ):
        super().__init__(sketch, name)
        self.points = points or []
        self.entities = entities or []
        self.constraints = constraints or []

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
        entities: Optional[List["Entity"]] = None,
        constraints: Optional[List["Constraint"]] = None,
    ):
        super().__init__(sketch, name)
        self.points = points or []
        self.entities = entities or []
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


class ModifyConstraintCommand(SketchChangeCommand):
    """
    Command to modify the value or expression of a constraint.
    """

    def __init__(
        self,
        sketch: "Sketch",
        constraint: "Constraint",
        new_value: float,
        new_expression: Optional[str] = None,
        name: str = _("Edit Constraint"),
    ):
        super().__init__(sketch, name)
        self.constraint = constraint
        self.new_value = float(new_value)
        self.new_expression = new_expression

        self.old_value = float(constraint.value)
        self.old_expression = getattr(constraint, "expression", None)

    def _do_execute(self) -> None:
        self.constraint.value = self.new_value
        self.constraint.expression = self.new_expression

    def _do_undo(self) -> None:
        self.constraint.value = self.old_value
        self.constraint.expression = self.old_expression


class ToggleConstructionCommand(SketchChangeCommand):
    """Command to toggle the construction state of multiple entities."""

    def __init__(self, sketch: "Sketch", name: str, entity_ids: List[int]):
        super().__init__(sketch, name)
        self.entity_ids = entity_ids
        self.original_states: Dict[int, bool] = {}
        self.new_state: Optional[bool] = None

    def _do_execute(self) -> None:
        self.original_states.clear()
        entities_to_modify = []
        for eid in self.entity_ids:
            ent = self.sketch.registry.get_entity(eid)
            if ent:
                entities_to_modify.append(ent)
                self.original_states[eid] = ent.construction

        if not entities_to_modify:
            return

        # Logic: If any selected entity is NOT construction, set all to
        # construction.
        # Otherwise (all are construction), set all to normal.
        if self.new_state is None:
            has_normal = any(not e.construction for e in entities_to_modify)
            self.new_state = has_normal

        for e in entities_to_modify:
            e.construction = self.new_state

    def _do_undo(self) -> None:
        for eid, old_state in self.original_states.items():
            ent = self.sketch.registry.get_entity(eid)
            if ent:
                ent.construction = old_state


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


class AddFillCommand(SketchChangeCommand):
    """Command to add a Fill to a sketch."""

    def __init__(
        self,
        sketch: "Sketch",
        boundary: List[Tuple[int, bool]],
        name: str = _("Add Fill"),
    ):
        super().__init__(sketch, name)
        self.fill: Optional[Fill] = None
        self._boundary = boundary

    def _do_execute(self) -> None:
        if self.fill is None:
            self.fill = Fill(uid=str(uuid.uuid4()), boundary=self._boundary)
        self.sketch.fills.append(self.fill)

    def _do_undo(self) -> None:
        if self.fill and self.fill in self.sketch.fills:
            self.sketch.fills.remove(self.fill)


class RemoveFillCommand(SketchChangeCommand):
    """Command to remove a Fill from a sketch."""

    def __init__(
        self,
        sketch: "Sketch",
        fill: Fill,
        name: str = _("Remove Fill"),
    ):
        super().__init__(sketch, name)
        self.fill = fill

    def _do_execute(self) -> None:
        if self.fill in self.sketch.fills:
            self.sketch.fills.remove(self.fill)

    def _do_undo(self) -> None:
        self.sketch.fills.append(self.fill)
