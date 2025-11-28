from __future__ import annotations
import logging
from typing import TYPE_CHECKING, List, Tuple, Optional, Dict
from rayforge.undo.models.command import Command
from rayforge.undo.models.property_cmd import ChangePropertyCommand
from rayforge.core.sketcher.entities import Line, Arc, Circle

if TYPE_CHECKING:
    from rayforge.core.sketcher.entities import Point, Entity
    from rayforge.core.sketcher.constraints import Constraint
    from .sketchelement import SketchElement

logger = logging.getLogger(__name__)


class SketchChangeCommand(Command):
    """
    Base class for commands that modify a sketch and need to trigger a solve.
    """

    def __init__(self, element: "SketchElement", name: str):
        super().__init__(name)
        self.element = element

    def _solve_and_update(self):
        """Helper to run the solver and update element bounds."""
        self.element.sketch.solve()
        self.element.update_bounds_from_sketch()
        self.element.mark_dirty()

    def execute(self) -> None:
        self._do_execute()
        self._solve_and_update()

    def undo(self) -> None:
        self._do_undo()
        self._solve_and_update()

    def _do_execute(self) -> None:
        raise NotImplementedError

    def _do_undo(self) -> None:
        raise NotImplementedError


class AddItemsCommand(SketchChangeCommand):
    """Command to add points, entities, and constraints to a sketch."""

    def __init__(
        self,
        element: "SketchElement",
        name: str,
        points: Optional[List["Point"]] = None,
        entities: Optional[List["Entity"]] = None,
        constraints: Optional[List["Constraint"]] = None,
    ):
        super().__init__(element, name)
        self.points = points or []
        self.entities = entities or []
        self.constraints = constraints or []

    def _do_execute(self) -> None:
        registry = self.element.sketch.registry
        new_points = []
        for p in self.points:
            # Assign a real ID if it's a new point
            if p.id >= registry._id_counter:
                p.id = registry._id_counter
                registry._id_counter += 1
            new_points.append(p)
        registry.points.extend(new_points)

        new_entities = []
        for e in self.entities:
            if e.id >= registry._id_counter:
                e.id = registry._id_counter
                registry._id_counter += 1
            new_entities.append(e)
        registry.entities.extend(new_entities)

        # Rebuild entity map after adding
        registry._entity_map = {e.id: e for e in registry.entities}
        self.element.sketch.constraints.extend(self.constraints)

    def _do_undo(self) -> None:
        registry = self.element.sketch.registry
        point_ids = {p.id for p in self.points}
        entity_ids = {e.id for e in self.entities}

        registry.points = [p for p in registry.points if p.id not in point_ids]
        registry.entities = [
            e for e in registry.entities if e.id not in entity_ids
        ]
        registry._entity_map = {e.id: e for e in registry.entities}
        for c in self.constraints:
            if c in self.element.sketch.constraints:
                self.element.sketch.constraints.remove(c)


class RemoveItemsCommand(SketchChangeCommand):
    """Command to remove points, entities, and constraints from a sketch."""

    def __init__(
        self,
        element: "SketchElement",
        name: str,
        points: Optional[List["Point"]] = None,
        entities: Optional[List["Entity"]] = None,
        constraints: Optional[List["Constraint"]] = None,
    ):
        super().__init__(element, name)
        self.points = points or []
        self.entities = entities or []
        self.constraints = constraints or []

    def _do_execute(self) -> None:
        registry = self.element.sketch.registry
        point_ids = {p.id for p in self.points}
        entity_ids = {e.id for e in self.entities}

        registry.points = [p for p in registry.points if p.id not in point_ids]
        registry.entities = [
            e for e in registry.entities if e.id not in entity_ids
        ]
        registry._entity_map = {e.id: e for e in registry.entities}
        for c in self.constraints:
            if c in self.element.sketch.constraints:
                self.element.sketch.constraints.remove(c)

    def _do_undo(self) -> None:
        registry = self.element.sketch.registry
        registry.points.extend(self.points)
        registry.entities.extend(self.entities)
        registry._entity_map = {e.id: e for e in registry.entities}
        self.element.sketch.constraints.extend(self.constraints)


class MovePointCommand(SketchChangeCommand):
    """An undoable command for moving a sketch point, with coalescing."""

    def __init__(
        self,
        element: "SketchElement",
        point_id: int,
        start_pos: Tuple[float, float],
        end_pos: Tuple[float, float],
    ):
        super().__init__(element, _("Move Point"))
        self.point_id = point_id
        self.start_pos = start_pos
        self.end_pos = end_pos
        self._point_ref: Optional[Point] = None

    def _get_point(self) -> Optional["Point"]:
        """Gets a live reference to the point object."""
        # Check cache first
        if self._point_ref and self._point_ref.id == self.point_id:
            return self._point_ref
        # Find in registry if not cached or mismatched
        try:
            self._point_ref = self.element.sketch.registry.get_point(
                self.point_id
            )
            return self._point_ref
        except IndexError:
            return None

    def _do_execute(self) -> None:
        p = self._get_point()
        if p:
            p.x, p.y = self.end_pos

    def _do_undo(self) -> None:
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
        return True


class ModifyConstraintValueCommand(SketchChangeCommand):
    """Command to modify the value of a constraint and re-solve."""

    def __init__(
        self,
        element: "SketchElement",
        constraint: "Constraint",
        new_value: float,
        name: str = _("Edit Constraint"),
    ):
        super().__init__(element, name)
        # We wrap a ChangePropertyCommand to handle the value change
        self.prop_cmd = ChangePropertyCommand(
            target=constraint, property_name="value", new_value=new_value
        )

    def _do_execute(self) -> None:
        self.prop_cmd.execute()

    def _do_undo(self) -> None:
        self.prop_cmd.undo()


class ToggleConstructionCommand(SketchChangeCommand):
    """Command to toggle the construction state of multiple entities."""

    def __init__(
        self, element: "SketchElement", name: str, entity_ids: List[int]
    ):
        super().__init__(element, name)
        self.entity_ids = entity_ids
        self.original_states: Dict[int, bool] = {}
        self.new_state: Optional[bool] = None

    def _do_execute(self) -> None:
        self.original_states.clear()
        entities_to_modify = []
        for eid in self.entity_ids:
            ent = self.element.sketch.registry.get_entity(eid)
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
            ent = self.element.sketch.registry.get_entity(eid)
            if ent:
                ent.construction = old_state


class UnstickJunctionCommand(SketchChangeCommand):
    """Command to separate entities at a shared point."""

    def __init__(self, element: "SketchElement", junction_pid: int):
        super().__init__(element, _("Unstick Junction"))
        self.junction_pid = junction_pid
        self.new_point: Optional[Point] = None
        # Stores {entity_id: (attribute_name, old_pid)}
        self.modified_map: Dict[int, Tuple[str, int]] = {}

    def _do_execute(self) -> None:
        try:
            junction_pt = self.element.sketch.registry.get_point(
                self.junction_pid
            )
        except IndexError:
            return

        entities_at_junction = []
        for e in self.element.sketch.registry.entities:
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
        new_pid = self.element.sketch.add_point(junction_pt.x, junction_pt.y)
        self.new_point = self.element.sketch.registry.get_point(new_pid)

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
            e = self.element.sketch.registry.get_entity(eid)
            if e:
                setattr(e, attr, old_pid)

        # Remove the added point
        if self.new_point:
            registry = self.element.sketch.registry
            registry.points = [
                p for p in registry.points if p.id != self.new_point.id
            ]
