from typing import List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from .registry import EntityRegistry
from blinker import Signal


class SketchSelection:
    """Manages the selection state of the sketch editor."""

    def __init__(self):
        self.point_ids: List[int] = []
        self.entity_ids: List[int] = []
        self.constraint_idx: Optional[int] = None
        self.junction_pid: Optional[int] = None
        self.changed = Signal()

    def clear(self):
        """Clears all selections."""
        self.point_ids.clear()
        self.entity_ids.clear()
        self.constraint_idx = None
        self.junction_pid = None
        self.changed.send(self)

    def copy(self) -> "SketchSelection":
        """Creates a shallow copy of the selection state."""
        new_sel = SketchSelection()
        new_sel.point_ids = self.point_ids[:]
        new_sel.entity_ids = self.entity_ids[:]
        new_sel.constraint_idx = self.constraint_idx
        new_sel.junction_pid = self.junction_pid
        return new_sel

    def is_empty(self) -> bool:
        """Returns True if nothing is selected."""
        return (
            not self.point_ids
            and not self.entity_ids
            and self.constraint_idx is None
            and self.junction_pid is None
        )

    def select_constraint(self, idx: int, is_multi: bool):
        """Selects a constraint by index."""
        self.constraint_idx = idx
        if not is_multi:
            self.point_ids.clear()
            self.entity_ids.clear()
            self.junction_pid = None
        self.changed.send(self)

    def select_junction(self, pid: int, is_multi: bool):
        """Selects an implicit junction point."""
        self.junction_pid = pid
        if not is_multi:
            self.point_ids.clear()
            self.entity_ids.clear()
            self.constraint_idx = None
        self.changed.send(self)

    def select_point(self, pid: int, is_multi: bool):
        """Selects a point by ID."""
        self._update_list(self.point_ids, pid, is_multi)
        if not is_multi:
            self.entity_ids.clear()
            self.constraint_idx = None
            self.junction_pid = None
        self.changed.send(self)

    def select_entity(self, entity, is_multi: bool):
        """Selects a geometric entity (Line or Arc)."""
        self._update_list(self.entity_ids, entity.id, is_multi)

        # Note: We do NOT select the control points here anymore.
        # This prevents deletion of an entity from cascading to its points
        # and destroying shared geometry.
        # Visual highlighting is handled by the renderer.

        if not is_multi:
            self.point_ids.clear()
            self.constraint_idx = None
            self.junction_pid = None
        self.changed.send(self)

    def _update_list(
        self, collection: List[int], item_id: int, is_multi: bool
    ):
        """Helper to handle toggle vs replace selection logic."""
        if is_multi:
            if item_id in collection:
                collection.remove(item_id)
            else:
                collection.append(item_id)
        else:
            if item_id not in collection:
                collection.clear()
                collection.append(item_id)

    def select_connected_entities(
        self, entity_id: int, registry: "EntityRegistry"
    ):
        """
        Adds all entities connected to the given entity through shared points
        to the current selection.

        Args:
            entity_id: The ID of the starting entity.
            registry: The EntityRegistry to query for connected entities.
        """
        connected_ids = registry.get_connected_entity_ids(entity_id)
        for eid in connected_ids:
            if eid not in self.entity_ids:
                self.entity_ids.append(eid)
        self.point_ids.clear()
        self.constraint_idx = None
        self.junction_pid = None
        self.changed.send(self)
