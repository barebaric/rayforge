from typing import List, Optional


class SketchSelection:
    """Manages the selection state of the sketch editor."""

    def __init__(self):
        self.point_ids: List[int] = []
        self.entity_ids: List[int] = []
        self.constraint_idx: Optional[int] = None
        self.junction_pid: Optional[int] = None

    def clear(self):
        """Clears all selections."""
        self.point_ids.clear()
        self.entity_ids.clear()
        self.constraint_idx = None
        self.junction_pid = None

    def copy(self) -> "SketchSelection":
        """Creates a shallow copy of the selection state."""
        new_sel = SketchSelection()
        new_sel.point_ids = self.point_ids[:]
        new_sel.entity_ids = self.entity_ids[:]
        new_sel.constraint_idx = self.constraint_idx
        new_sel.junction_pid = self.junction_pid
        return new_sel

    def select_constraint(self, idx: int, is_multi: bool):
        """Selects a constraint by index."""
        self.constraint_idx = idx
        if not is_multi:
            self.point_ids.clear()
            self.entity_ids.clear()
            self.junction_pid = None

    def select_junction(self, pid: int, is_multi: bool):
        """Selects an implicit junction point."""
        self.junction_pid = pid
        if not is_multi:
            self.point_ids.clear()
            self.entity_ids.clear()
            self.constraint_idx = None

    def select_point(self, pid: int, is_multi: bool):
        """Selects a point by ID."""
        self._update_list(self.point_ids, pid, is_multi)
        if not is_multi:
            self.entity_ids.clear()
            self.constraint_idx = None
            self.junction_pid = None

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
