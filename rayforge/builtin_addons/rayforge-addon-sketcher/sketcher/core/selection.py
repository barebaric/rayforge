from typing import TYPE_CHECKING

from blinker import Signal
from raygeo.sketch.selection import Selection

from .types import EntityID

if TYPE_CHECKING:
    from raygeo.sketch.entities import Entity
    from raygeo.sketch.registry import EntityRegistry


class SketchSelection:
    """Manages the selection state of the sketch editor."""

    def __init__(self):
        self._selection = Selection()
        self.changed = Signal()

    @property
    def point_ids(self) -> list[EntityID]:
        """The selected point IDs."""
        return self._selection.point_ids

    @point_ids.setter
    def point_ids(self, value: list[EntityID]):
        self._selection.point_ids = list(value)

    @property
    def entity_ids(self) -> list[EntityID]:
        """The selected entity IDs."""
        return self._selection.entity_ids

    @entity_ids.setter
    def entity_ids(self, value: list[EntityID]):
        self._selection.entity_ids = list(value)

    @property
    def constraint_idx(self) -> int | None:
        """The selected constraint index, if any."""
        return self._selection.constraint_idx

    @constraint_idx.setter
    def constraint_idx(self, value: int | None):
        self._selection.constraint_idx = value

    @property
    def junction_pid(self) -> EntityID | None:
        """The selected implicit junction point, if any."""
        return self._selection.junction_pid

    @junction_pid.setter
    def junction_pid(self, value: EntityID | None):
        self._selection.junction_pid = value

    def clear(self):
        """Clears all selections."""
        self._selection.clear()
        self.changed.send(self)

    def copy(self) -> "SketchSelection":
        """Creates a shallow copy of the selection state."""
        new_sel = SketchSelection()
        new_sel._selection = self._selection.copy()
        return new_sel

    def is_empty(self) -> bool:
        """Returns True if nothing is selected."""
        return self._selection.is_empty()

    def resolve_entities(
        self, registry: "EntityRegistry | None"
    ) -> "list[Entity] | None":
        """Resolves the selected entities, or None if the registry is
        unavailable or an entity cannot be found."""
        if registry is None:
            return None
        return self._selection.resolve_entities(registry)

    def select_constraint(self, idx: int, is_multi: bool):
        """Selects a constraint by index."""
        self._selection.select_constraint(idx, is_multi)
        self.changed.send(self)

    def select_junction(self, pid: EntityID, is_multi: bool):
        """Selects an implicit junction point."""
        self._selection.select_junction(pid, is_multi)
        self.changed.send(self)

    def select_point(self, pid: EntityID, is_multi: bool):
        """Selects a point by ID."""
        self._selection.select_point(pid, is_multi)
        self.changed.send(self)

    def select_entity(self, entity, is_multi: bool):
        """Selects a geometric entity (Line or Arc)."""
        # We do NOT select the control points here: deleting an entity
        # must not cascade to its points and destroy shared geometry.
        # Visual highlighting is handled by the renderer.
        self._selection.select_entity(entity.id, is_multi)
        self.changed.send(self)

    def select_connected_entities(
        self, entity_id: EntityID, registry: "EntityRegistry"
    ):
        """
        Adds all entities connected to the given entity through shared
        points to the current selection.

        Args:
            entity_id: The ID of the starting entity.
            registry: The EntityRegistry to query for connected entities.
        """
        self._selection.select_connected_entities(entity_id, registry)
        self.changed.send(self)

    def add_point(self, pid: EntityID):
        """Adds one point without emitting the changed signal."""
        self._selection.add_point(pid)

    def add_entity(self, entity_id: EntityID):
        """Adds one entity without emitting the changed signal."""
        self._selection.add_entity(entity_id)

    def clear_points(self):
        """Clears the selected points without emitting the changed signal."""
        self._selection.clear_points()

    def clear_entities(self):
        """Clears the selected entities without emitting the changed signal."""
        self._selection.clear_entities()
