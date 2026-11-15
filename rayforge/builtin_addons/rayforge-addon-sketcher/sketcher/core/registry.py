"""Compatibility surface for the raygeo sketch document model.

The document model (points, entities, constraints and id
allocation) is owned by raygeo's ``EntityRegistry``; this module
re-exports it and adds the connected-component query rayforge used
to provide itself.
"""

from __future__ import annotations

from collections.abc import Iterator

from raygeo.sketch.registry import EntityRegistry
from raygeo.sketch.types import EntityID

__all__ = ["EntityRegistry", "iter_connected_entity_ids"]


def iter_connected_entity_ids(
    registry: EntityRegistry, start_entity_id: EntityID
) -> Iterator[EntityID]:
    """Yields all entities transitively connected to the start entity
    through shared points, breadth first, including the start
    entity."""
    entity_ids = set(registry.entity_ids())
    start_points = registry.entity_point_ids(start_entity_id)
    if start_points is None:
        return
    connected: set[EntityID] = {start_entity_id}
    points_to_visit = set(start_points)
    visited_points: set[EntityID] = set()
    while points_to_visit:
        pid = points_to_visit.pop()
        if pid in visited_points:
            continue
        visited_points.add(pid)
        for eid in entity_ids:
            if eid in connected:
                continue
            entity_points = registry.entity_point_ids(eid) or []
            if pid in entity_points:
                connected.add(eid)
                points_to_visit.update(entity_points)
    yield from connected


def get_connected_entity_ids(
    registry: EntityRegistry, start_entity_id: EntityID
) -> set[EntityID]:
    """All entities transitively connected to the start entity
    through shared points, including the start entity."""
    return set(iter_connected_entity_ids(registry, start_entity_id))
