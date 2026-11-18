"""Compatibility helpers for the compiled sketch registry.

These keep behavior the compiled registry does not expose yet. They are
temporary and should move into raygeo as the backend migrates.
"""

from __future__ import annotations

from typing import Any

from raygeo.sketch.registry import EntityRegistry

from .types import EntityID


def require_point(registry: EntityRegistry, pid: EntityID):
    """Returns a point, raising IndexError when the id is unknown."""
    point = registry.get_point(pid)
    if point is None:
        raise IndexError(f"Point with ID {pid} not found")
    return point


def connected_entity_ids(
    registry: EntityRegistry, start_entity_id: EntityID
) -> set[EntityID]:
    """Entity ids transitively connected through shared control points."""
    if registry.get_entity(start_entity_id) is None:
        return set()

    connected: set[EntityID] = {start_entity_id}
    points_to_visit = set(registry.entity_point_ids(start_entity_id) or [])
    visited_points: set[EntityID] = set()

    while points_to_visit:
        pid = points_to_visit.pop()
        if pid in visited_points:
            continue
        visited_points.add(pid)

        for entity_id in registry.entity_ids():
            if entity_id in connected:
                continue
            entity_points = registry.entity_point_ids(entity_id) or []
            if pid in entity_points:
                connected.add(entity_id)
                points_to_visit.update(entity_points)

    return connected


def replace_entity_in_order(
    registry: EntityRegistry, entity_id: EntityID, entity: Any
) -> bool:
    """Replaces an entity while preserving its position in the list."""
    data = registry.to_dict()
    for index, item in enumerate(data["entities"]):
        if item.get("id") == entity_id:
            data["entities"][index] = entity.to_dict()
            registry.load_from_dict(data)
            return True
    return False


def insert_entity_in_order(
    registry: EntityRegistry, index: int, entity: Any
) -> None:
    """Inserts an entity at a specific position in the list."""
    data = registry.to_dict()
    data["entities"].insert(index, entity.to_dict())
    data["id_counter"] = max(
        data.get("id_counter", 0),
        int(entity.id) + 1,
    )
    registry.load_from_dict(data)


def next_temp_id(registry: EntityRegistry) -> EntityID:
    """The next free id, without allocating it."""
    return int(registry.to_dict().get("id_counter", 0))
