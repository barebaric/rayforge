from typing import Any

from raygeo.geo.shape.text import FontConfig
from raygeo.geo.types import Point as GeoPoint

from .entities.arc import Arc
from .entities.bezier import Bezier
from .entities.circle import Circle
from .entities.ellipse import Ellipse
from .entities.entity import Entity
from .entities.line import Line
from .entities.point import Point
from .entities.polygon import PolygonEntity
from .entities.text_box import TextBoxEntity
from .types import EntityID

_ENTITY_CLASSES = {
    "arc": Arc,
    "bezier": Bezier,
    "circle": Circle,
    "ellipse": Ellipse,
    "line": Line,
    "polygon": PolygonEntity,
    "text_box": TextBoxEntity,
}


class EntityRegistry:
    """Stores all points and primitives."""

    def __init__(self) -> None:
        self.points: list[Point] = []
        self.entities: list[Entity] = []
        self._entity_map: dict[EntityID, Entity] = {}
        self._id_counter: EntityID = 0
        self._entity_version: int = 0
        # Tracks how many entities reference each point. O(1) is_point_used.
        self._point_usage_count: dict[EntityID, int] = {}

    def _increment_point_usage(self, pid: EntityID) -> None:
        self._point_usage_count[pid] = self._point_usage_count.get(pid, 0) + 1

    def _decrement_point_usage(self, pid: EntityID) -> None:
        count = self._point_usage_count.get(pid, 0)
        if count <= 1:
            self._point_usage_count.pop(pid, None)
        else:
            self._point_usage_count[pid] = count - 1

    def _count_entities_for(self, entity: Entity) -> None:
        """Increment usage count for all points referenced by an entity."""
        for pid in entity.get_point_ids():
            self._increment_point_usage(pid)

    def _decide_entities_for(self, entity: Entity) -> None:
        """Decrement usage count for all points referenced by an entity."""
        for pid in entity.get_point_ids():
            self._decrement_point_usage(pid)

    def to_dict(self) -> dict[str, Any]:
        """Serializes the registry to a dictionary."""
        return {
            "points": [p.to_dict() for p in self.points],
            "entities": [e.to_dict() for e in self.entities],
            "id_counter": self._id_counter,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "EntityRegistry":
        """Deserializes a dictionary into an EntityRegistry instance."""
        new_reg = cls()
        new_reg.points = [
            Point.from_dict(p_data) for p_data in data.get("points", [])
        ]
        entities_data = data.get("entities", [])
        for e_data in entities_data:
            e_type = e_data.get("type")
            e_cls = _ENTITY_CLASSES.get(e_type)
            if e_cls:
                entity = e_cls.from_dict(e_data)
                entity.restore_base_flags(e_data)
                new_reg.entities.append(entity)
                new_reg._entity_map[entity.id] = entity

        new_reg._id_counter = data.get("id_counter", 0)
        new_reg.rebuild_usage_counts()
        return new_reg

    def rebuild_usage_counts(self) -> None:
        """Rebuilds _point_usage_count by scanning all entities.
        Called after deserialization so the counter is consistent."""
        self._point_usage_count.clear()
        for entity in self.entities:
            self._count_entities_for(entity)

    def add_arc(
        self,
        start: EntityID,
        end: EntityID,
        center: EntityID,
        cw: bool = False,
        construction: bool = False,
    ) -> EntityID:
        eid = self._id_counter
        entity = Arc(
            eid, start, end, center, clockwise=cw, construction=construction
        )
        self.entities.append(entity)
        self._entity_map[eid] = entity
        self._id_counter += 1
        self._entity_version += 1
        self._count_entities_for(entity)
        return eid

    def add_bezier(
        self,
        start_idx: EntityID,
        end_idx: EntityID,
        construction: bool = False,
        cp1: GeoPoint | None = None,
        cp2: GeoPoint | None = None,
    ) -> EntityID:
        eid = self._id_counter
        entity = Bezier(
            eid,
            start_idx,
            end_idx,
            construction=construction,
            cp1=cp1,
            cp2=cp2,
        )
        self.entities.append(entity)
        self._entity_map[eid] = entity
        self._id_counter += 1
        self._entity_version += 1
        self._count_entities_for(entity)
        return eid

    def add_circle(
        self,
        center_idx: EntityID,
        radius_pt_idx: EntityID,
        construction: bool = False,
    ) -> EntityID:
        eid = self._id_counter
        entity = Circle(
            eid, center_idx, radius_pt_idx, construction=construction
        )
        self.entities.append(entity)
        self._entity_map[eid] = entity
        self._id_counter += 1
        self._entity_version += 1
        self._count_entities_for(entity)
        return eid

    def add_ellipse(
        self,
        center_idx: EntityID,
        radius_x_pt_idx: EntityID,
        radius_y_pt_idx: EntityID,
        construction: bool = False,
    ) -> EntityID:
        eid = self._id_counter
        entity = Ellipse(
            eid,
            center_idx,
            radius_x_pt_idx,
            radius_y_pt_idx,
            construction=construction,
        )
        self.entities.append(entity)
        self._entity_map[eid] = entity
        self._id_counter += 1
        self._entity_version += 1
        self._count_entities_for(entity)
        return eid

    def add_line(
        self, p1_idx: EntityID, p2_idx: EntityID, construction: bool = False
    ) -> EntityID:
        eid = self._id_counter
        entity = Line(eid, p1_idx, p2_idx, construction=construction)
        self.entities.append(entity)
        self._entity_map[eid] = entity
        self._id_counter += 1
        self._entity_version += 1
        self._count_entities_for(entity)
        return eid

    def add_point(self, x: float, y: float, fixed: bool = False) -> EntityID:
        pid = self._id_counter
        self.points.append(Point(pid, x, y, fixed))
        self._id_counter += 1
        self._entity_version += 1
        return pid

    def add_text_box(
        self,
        origin_id: EntityID,
        width_id: EntityID,
        height_id: EntityID,
        content: str = "",
        font_config: FontConfig | None = None,
    ) -> EntityID:
        eid = self._id_counter
        entity = TextBoxEntity(
            eid,
            origin_id,
            width_id,
            height_id,
            content=content,
            font_config=font_config,
        )
        self.entities.append(entity)
        self._entity_map[eid] = entity
        self._id_counter += 1
        self._entity_version += 1
        self._count_entities_for(entity)
        return eid

    def remove_entities_by_id(self, entity_ids: list[EntityID]):
        """Removes one or more entities from the registry by their IDs."""
        ids_to_remove = set(entity_ids)
        # Decrement usage counts *before* removing entities so we
        # still have access to their point references.
        for eid in ids_to_remove:
            entity = self._entity_map.get(eid)
            if entity is not None:
                self._decide_entities_for(entity)
        self.entities = [e for e in self.entities if e.id not in ids_to_remove]
        self._entity_map = {e.id: e for e in self.entities}
        self._entity_version += 1

    def is_point_used(self, pid: EntityID) -> bool:
        """Checks if a point is used by any entity in the sketch.

        O(1) lookup via the usage-count counter maintained on every
        entity add/remove.
        """
        return self._point_usage_count.get(pid, 0) > 0

    def get_point(self, idx: EntityID) -> Point:
        """Retrieves a point by its ID."""
        if 0 <= idx < len(self.points) and self.points[idx].id == idx:
            return self.points[idx]

        for p in self.points:
            if p.id == idx:
                return p
        raise IndexError(f"Point with ID {idx} not found")

    def get_entity(self, idx: EntityID) -> Entity | None:
        """Retrieves a geometric entity (Line/Arc/Circle) by ID in O(1)."""
        return self._entity_map.get(idx)

    def geometry_signature(self, entity_id: EntityID) -> tuple | None:
        """Returns the quantized shape signature of the entity with
        the given ID, or None if no such entity exists. Delegates to
        the entity's polymorphic implementation (see
        ``Entity.geometry_signature``)."""
        entity = self._entity_map.get(entity_id)
        if entity is None:
            return None
        return entity.geometry_signature(self)

    def get_connected_entity_ids(
        self, start_entity_id: EntityID
    ) -> set[EntityID]:
        """
        Finds all entities transitively connected to the start entity
        through shared points using BFS.

        Args:
            start_entity_id: The ID of the entity to start from.

        Returns:
            A set of entity IDs that are connected to the start entity,
            including the start entity itself.
        """
        start_entity = self.get_entity(start_entity_id)
        if start_entity is None:
            return set()

        connected_entities: set[EntityID] = {start_entity_id}
        points_to_visit: set[EntityID] = set(start_entity.get_point_ids())
        visited_points: set[EntityID] = set()

        while points_to_visit:
            pid = points_to_visit.pop()
            if pid in visited_points:
                continue
            visited_points.add(pid)

            for entity in self.entities:
                if entity.id in connected_entities:
                    continue
                entity_points = entity.get_point_ids()
                if pid in entity_points:
                    connected_entities.add(entity.id)
                    for ep in entity_points:
                        if ep not in visited_points:
                            points_to_visit.add(ep)

        return connected_entities

    def get_rigidly_connected_points(
        self, point_id: EntityID
    ) -> list[EntityID]:
        """
        Returns point IDs that should move together with the given point
        as a rigid body during dragging. Iterates over all entities to find
        any that have rigid connections for this point.
        """
        result = []
        for entity in self.entities:
            rigid_points = entity.get_rigidly_connected_points(point_id)
            result.extend(rigid_points)
        return list(set(result))

    def get_drag_anchor_points(self, point_id: EntityID) -> list[EntityID]:
        """
        Returns point IDs that should be pinned at their current position
        while the given point is dragged. Iterates over all entities to
        find any that define anchor points for this point.
        """
        result = []
        for entity in self.entities:
            anchor_points = entity.get_drag_anchor_points(point_id)
            result.extend(anchor_points)
        return list(set(result))
