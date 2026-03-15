from typing import List, Dict, Optional, Any, Set
from ..geo.font_config import FontConfig
from .entities.point import Point
from .entities.entity import Entity
from .entities.line import Line
from .entities.arc import Arc
from .entities.circle import Circle
from .entities.text_box import TextBoxEntity

_ENTITY_CLASSES = {
    "line": Line,
    "arc": Arc,
    "circle": Circle,
    "text_box": TextBoxEntity,
}


class EntityRegistry:
    """Stores all points and primitives."""

    def __init__(self) -> None:
        self.points: List[Point] = []
        self.entities: List[Entity] = []
        self._entity_map: Dict[int, Entity] = {}
        self._id_counter = 0

    def to_dict(self) -> Dict[str, Any]:
        """Serializes the registry to a dictionary."""
        return {
            "points": [p.to_dict() for p in self.points],
            "entities": [e.to_dict() for e in self.entities],
            "id_counter": self._id_counter,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "EntityRegistry":
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
                new_reg.entities.append(entity)
                new_reg._entity_map[entity.id] = entity

        new_reg._id_counter = data.get("id_counter", 0)
        return new_reg

    def add_point(self, x: float, y: float, fixed: bool = False) -> int:
        pid = self._id_counter
        self.points.append(Point(pid, x, y, fixed))
        self._id_counter += 1
        return pid

    def add_line(
        self, p1_idx: int, p2_idx: int, construction: bool = False
    ) -> int:
        eid = self._id_counter
        entity = Line(eid, p1_idx, p2_idx, construction=construction)
        self.entities.append(entity)
        self._entity_map[eid] = entity
        self._id_counter += 1
        return eid

    def add_arc(
        self,
        start: int,
        end: int,
        center: int,
        cw: bool = False,
        construction: bool = False,
    ) -> int:
        eid = self._id_counter
        entity = Arc(
            eid, start, end, center, clockwise=cw, construction=construction
        )
        self.entities.append(entity)
        self._entity_map[eid] = entity
        self._id_counter += 1
        return eid

    def add_circle(
        self, center_idx: int, radius_pt_idx: int, construction: bool = False
    ) -> int:
        eid = self._id_counter
        entity = Circle(
            eid, center_idx, radius_pt_idx, construction=construction
        )
        self.entities.append(entity)
        self._entity_map[eid] = entity
        self._id_counter += 1
        return eid

    def add_text_box(
        self,
        origin_id: int,
        width_id: int,
        height_id: int,
        content: str = "",
        font_config: Optional[FontConfig] = None,
    ) -> int:
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
        return eid

    def remove_entities_by_id(self, entity_ids: List[int]):
        """Removes one or more entities from the registry by their IDs."""
        ids_to_remove = set(entity_ids)
        self.entities = [e for e in self.entities if e.id not in ids_to_remove]
        # Rebuild the map for simplicity and correctness
        self._entity_map = {e.id: e for e in self.entities}

    def is_point_used(self, pid: int) -> bool:
        """Checks if a point is used by any entity in the sketch."""
        for e in self.entities:
            if pid in e.get_point_ids():
                return True
        return False

    def get_point(self, idx: int) -> Point:
        """Retrieves a point by its ID."""
        if 0 <= idx < len(self.points) and self.points[idx].id == idx:
            return self.points[idx]

        for p in self.points:
            if p.id == idx:
                return p
        raise IndexError(f"Point with ID {idx} not found")

    def get_entity(self, idx: int) -> Optional[Entity]:
        """Retrieves a geometric entity (Line/Arc/Circle) by ID in O(1)."""
        return self._entity_map.get(idx)

    def get_connected_entity_ids(self, start_entity_id: int) -> Set[int]:
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

        connected_entities: Set[int] = {start_entity_id}
        points_to_visit: Set[int] = set(start_entity.get_point_ids())
        visited_points: Set[int] = set()

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
