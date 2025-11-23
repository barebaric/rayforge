from typing import List, Tuple, Dict, Optional


class Point:
    def __init__(self, id: int, x: float, y: float, fixed: bool = False):
        self.id = id
        self.x = x
        self.y = y
        self.fixed = fixed
        # State tracked by solver
        self.constrained: bool = False

    def pos(self) -> Tuple[float, float]:
        return (self.x, self.y)

    def __repr__(self) -> str:
        return (
            f"Point(id={self.id}, x={self.x}, y={self.y}, fixed={self.fixed})"
        )


class Entity:
    """Base class for geometric primitives."""

    def __init__(self, id: int, construction: bool = False):
        self.id = id
        self.construction = construction
        self.type = "entity"
        # Constrained state is calculated by solver
        self.constrained = False

    def __repr__(self) -> str:
        return f"Entity(id={self.id}, type={self.type})"


class Line(Entity):
    def __init__(
        self, id: int, p1_idx: int, p2_idx: int, construction: bool = False
    ):
        super().__init__(id, construction)
        self.p1_idx = p1_idx
        self.p2_idx = p2_idx
        self.type = "line"

    def __repr__(self) -> str:
        return (
            f"Line(id={self.id}, p1={self.p1_idx}, p2={self.p2_idx}, "
            f"construction={self.construction})"
        )


class Arc(Entity):
    def __init__(
        self,
        id: int,
        start_idx: int,
        end_idx: int,
        center_idx: int,
        clockwise: bool = False,
        construction: bool = False,
    ):
        super().__init__(id, construction)
        self.start_idx = start_idx
        self.end_idx = end_idx
        self.center_idx = center_idx
        self.clockwise = clockwise
        self.type = "arc"

    def __repr__(self) -> str:
        return (
            f"Arc(id={self.id}, start={self.start_idx}, end={self.end_idx}, "
            f"center={self.center_idx}, cw={self.clockwise})"
        )


class EntityRegistry:
    """Stores all points and primitives."""

    def __init__(self) -> None:
        self.points: List[Point] = []
        self.entities: List[Entity] = []
        self._entity_map: Dict[int, Entity] = {}
        self._id_counter = 0

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

    def get_point(self, idx: int) -> Point:
        """Retrieves a point by its ID."""
        if 0 <= idx < len(self.points) and self.points[idx].id == idx:
            return self.points[idx]

        for p in self.points:
            if p.id == idx:
                return p
        raise IndexError(f"Point with ID {idx} not found")

    def get_entity(self, idx: int) -> Optional[Entity]:
        """Retrieves a geometric entity (Line/Arc) by ID in O(1)."""
        return self._entity_map.get(idx)
