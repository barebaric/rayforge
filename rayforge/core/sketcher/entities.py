from typing import List, Tuple, Dict, Optional, Any


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

    def to_dict(self) -> Dict[str, Any]:
        """Serializes the Point to a dictionary."""
        return {
            "id": self.id,
            "x": self.x,
            "y": self.y,
            "fixed": self.fixed,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Point":
        """Deserializes a dictionary into a Point instance."""
        return cls(
            id=data["id"],
            x=data["x"],
            y=data["y"],
            fixed=data.get("fixed", False),
        )

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

    def to_dict(self) -> Dict[str, Any]:
        """Base serialization method for entities."""
        return {
            "id": self.id,
            "type": self.type,
            "construction": self.construction,
        }

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

    def to_dict(self) -> Dict[str, Any]:
        """Serializes the Line to a dictionary."""
        data = super().to_dict()
        data.update({"p1_idx": self.p1_idx, "p2_idx": self.p2_idx})
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Line":
        """Deserializes a dictionary into a Line instance."""
        return cls(
            id=data["id"],
            p1_idx=data["p1_idx"],
            p2_idx=data["p2_idx"],
            construction=data.get("construction", False),
        )

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

    def to_dict(self) -> Dict[str, Any]:
        """Serializes the Arc to a dictionary."""
        data = super().to_dict()
        data.update(
            {
                "start_idx": self.start_idx,
                "end_idx": self.end_idx,
                "center_idx": self.center_idx,
                "clockwise": self.clockwise,
            }
        )
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Arc":
        """Deserializes a dictionary into an Arc instance."""
        return cls(
            id=data["id"],
            start_idx=data["start_idx"],
            end_idx=data["end_idx"],
            center_idx=data["center_idx"],
            clockwise=data.get("clockwise", False),
            construction=data.get("construction", False),
        )

    def __repr__(self) -> str:
        return (
            f"Arc(id={self.id}, start={self.start_idx}, end={self.end_idx}, "
            f"center={self.center_idx}, cw={self.clockwise})"
        )


class Circle(Entity):
    def __init__(
        self,
        id: int,
        center_idx: int,
        radius_pt_idx: int,
        construction: bool = False,
    ):
        super().__init__(id, construction)
        self.center_idx = center_idx
        self.radius_pt_idx = radius_pt_idx
        self.type = "circle"

    def to_dict(self) -> Dict[str, Any]:
        """Serializes the Circle to a dictionary."""
        data = super().to_dict()
        data.update(
            {
                "center_idx": self.center_idx,
                "radius_pt_idx": self.radius_pt_idx,
            }
        )
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Circle":
        """Deserializes a dictionary into a Circle instance."""
        return cls(
            id=data["id"],
            center_idx=data["center_idx"],
            radius_pt_idx=data["radius_pt_idx"],
            construction=data.get("construction", False),
        )

    def __repr__(self) -> str:
        return (
            f"Circle(id={self.id}, center={self.center_idx}, "
            f"radius_pt={self.radius_pt_idx}, "
            f"construction={self.construction})"
        )


_ENTITY_CLASSES = {"line": Line, "arc": Arc, "circle": Circle}


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

    def is_point_used(self, pid: int) -> bool:
        """Checks if a point is used by any entity in the sketch."""
        for e in self.entities:
            p_ids = []
            if isinstance(e, Line):
                p_ids = [e.p1_idx, e.p2_idx]
            elif isinstance(e, Arc):
                p_ids = [e.start_idx, e.end_idx, e.center_idx]
            elif isinstance(e, Circle):
                p_ids = [e.center_idx, e.radius_pt_idx]
            if pid in p_ids:
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
