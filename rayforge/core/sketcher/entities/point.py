import math
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, TYPE_CHECKING

from ...geo import Point as GeoPoint, Rect, primitives
from .bezier import Bezier

if TYPE_CHECKING:
    from ..registry import EntityRegistry


class WaypointType(Enum):
    SHARP = "sharp"
    SMOOTH = "smooth"
    SYMMETRIC = "symmetric"


class Point:
    def __init__(
        self,
        id: int,
        x: float,
        y: float,
        fixed: bool = False,
        waypoint_type: WaypointType = WaypointType.SHARP,
    ):
        self.id = id
        self.x = x
        self.y = y
        self.fixed = fixed
        self.waypoint_type = waypoint_type
        self.constrained: bool = False

    def is_sharp(self) -> bool:
        return self.waypoint_type == WaypointType.SHARP

    def is_smooth(self) -> bool:
        return self.waypoint_type == WaypointType.SMOOTH

    def is_symmetric(self) -> bool:
        return self.waypoint_type == WaypointType.SYMMETRIC

    def get_connected_beziers(
        self, registry: "EntityRegistry"
    ) -> List["Bezier"]:
        connected = []
        for entity in registry.entities:
            if isinstance(entity, Bezier):
                if self.id in (entity.start_idx, entity.end_idx):
                    connected.append(entity)
        return connected

    def get_paired_beziers(
        self, registry: "EntityRegistry"
    ) -> Tuple[Optional["Bezier"], Optional["Bezier"]]:
        connected = self.get_connected_beziers(registry)
        if len(connected) >= 2:
            return connected[0], connected[1]
        elif len(connected) == 1:
            return connected[0], None
        return None, None

    def apply_constraint(
        self,
        registry: "EntityRegistry",
        bezier: "Bezier",
        cp_index: int,
    ) -> None:
        if self.is_sharp():
            return

        b1, b2 = self.get_paired_beziers(registry)
        if b1 is None or b2 is None:
            return

        other_bezier = b2 if bezier == b1 else b1

        if bezier.start_idx == self.id and cp_index == 1:
            cp_out = bezier.cp1
            if cp_out is not None:
                if other_bezier.end_idx == self.id:
                    other_bezier.cp2 = self._compute_constrained_cp(
                        cp_out,
                        other_bezier.cp2,
                        symmetric=self.is_symmetric(),
                    )
                elif other_bezier.start_idx == self.id:
                    other_bezier.cp1 = self._compute_constrained_cp(
                        cp_out,
                        other_bezier.cp1,
                        symmetric=self.is_symmetric(),
                    )
        elif bezier.end_idx == self.id and cp_index == 2:
            cp_in = bezier.cp2
            if cp_in is not None:
                if other_bezier.start_idx == self.id:
                    other_bezier.cp1 = self._compute_constrained_cp(
                        cp_in,
                        other_bezier.cp1,
                        symmetric=self.is_symmetric(),
                    )
                elif other_bezier.end_idx == self.id:
                    other_bezier.cp2 = self._compute_constrained_cp(
                        cp_in,
                        other_bezier.cp2,
                        symmetric=self.is_symmetric(),
                    )

    def _compute_constrained_cp(
        self,
        modified_cp: Tuple[float, float],
        other_cp: Optional[Tuple[float, float]],
        symmetric: bool,
    ) -> Tuple[float, float]:
        if symmetric:
            return (-modified_cp[0], -modified_cp[1])
        else:
            if other_cp is None:
                return (-modified_cp[0], -modified_cp[1])
            other_length = math.sqrt(other_cp[0] ** 2 + other_cp[1] ** 2)
            if other_length < 1e-10:
                return (-modified_cp[0], -modified_cp[1])
            modified_length = math.sqrt(
                modified_cp[0] ** 2 + modified_cp[1] ** 2
            )
            if modified_length < 1e-10:
                return (0.0, 0.0)
            direction = (
                -modified_cp[0] / modified_length,
                -modified_cp[1] / modified_length,
            )
            return (
                direction[0] * other_length,
                direction[1] * other_length,
            )

    def pos(self) -> GeoPoint:
        return (self.x, self.y)

    def to_dict(self) -> Dict[str, Any]:
        data = {
            "id": self.id,
            "x": self.x,
            "y": self.y,
            "fixed": self.fixed,
        }
        if self.waypoint_type != WaypointType.SHARP:
            data["waypoint_type"] = self.waypoint_type.value
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Point":
        wp_type_str = data.get("waypoint_type", "sharp")
        wp_type = WaypointType(wp_type_str)
        return cls(
            id=data["id"],
            x=data["x"],
            y=data["y"],
            fixed=data.get("fixed", False),
            waypoint_type=wp_type,
        )

    def is_in_rect(self, rect: Rect) -> bool:
        return primitives.is_point_in_rect(self.pos(), rect)

    def __repr__(self) -> str:
        return (
            f"Point(id={self.id}, x={self.x}, y={self.y}, "
            f"type={self.waypoint_type.value})"
        )
