from typing import Tuple, Dict, Any
from ...geo import primitives


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

    def is_in_rect(self, rect: Tuple[float, float, float, float]) -> bool:
        """Checks if point is inside (min_x, min_y, max_x, max_y)."""
        return primitives.is_point_in_rect(self.pos(), rect)

    def __repr__(self) -> str:
        return (
            f"Point(id={self.id}, x={self.x}, y={self.y}, fixed={self.fixed})"
        )
