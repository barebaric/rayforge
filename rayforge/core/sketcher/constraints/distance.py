from __future__ import annotations
import math
from typing import (
    Union,
    Tuple,
    Dict,
    Any,
    List,
    Optional,
    Callable,
    TYPE_CHECKING,
)
from .base import Constraint

if TYPE_CHECKING:
    from ..entities import EntityRegistry
    from ..params import ParameterContext


class DistanceConstraint(Constraint):
    """Enforces distance between two points."""

    def __init__(
        self,
        p1: int,
        p2: int,
        value: Union[str, float],
        expression: Optional[str] = None,
    ):
        self.p1 = p1
        self.p2 = p2

        # Handle migration or dual initialization
        if expression is not None:
            self.expression = expression
            self.value = float(value)
        elif isinstance(value, str):
            self.expression = value
            self.value = 0.0  # Will be updated by update_from_context shortly
        else:
            self.expression = None
            self.value = float(value)

    def to_dict(self) -> Dict[str, Any]:
        data = {
            "type": "DistanceConstraint",
            "p1": self.p1,
            "p2": self.p2,
            "value": self.value,
        }
        if self.expression:
            data["expression"] = self.expression
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DistanceConstraint":
        return cls(
            p1=data["p1"],
            p2=data["p2"],
            value=data["value"],
            expression=data.get("expression"),
        )

    def error(
        self, reg: "EntityRegistry", params: "ParameterContext"
    ) -> float:
        pt1 = reg.get_point(self.p1)
        pt2 = reg.get_point(self.p2)
        # We use self.value which is cached/updated via update_from_context
        target = self.value
        dist = math.hypot(pt2.x - pt1.x, pt2.y - pt1.y)
        return dist - target

    def gradient(
        self, reg: "EntityRegistry", params: "ParameterContext"
    ) -> Dict[int, List[Tuple[float, float]]]:
        pt1 = reg.get_point(self.p1)
        pt2 = reg.get_point(self.p2)
        dx = pt2.x - pt1.x
        dy = pt2.y - pt1.y
        dist = math.hypot(dx, dy)

        ux, uy = 1.0, 0.0  # Default if points are coincident
        if dist > 1e-9:
            # Gradient is the unit vector
            ux, uy = dx / dist, dy / dist

        return {
            self.p1: [(-ux, -uy)],
            self.p2: [(ux, uy)],
        }

    def is_hit(
        self,
        sx: float,
        sy: float,
        reg: "EntityRegistry",
        to_screen: Callable[[Tuple[float, float]], Tuple[float, float]],
        element: Any,
        threshold: float,
    ) -> bool:
        p1 = reg.get_point(self.p1)
        p2 = reg.get_point(self.p2)
        if p1 and p2:
            s1 = to_screen((p1.x, p1.y))
            s2 = to_screen((p2.x, p2.y))
            mx, my = (s1[0] + s2[0]) / 2, (s1[1] + s2[1]) / 2
            return math.hypot(sx - mx, sy - my) < 15
        return False
