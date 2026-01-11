# constraints/horizontal.py

from __future__ import annotations
import math
from typing import (
    Tuple,
    Dict,
    Any,
    List,
    Callable,
    TYPE_CHECKING,
)
from .base import Constraint

if TYPE_CHECKING:
    from ..params import ParameterContext
    from ..registry import EntityRegistry


class HorizontalConstraint(Constraint):
    """Enforces two points have the same Y coordinate."""

    def __init__(self, p1: int, p2: int):
        self.p1 = p1
        self.p2 = p2

    def to_dict(self) -> Dict[str, Any]:
        return {"type": "HorizontalConstraint", "p1": self.p1, "p2": self.p2}

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "HorizontalConstraint":
        return cls(p1=data["p1"], p2=data["p2"])

    def error(
        self, reg: "EntityRegistry", params: "ParameterContext"
    ) -> float:
        return reg.get_point(self.p1).y - reg.get_point(self.p2).y

    def gradient(
        self, reg: "EntityRegistry", params: "ParameterContext"
    ) -> Dict[int, List[Tuple[float, float]]]:
        return {
            self.p1: [(0.0, 1.0)],
            self.p2: [(0.0, -1.0)],
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

            t = 0.2
            mx = s1[0] + (s2[0] - s1[0]) * t
            my = s1[1] + (s2[1] - s1[1]) * t
            cx = mx
            cy = my - 10
            return math.hypot(sx - cx, sy - cy) < threshold
        return False
