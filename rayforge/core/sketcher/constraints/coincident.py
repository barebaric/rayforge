# constraints/coincident.py

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


class CoincidentConstraint(Constraint):
    """Enforces two points are at the same location."""

    def __init__(self, p1: int, p2: int):
        self.p1 = p1
        self.p2 = p2

    def to_dict(self) -> Dict[str, Any]:
        return {"type": "CoincidentConstraint", "p1": self.p1, "p2": self.p2}

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CoincidentConstraint":
        return cls(p1=data["p1"], p2=data["p2"])

    def error(
        self, reg: "EntityRegistry", params: "ParameterContext"
    ) -> Tuple[float, float]:
        pt1 = reg.get_point(self.p1)
        pt2 = reg.get_point(self.p2)
        return (pt1.x - pt2.x, pt1.y - pt2.y)

    def gradient(
        self, reg: "EntityRegistry", params: "ParameterContext"
    ) -> Dict[int, List[Tuple[float, float]]]:
        return {
            self.p1: [(1.0, 0.0), (0.0, 1.0)],
            self.p2: [(-1.0, 0.0), (0.0, -1.0)],
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
        origin_id = getattr(element.sketch, "origin_id", -1)
        pid_to_check = self.p1
        if self.p1 == origin_id and origin_id != -1:
            pid_to_check = self.p2

        pt_to_check = reg.get_point(pid_to_check)
        if pt_to_check:
            s_pt = to_screen((pt_to_check.x, pt_to_check.y))
            return math.hypot(sx - s_pt[0], sy - s_pt[1]) < threshold
        return False
