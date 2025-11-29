# constraints/equal_distance.py

from __future__ import annotations
import math
from typing import (
    Tuple,
    Dict,
    Any,
    List,
    TYPE_CHECKING,
)
from .base import Constraint

if TYPE_CHECKING:
    from ..entities import EntityRegistry
    from ..params import ParameterContext


class EqualDistanceConstraint(Constraint):
    """Enforces that distance(p1, p2) equals distance(p3, p4)."""

    def __init__(self, p1: int, p2: int, p3: int, p4: int):
        self.p1 = p1
        self.p2 = p2
        self.p3 = p3
        self.p4 = p4

    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": "EqualDistanceConstraint",
            "p1": self.p1,
            "p2": self.p2,
            "p3": self.p3,
            "p4": self.p4,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "EqualDistanceConstraint":
        return cls(p1=data["p1"], p2=data["p2"], p3=data["p3"], p4=data["p4"])

    def error(
        self, reg: "EntityRegistry", params: "ParameterContext"
    ) -> float:
        pt1 = reg.get_point(self.p1)
        pt2 = reg.get_point(self.p2)
        dist1 = math.hypot(pt2.x - pt1.x, pt2.y - pt1.y)

        pt3 = reg.get_point(self.p3)
        pt4 = reg.get_point(self.p4)
        dist2 = math.hypot(pt4.x - pt3.x, pt4.y - pt3.y)

        return dist1 - dist2

    def gradient(
        self, reg: "EntityRegistry", params: "ParameterContext"
    ) -> Dict[int, List[Tuple[float, float]]]:
        pt1 = reg.get_point(self.p1)
        pt2 = reg.get_point(self.p2)
        pt3 = reg.get_point(self.p3)
        pt4 = reg.get_point(self.p4)

        dx1 = pt2.x - pt1.x
        dy1 = pt2.y - pt1.y
        dist1 = math.hypot(dx1, dy1)

        dx2 = pt4.x - pt3.x
        dy2 = pt4.y - pt3.y
        dist2 = math.hypot(dx2, dy2)

        grad: Dict[int, List[Tuple[float, float]]] = {}

        def add(pid, gx, gy):
            if pid not in grad:
                grad[pid] = [(0.0, 0.0)]
            cx, cy = grad[pid][0]
            grad[pid][0] = (cx + gx, cy + gy)

        if dist1 > 1e-9:
            u1x, u1y = dx1 / dist1, dy1 / dist1
            add(self.p1, -u1x, -u1y)
            add(self.p2, u1x, u1y)

        if dist2 > 1e-9:
            u2x, u2y = dx2 / dist2, dy2 / dist2
            # Subtracting dist2, so flip signs
            add(self.p3, u2x, u2y)  # -(-u2)
            add(self.p4, -u2x, -u2y)  # -(u2)

        return grad
