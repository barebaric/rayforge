from __future__ import annotations
from typing import Tuple, Dict, Any, List, TYPE_CHECKING
from .base import Constraint

if TYPE_CHECKING:
    from ..params import ParameterContext
    from ..registry import EntityRegistry


class CollinearConstraint(Constraint):
    """Enforces that three points (p1, p2, p3) lie on the same line."""

    def __init__(self, p1: int, p2: int, p3: int):
        self.p1 = p1
        self.p2 = p2
        self.p3 = p3

    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": "CollinearConstraint",
            "p1": self.p1,
            "p2": self.p2,
            "p3": self.p3,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CollinearConstraint":
        return cls(p1=data["p1"], p2=data["p2"], p3=data["p3"])

    def error(
        self, reg: "EntityRegistry", params: "ParameterContext"
    ) -> float:
        pt1 = reg.get_point(self.p1)
        pt2 = reg.get_point(self.p2)
        pt3 = reg.get_point(self.p3)
        # Cross product of (p2 - p1) and (p3 - p1)
        return (pt2.x - pt1.x) * (pt3.y - pt1.y) - (pt2.y - pt1.y) * (
            pt3.x - pt1.x
        )

    def gradient(
        self, reg: "EntityRegistry", params: "ParameterContext"
    ) -> Dict[int, List[Tuple[float, float]]]:
        p1 = reg.get_point(self.p1)
        p2 = reg.get_point(self.p2)
        p3 = reg.get_point(self.p3)

        # E = (p2x - p1x) * (p3y - p1y) - (p2y - p1y) * (p3x - p1x)
        # dE/dp1x = -(p3y - p1y) + (p2y - p1y) = p2y - p3y
        # dE/dp1y = -(p2x - p1x) + (p3x - p1x) = p3x - p2x
        # dE/dp2x = (p3y - p1y)
        # dE/dp2y = -(p3x - p1x)
        # dE/dp3x = -(p2y - p1y)
        # dE/dp3y = (p2x - p1x)

        return {
            self.p1: [(p2.y - p3.y, p3.x - p2.x)],
            self.p2: [(p3.y - p1.y, -(p3.x - p1.x))],
            self.p3: [(-(p2.y - p1.y), p2.x - p1.x)],
        }
