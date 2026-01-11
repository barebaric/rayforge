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
from ...geo.primitives import find_closest_point_on_line_segment
from .base import Constraint

if TYPE_CHECKING:
    from ..params import ParameterContext
    from ..registry import EntityRegistry


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

    def get_label_pos(
        self,
        reg: "EntityRegistry",
        to_screen: Callable[[Tuple[float, float]], Tuple[float, float]],
        element: Any,
    ):
        """Calculates screen position for distance constraint label."""
        p1 = reg.get_point(self.p1)
        p2 = reg.get_point(self.p2)
        if not (p1 and p2):
            return None

        s1 = to_screen((p1.x, p1.y))
        s2 = to_screen((p2.x, p2.y))
        mx = (s1[0] + s2[0]) / 2
        my = (s1[1] + s2[1]) / 2

        return mx, my

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

            _, _, dist_sq = find_closest_point_on_line_segment(s1, s2, sx, sy)

            if dist_sq < threshold**2:
                return True

        pos_data = self.get_label_pos(reg, to_screen, element)
        if pos_data:
            label_sx, label_sy = pos_data

            label_width = 20.0
            label_height = 20.0
            half_w = label_width / 2.0
            half_h = label_height / 2.0

            x_min = label_sx - half_w - 4.0
            x_max = label_sx + half_w + 4.0
            y_min = label_sy - half_h - 4.0
            y_max = label_sy + half_h + 4.0

            return x_min <= sx <= x_max and y_min <= sy <= y_max
        return False
