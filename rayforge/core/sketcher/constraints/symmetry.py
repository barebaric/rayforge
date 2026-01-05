# constraints/symmetry.py

from __future__ import annotations
import math
from typing import (
    Tuple,
    Dict,
    Any,
    List,
    Optional,
    Callable,
    TYPE_CHECKING,
)
from .base import Constraint
from ..entities import Line

if TYPE_CHECKING:
    from ..params import ParameterContext
    from ..registry import EntityRegistry


class SymmetryConstraint(Constraint):
    """
    Enforces symmetry between two points (p1, p2) with respect to:
    1. A Center Point (Point Symmetry)
    2. An Axis Line (Line Symmetry)
    """

    def __init__(
        self,
        p1: int,
        p2: int,
        center: Optional[int] = None,
        axis: Optional[int] = None,
    ):
        self.p1 = p1
        self.p2 = p2
        self.center = center
        self.axis = axis

    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": "SymmetryConstraint",
            "p1": self.p1,
            "p2": self.p2,
            "center": self.center,
            "axis": self.axis,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SymmetryConstraint":
        return cls(
            p1=data["p1"],
            p2=data["p2"],
            center=data.get("center"),
            axis=data.get("axis"),
        )

    def error(
        self, reg: "EntityRegistry", params: "ParameterContext"
    ) -> List[float]:
        pt1 = reg.get_point(self.p1)
        pt2 = reg.get_point(self.p2)

        if self.center is not None:
            # Case 1: Point Symmetry
            # Constraint: Center is the midpoint of P1 and P2
            # (P1 + P2) / 2 = Center  =>  P1 + P2 - 2*Center = 0
            s = reg.get_point(self.center)
            return [
                (pt1.x + pt2.x) - 2 * s.x,
                (pt1.y + pt2.y) - 2 * s.y,
            ]

        elif self.axis is not None:
            # Case 2: Line Symmetry
            # Constraint A: The segment P1-P2 is perpendicular to the Axis Line
            # Constraint B: The midpoint of P1-P2 lies on the Axis Line
            line = reg.get_entity(self.axis)
            if not isinstance(line, Line):
                return [0.0, 0.0]

            l1 = reg.get_point(line.p1_idx)
            l2 = reg.get_point(line.p2_idx)

            # Vector of the Axis Line
            dx_l = l2.x - l1.x
            dy_l = l2.y - l1.y

            # 1. Perpendicularity: Dot product (P2 - P1) . (L2 - L1) = 0
            dx_p = pt2.x - pt1.x
            dy_p = pt2.y - pt1.y
            err_perp = dx_p * dx_l + dy_p * dy_l

            # 2. Midpoint on Line: Cross product (Mid - L1) x (L2 - L1) = 0
            mx = (pt1.x + pt2.x) * 0.5
            my = (pt1.y + pt2.y) * 0.5
            err_collinear = (mx - l1.x) * dy_l - (my - l1.y) * dx_l

            return [err_perp, err_collinear]

        return [0.0, 0.0]

    def gradient(
        self, reg: "EntityRegistry", params: "ParameterContext"
    ) -> Dict[int, List[Tuple[float, float]]]:
        if self.center is not None:
            # P1+P2 - 2C = 0. Safe to use dict literal as keys are distinct.
            return {
                self.p1: [(1, 0), (0, 1)],
                self.p2: [(1, 0), (0, 1)],
                self.center: [(-2, 0), (0, -2)],
            }
        elif self.axis is not None:
            line = reg.get_entity(self.axis)
            if not isinstance(line, Line):
                return {}

            l1 = reg.get_point(line.p1_idx)
            l2 = reg.get_point(line.p2_idx)
            dxl = l2.x - l1.x
            dyl = l2.y - l1.y
            pt1 = reg.get_point(self.p1)
            pt2 = reg.get_point(self.p2)
            dxp = pt2.x - pt1.x
            dyp = pt2.y - pt1.y
            mx = (pt1.x + pt2.x) * 0.5
            my = (pt1.y + pt2.y) * 0.5

            grad: Dict[int, List[Tuple[float, float]]] = {}
            num_residuals = 2

            def add(pid, row, gx, gy):
                if pid not in grad:
                    grad[pid] = [(0.0, 0.0)] * num_residuals
                # Tuples are immutable, so we must replace it
                px, py = grad[pid][row]
                grad[pid][row] = (px + gx, py + gy)

            # Row 0: Perpendicularity: dxp*dxl + dyp*dyl
            add(self.p1, 0, -dxl, -dyl)
            add(self.p2, 0, dxl, dyl)
            add(line.p1_idx, 0, -dxp, -dyp)
            add(line.p2_idx, 0, dxp, dyp)

            # Row 1: Collinearity: (mx - l1x)*dyl - (my - l1y)*dxl
            add(self.p1, 1, 0.5 * dyl, -0.5 * dxl)
            add(self.p2, 1, 0.5 * dyl, -0.5 * dxl)
            add(line.p1_idx, 1, my - l1.y - dyl, dxl - (mx - l1.x))
            add(line.p2_idx, 1, -(my - l1.y), mx - l1.x)

            return grad

        return {}

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
        if not (p1 and p2):
            return False
        s1 = to_screen((p1.x, p1.y))
        s2 = to_screen((p2.x, p2.y))
        mx = (s1[0] + s2[0]) / 2.0
        my = (s1[1] + s2[1]) / 2.0
        angle = math.atan2(s2[1] - s1[1], s2[0] - s1[0])
        offset = 12.0
        lx = mx - offset * math.cos(angle)
        ly = my - offset * math.sin(angle)
        rx = mx + offset * math.cos(angle)
        ry = my + offset * math.sin(angle)
        if math.hypot(sx - lx, sy - ly) < threshold:
            return True
        if math.hypot(sx - rx, sy - ry) < threshold:
            return True
        return False
