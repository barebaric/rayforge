from __future__ import annotations
import math
from typing import (
    Tuple,
    Dict,
    Any,
    List,
    TYPE_CHECKING,
    Callable,
    Optional,
)
from .base import Constraint

if TYPE_CHECKING:
    from ..params import ParameterContext
    from ..registry import EntityRegistry


class AspectRatioConstraint(Constraint):
    """Enforces that distance(p1, p2) / distance(p3, p4) equals ratio."""

    def __init__(
        self,
        p1: int,
        p2: int,
        p3: int,
        p4: int,
        ratio: float,
        user_visible: bool = True,
    ):
        super().__init__(user_visible=user_visible)
        self.p1 = p1
        self.p2 = p2
        self.p3 = p3
        self.p4 = p4
        self.ratio = ratio

    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": "AspectRatioConstraint",
            "p1": self.p1,
            "p2": self.p2,
            "p3": self.p3,
            "p4": self.p4,
            "ratio": self.ratio,
            "user_visible": self.user_visible,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AspectRatioConstraint":
        return cls(
            p1=data["p1"],
            p2=data["p2"],
            p3=data["p3"],
            p4=data["p4"],
            ratio=data["ratio"],
            user_visible=data.get("user_visible", True),
        )

    def error(
        self, reg: "EntityRegistry", params: "ParameterContext"
    ) -> float:
        pt1 = reg.get_point(self.p1)
        pt2 = reg.get_point(self.p2)
        dist1 = math.hypot(pt2.x - pt1.x, pt2.y - pt1.y)

        pt3 = reg.get_point(self.p3)
        pt4 = reg.get_point(self.p4)
        dist2 = math.hypot(pt4.x - pt3.x, pt4.y - pt3.y)

        if dist2 < 1e-9:
            return dist1
        return dist1 - dist2 * self.ratio

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
            add(self.p3, self.ratio * u2x, self.ratio * u2y)
            add(self.p4, -self.ratio * u2x, -self.ratio * u2y)

        return grad

    def _get_icon_pos(
        self,
        reg: "EntityRegistry",
        to_screen: Callable[[Tuple[float, float]], Tuple[float, float]],
    ) -> Optional[Tuple[float, float]]:
        """Calculates the screen position of the constraint icon."""
        p_ids1 = {self.p1, self.p2}
        p_ids2 = {self.p3, self.p4}
        junction_ids = p_ids1.intersection(p_ids2)

        if len(junction_ids) == 1:
            # Position near the common junction point
            junction_id = junction_ids.pop()
            p_junc = reg.get_point(junction_id)

            p_other1_id = self.p1 if self.p2 == junction_id else self.p2
            p_other2_id = self.p3 if self.p4 == junction_id else self.p4
            p_other1 = reg.get_point(p_other1_id)
            p_other2 = reg.get_point(p_other2_id)

            # Calculate vectors in screen space for a consistent offset
            s_junc = to_screen((p_junc.x, p_junc.y))
            s_other1 = to_screen((p_other1.x, p_other1.y))
            s_other2 = to_screen((p_other2.x, p_other2.y))

            sv1 = (s_other1[0] - s_junc[0], s_other1[1] - s_junc[1])
            sv2 = (s_other2[0] - s_junc[0], s_other2[1] - s_junc[1])
            slen1 = math.hypot(sv1[0], sv1[1])
            slen2 = math.hypot(sv2[0], sv2[1])

            if slen1 < 1e-9 or slen2 < 1e-9:
                return s_junc

            su1 = (sv1[0] / slen1, sv1[1] / slen1)
            su2 = (sv2[0] / slen2, sv2[1] / slen2)

            # External angle bisector direction in screen space
            s_bisector = (-(su1[0] + su2[0]), -(su1[1] + su2[1]))
            len_sb = math.hypot(s_bisector[0], s_bisector[1])

            if len_sb < 1e-9:
                # Fallback for parallel/opposite vectors
                s_bisector = (-su1[1], su1[0])
                len_sb = 1.0

            su_bisector = (s_bisector[0] / len_sb, s_bisector[1] / len_sb)

            offset = 18.0  # screen pixels
            return (
                s_junc[0] + offset * su_bisector[0],
                s_junc[1] + offset * su_bisector[1],
            )
        else:
            # Fallback for non-adjoining lines: position at center of midpoints
            p1 = reg.get_point(self.p1)
            p2 = reg.get_point(self.p2)
            p3 = reg.get_point(self.p3)
            p4 = reg.get_point(self.p4)
            m1_x, m1_y = (p1.x + p2.x) / 2.0, (p1.y + p2.y) / 2.0
            m2_x, m2_y = (p3.x + p4.x) / 2.0, (p3.y + p4.y) / 2.0
            center_mx, center_my = (m1_x + m2_x) / 2.0, (m1_y + m2_y) / 2.0
            return to_screen((center_mx, center_my))

    def is_hit(
        self,
        sx: float,
        sy: float,
        reg: "EntityRegistry",
        to_screen: Callable[[Tuple[float, float]], Tuple[float, float]],
        element: Any,
        threshold: float,
    ) -> bool:
        icon_pos = self._get_icon_pos(reg, to_screen)
        if icon_pos:
            cx, cy = icon_pos
            return math.hypot(sx - cx, sy - cy) < threshold
        return False
