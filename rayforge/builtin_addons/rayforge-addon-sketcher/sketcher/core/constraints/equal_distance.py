# constraints/equal_distance.py

from __future__ import annotations
import math
from typing import Optional, Dict, Any, List, TYPE_CHECKING
from gettext import gettext as _
from rayforge.core.geo import Point
from ..entities import Line, Arc, Circle
from .base import Constraint
from ..types import EntityID

if TYPE_CHECKING:
    from ..params import ParameterContext
    from ..registry import EntityRegistry
    from ..selection import SketchSelection
    from ..sketch import Sketch


class EqualDistanceConstraint(Constraint):
    """Enforces that distance(p1, p2) equals distance(p3, p4)."""

    def __init__(
        self,
        p1: EntityID,
        p2: EntityID,
        p3: EntityID,
        p4: EntityID,
        user_visible: bool = True,
    ):
        super().__init__(user_visible=user_visible)
        self.p1: EntityID = p1
        self.p2: EntityID = p2
        self.p3: EntityID = p3
        self.p4: EntityID = p4

    @classmethod
    def can_apply_to(
        cls, selection: "SketchSelection", sketch: Optional["Sketch"] = None
    ) -> bool:
        if selection.point_ids or len(selection.entity_ids) < 2:
            return False
        if sketch is None:
            return False
        for eid in selection.entity_ids:
            entity = sketch.registry.get_entity(eid)
            if not isinstance(entity, (Line, Arc, Circle)):
                return False
        return True

    @staticmethod
    def get_type_name() -> str:
        """Returns to human-readable name of this constraint type."""
        return _("Equal Distance")

    def get_title(self) -> str:
        """Returns a human-readable title for this constraint."""
        return self.get_type_name()

    def get_subtitle(self, registry: "EntityRegistry") -> str:
        """Returns subtitle describing constrained segments."""
        p1 = registry.get_point(self.p1)
        p2 = registry.get_point(self.p2)
        p3 = registry.get_point(self.p3)
        p4 = registry.get_point(self.p4)
        if p1 and p2 and p3 and p4:
            return _("{}-{} and {}-{}").format(
                self._format_coord(p1.x, p1.y),
                self._format_coord(p2.x, p2.y),
                self._format_coord(p3.x, p3.y),
                self._format_coord(p4.x, p4.y),
            )
        return ""

    def targets_segment(
        self, p1: EntityID, p2: EntityID, entity_id: Optional[EntityID]
    ) -> bool:
        target = {p1, p2}
        return target == {self.p1, self.p2} or target == {self.p3, self.p4}

    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": "EqualDistanceConstraint",
            "p1": self.p1,
            "p2": self.p2,
            "p3": self.p3,
            "p4": self.p4,
            "user_visible": self.user_visible,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "EqualDistanceConstraint":
        return cls(
            p1=data["p1"],
            p2=data["p2"],
            p3=data["p3"],
            p4=data["p4"],
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

        return dist1 - dist2

    def gradient(
        self, reg: "EntityRegistry", params: "ParameterContext"
    ) -> Dict[EntityID, List[Point]]:
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

        grad: Dict[EntityID, List[Point]] = {}

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
