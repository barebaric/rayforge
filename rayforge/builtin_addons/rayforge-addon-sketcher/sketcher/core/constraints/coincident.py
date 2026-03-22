# constraints/coincident.py

from __future__ import annotations
import math
from gettext import gettext as _
from typing import Dict, Any, List, Callable, Optional, TYPE_CHECKING
from rayforge.core.geo import Point
from .base import Constraint, ConstraintStatus
from ..types import EntityID

if TYPE_CHECKING:
    import cairo
    from ..params import ParameterContext
    from ..registry import EntityRegistry
    from ..selection import SketchSelection
    from ..sketch import Sketch


class CoincidentConstraint(Constraint):
    """Enforces two points are at the same location."""

    def __init__(self, p1: EntityID, p2: EntityID, user_visible: bool = True):
        super().__init__(user_visible=user_visible)
        self.p1: EntityID = p1
        self.p2: EntityID = p2

    @classmethod
    def get_type_key(cls) -> str:
        return "coincident"

    @classmethod
    def can_apply_to(
        cls, selection: "SketchSelection", sketch: Optional["Sketch"] = None
    ) -> bool:
        return len(selection.point_ids) == 2 and not selection.entity_ids

    @staticmethod
    def get_type_name() -> str:
        """Returns to human-readable name of this constraint type."""
        return _("Coincident")

    def get_title(self) -> str:
        """Returns a human-readable title for this constraint."""
        return self.get_type_name()

    def get_subtitle(self, registry: "EntityRegistry") -> str:
        """Returns a human-readable subtitle describing constrained points."""
        p1 = registry.get_point(self.p1)
        if p1:
            return _("At {}").format(self._format_coord(p1.x, p1.y))
        return ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": "CoincidentConstraint",
            "p1": self.p1,
            "p2": self.p2,
            "user_visible": self.user_visible,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CoincidentConstraint":
        return cls(
            p1=data["p1"],
            p2=data["p2"],
            user_visible=data.get("user_visible", True),
        )

    def error(
        self, reg: "EntityRegistry", params: "ParameterContext"
    ) -> Point:
        pt1 = reg.get_point(self.p1)
        pt2 = reg.get_point(self.p2)
        return (pt1.x - pt2.x, pt1.y - pt2.y)

    def gradient(
        self, reg: "EntityRegistry", params: "ParameterContext"
    ) -> Dict[EntityID, List[Point]]:
        return {
            self.p1: [(1.0, 0.0), (0.0, 1.0)],
            self.p2: [(-1.0, 0.0), (0.0, -1.0)],
        }

    def is_hit(
        self,
        sx: float,
        sy: float,
        reg: "EntityRegistry",
        to_screen: Callable[[Point], Point],
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

    def draw(
        self,
        ctx: "cairo.Context",
        registry: "EntityRegistry",
        to_screen: Callable[[Point], Point],
        is_selected: bool = False,
        is_hovered: bool = False,
        point_radius: float = 5.0,
    ) -> None:
        # Determine which point to draw on. Prefer the non-origin point if
        # one is the origin, to avoid clutter on the origin.
        # We don't have access to sketch.origin_id directly here without the
        # sketch object, but usually point 0 is origin.
        # Assuming p1 or p2 exists in registry.
        try:
            p = registry.get_point(self.p1)
        except IndexError:
            try:
                p = registry.get_point(self.p2)
            except IndexError:
                return

        # Heuristic: if p1 is fixed (likely origin), draw on p2.
        if p.fixed and not registry.get_point(self.p2).fixed:
            p = registry.get_point(self.p2)

        sx, sy = to_screen((p.x, p.y))

        ctx.save()
        ctx.set_line_width(1.5)

        radius = point_radius + 4
        ctx.new_sub_path()
        ctx.arc(sx, sy, radius, 0, 2 * math.pi)

        if is_selected:
            self._draw_selection_underlay(ctx)

        if self.status == ConstraintStatus.CONFLICTING:
            self._draw_conflict_underlay(ctx)

        self._set_color(ctx, is_hovered)
        ctx.stroke()
        ctx.restore()

    def get_draggable_point(self) -> EntityID:
        """Returns p1 as the draggable point for coincident constraints."""
        return self.p1
