from __future__ import annotations
import math
import cairo
from typing import Dict, Any, List, Callable, Optional, TYPE_CHECKING
from gettext import gettext as _
from ...geo import Point
from ..entities import Line
from .base import Constraint, ConstraintStatus

if TYPE_CHECKING:
    from ..params import ParameterContext
    from ..registry import EntityRegistry
    from ..selection import SketchSelection
    from ..sketch import Sketch


class VerticalConstraint(Constraint):
    """Enforces two points have the same X coordinate."""

    def __init__(self, p1: int, p2: int, user_visible: bool = True):
        super().__init__(user_visible=user_visible)
        self.p1 = p1
        self.p2 = p2

    @classmethod
    def can_apply_to(
        cls, selection: "SketchSelection", sketch: Optional["Sketch"] = None
    ) -> bool:
        if len(selection.point_ids) == 2 and not selection.entity_ids:
            return True
        if selection.point_ids:
            return False
        if sketch is None:
            return False

        lines = [
            sketch.registry.get_entity(eid)
            for eid in selection.entity_ids
            if isinstance(sketch.registry.get_entity(eid), Line)
        ]
        return len(lines) > 0 and len(lines) == len(selection.entity_ids)

    @staticmethod
    def get_type_name() -> str:
        """Returns to human-readable name of this constraint type."""
        return _("Vertical")

    def get_title(self) -> str:
        """Returns a human-readable title for this constraint."""
        return self.get_type_name()

    def get_subtitle(self, registry: "EntityRegistry") -> str:
        """Returns a subtitle describing the constrained points."""
        p1 = registry.get_point(self.p1)
        p2 = registry.get_point(self.p2)
        if p1 and p2:
            return _("From {} to {}").format(
                self._format_coord(p1.x, p1.y),
                self._format_coord(p2.x, p2.y),
            )
        return ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": "VerticalConstraint",
            "p1": self.p1,
            "p2": self.p2,
            "user_visible": self.user_visible,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "VerticalConstraint":
        return cls(
            p1=data["p1"],
            p2=data["p2"],
            user_visible=data.get("user_visible", True),
        )

    def error(
        self, reg: "EntityRegistry", params: "ParameterContext"
    ) -> float:
        return reg.get_point(self.p1).x - reg.get_point(self.p2).x

    def gradient(
        self, reg: "EntityRegistry", params: "ParameterContext"
    ) -> Dict[int, List[Point]]:
        return {
            self.p1: [(1.0, 0.0)],
            self.p2: [(-1.0, 0.0)],
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
        p1 = reg.get_point(self.p1)
        p2 = reg.get_point(self.p2)
        if p1 and p2:
            s1 = to_screen((p1.x, p1.y))
            s2 = to_screen((p2.x, p2.y))

            t = 0.2
            mx = s1[0] + (s2[0] - s1[0]) * t
            my = s1[1] + (s2[1] - s1[1]) * t
            cx = mx + 10
            cy = my
            return math.hypot(sx - cx, sy - cy) < threshold
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
        try:
            p1 = registry.get_point(self.p1)
            p2 = registry.get_point(self.p2)
        except IndexError:
            return

        s1 = to_screen((p1.x, p1.y))
        s2 = to_screen((p2.x, p2.y))

        t_marker = 0.2
        mx = s1[0] + (s2[0] - s1[0]) * t_marker
        my = s1[1] + (s2[1] - s1[1]) * t_marker

        size = 8
        ctx.save()
        ctx.set_line_width(2)
        ctx.move_to(mx + 10, my - size)
        ctx.line_to(mx + 10, my + size)

        if is_selected:
            self._draw_selection_underlay(ctx)

        if self.status == ConstraintStatus.CONFLICTING:
            self._draw_conflict_underlay(ctx)

        self._set_color(ctx, is_hovered)
        ctx.stroke()
        ctx.restore()
