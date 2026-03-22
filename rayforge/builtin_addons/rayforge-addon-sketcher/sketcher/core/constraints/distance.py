from __future__ import annotations
import math
from typing import (
    Union,
    Dict,
    Any,
    List,
    Optional,
    Callable,
    TYPE_CHECKING,
)
from gettext import gettext as _
from rayforge.core.geo import Point
from rayforge.core.geo.primitives import find_closest_point_on_line_segment
from ..types import EntityID
from .base import Constraint, ConstraintStatus

if TYPE_CHECKING:
    import cairo
    from ..params import ParameterContext
    from ..registry import EntityRegistry
    from ..selection import SketchSelection
    from ..sketch import Sketch


class DistanceConstraint(Constraint):
    """Enforces distance between two points."""

    def __init__(
        self,
        p1: EntityID,
        p2: EntityID,
        value: Union[str, float],
        expression: Optional[str] = None,
        user_visible: bool = True,
    ):
        super().__init__(user_visible=user_visible)
        self.p1: EntityID = p1
        self.p2: EntityID = p2

        if expression is not None:
            self.expression = expression
            self.value = float(value)
        elif isinstance(value, str):
            self.expression = value
            self.value = 0.0
        else:
            self.expression = None
            self.value = float(value)

    @classmethod
    def get_type_key(cls) -> str:
        return "dist"

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
        from ..entities import Line

        lines = [
            sketch.registry.get_entity(eid)
            for eid in selection.entity_ids
            if isinstance(sketch.registry.get_entity(eid), Line)
        ]
        return len(lines) == 1 and len(lines) == len(selection.entity_ids)

    @staticmethod
    def get_type_name() -> str:
        """Returns to human-readable name of this constraint type."""
        return _("Distance")

    def get_title(self) -> str:
        """Returns a human-readable title for this constraint."""
        return f"{self.get_type_name()} {self._format_value()}"

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

    def targets_segment(
        self, p1: EntityID, p2: EntityID, entity_id: Optional[EntityID]
    ) -> bool:
        return {self.p1, self.p2} == {p1, p2}

    def to_dict(self) -> Dict[str, Any]:
        data = {
            "type": "DistanceConstraint",
            "p1": self.p1,
            "p2": self.p2,
            "value": self.value,
            "user_visible": self.user_visible,
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
            user_visible=data.get("user_visible", True),
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
    ) -> Dict[EntityID, List[Point]]:
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
        to_screen: Callable[[Point], Point],
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
        to_screen: Callable[[Point], Point],
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

    def draw(
        self,
        ctx: "cairo.Context",
        registry: "EntityRegistry",
        to_screen: Callable[[Point], Point],
        is_selected: bool = False,
        is_hovered: bool = False,
        point_radius: float = 5.0,
    ) -> None:
        from ..entities import Line

        try:
            p1 = registry.get_point(self.p1)
            p2 = registry.get_point(self.p2)
        except IndexError:
            return

        s1 = to_screen((p1.x, p1.y))
        s2 = to_screen((p2.x, p2.y))
        mx, my = (s1[0] + s2[0]) / 2, (s1[1] + s2[1]) / 2

        label = self._format_value()
        ext = ctx.text_extents(label)

        ctx.save()
        # Set background color based on selection, hover, and status
        if is_selected:
            ctx.set_source_rgba(0.2, 0.6, 1.0, 0.4)  # Blue selection
        elif is_hovered:
            ctx.set_source_rgba(1.0, 0.95, 0.85, 0.9)  # Light yellow hover
        elif self.status == ConstraintStatus.CONFLICTING:
            ctx.set_source_rgba(1.0, 0.6, 0.6, 0.9)  # Red background
        elif self.status == ConstraintStatus.ERROR:
            ctx.set_source_rgba(1.0, 0.8, 0.8, 0.9)  # Light red background
        elif self.status == ConstraintStatus.EXPRESSION_BASED:
            ctx.set_source_rgba(1.0, 0.9, 0.7, 0.9)  # Light orange background
        else:  # VALID
            ctx.set_source_rgba(1, 1, 1, 0.8)  # Default white background

        # Draw label background
        bg_x = mx - ext.width / 2 - 4
        bg_y = my - ext.height / 2 - 4
        ctx.rectangle(bg_x, bg_y, ext.width + 8, ext.height + 8)
        ctx.fill()
        ctx.new_path()

        # Set text color based on status
        if self.status in (
            ConstraintStatus.ERROR,
            ConstraintStatus.CONFLICTING,
        ):
            ctx.set_source_rgb(0.8, 0.0, 0.0)  # Red text for error/conflict
        else:
            ctx.set_source_rgb(0, 0, 0.5)  # Dark blue otherwise

        ctx.move_to(mx - ext.width / 2, my + ext.height / 2 - 2)
        ctx.show_text(label)
        ctx.new_path()

        # Draw Dash Line - only if no solid line entity connects these points
        has_geometry = False
        entities = registry.entities or []
        for entity in entities:
            if isinstance(entity, Line):
                if {entity.p1_idx, entity.p2_idx} == {self.p1, self.p2}:
                    has_geometry = True
                    break

        if not has_geometry:
            ctx.set_line_width(1)
            ctx.set_dash([4, 4])
            ctx.move_to(s1[0], s1[1])
            ctx.line_to(s2[0], s2[1])

            if self.status == ConstraintStatus.CONFLICTING:
                self._draw_conflict_underlay(ctx)

            self._set_color(ctx, is_hovered)
            ctx.stroke()

        ctx.restore()
