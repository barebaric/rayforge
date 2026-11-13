from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from gettext import gettext as _
from typing import (
    TYPE_CHECKING,
    Any,
)

from raygeo.geo.shape.line import get_line_segment_closest_point
from raygeo.geo.types import Point
from raygeo.sketch.constraints.distance import (
    DistanceConstraint as SolverDistanceConstraint,
)
from raygeo.sketch.protocols import SolverConstraint

from ..entities import Line
from ..types import EntityID
from .base import Constraint, SolverDrawConstraint

if TYPE_CHECKING:
    import cairo

    from ..entities import Entity
    from ..params import ParameterContext
    from ..registry import EntityRegistry
    from ..selection import SketchSelection
    from ..sketch import Sketch


class DistanceConstraint(SolverDrawConstraint, Constraint):
    """Enforces distance between two points."""

    def __init__(
        self,
        p1: EntityID,
        p2: EntityID,
        value: str | float,
        expression: str | None = None,
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
        cls, selection: SketchSelection, sketch: Sketch | None = None
    ) -> bool:
        if len(selection.point_ids) == 2 and not selection.entity_ids:
            return True
        if selection.point_ids:
            return False
        entities = selection.resolve_entities(
            sketch.registry if sketch else None
        )
        if entities is None or len(entities) != 1:
            return False
        return cls.applies_to_entities(entities)

    @classmethod
    def applies_to_entities(cls, entities: Sequence[Entity]) -> bool:
        """The operand must be a Line."""
        return len(entities) == 1 and isinstance(entities[0], Line)

    @staticmethod
    def get_type_name() -> str:
        """Returns to human-readable name of this constraint type."""
        return _("Distance")

    def get_title(self) -> str:
        """Returns a human-readable title for this constraint."""
        return f"{self.get_type_name()} {self._format_value()}"

    def get_subtitle(self, registry: EntityRegistry) -> str:
        """Returns a subtitle describing the constrained points."""
        p1 = registry.get_point(self.p1)
        p2 = registry.get_point(self.p2)
        if p1 and p2:
            return _("From {} to {}").format(
                self._format_coord(p1.x, p1.y),
                self._format_coord(p2.x, p2.y),
            )
        return ""

    def get_edit_subtitle(self) -> str:
        return _("Enter length or expression.")

    def targets_segment(
        self, p1: EntityID, p2: EntityID, entity_id: EntityID | None
    ) -> bool:
        return {self.p1, self.p2} == {p1, p2}

    def to_dict(self) -> dict[str, Any]:
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
    def from_dict(cls, data: dict[str, Any]) -> DistanceConstraint:
        return cls(
            p1=data["p1"],
            p2=data["p2"],
            value=data["value"],
            expression=data.get("expression"),
            user_visible=data.get("user_visible", True),
        )

    def error(self, reg: EntityRegistry, params: ParameterContext) -> float:
        pt1 = reg.get_point(self.p1)
        pt2 = reg.get_point(self.p2)
        # We use self.value which is cached/updated via update_from_context
        target = self.value
        dist = math.hypot(pt2.x - pt1.x, pt2.y - pt1.y)
        return dist - target

    def gradient(
        self, reg: EntityRegistry, params: ParameterContext
    ) -> dict[EntityID, list[Point]]:
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
        reg: EntityRegistry,
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
        reg: EntityRegistry,
        to_screen: Callable[[Point], Point],
        element: Any,
        threshold: float,
    ) -> bool:
        p1 = reg.get_point(self.p1)
        p2 = reg.get_point(self.p2)

        has_geometry = False
        if p1 and p2:
            entities = reg.entities or []
            for entity in entities:
                if isinstance(entity, Line) and {
                    entity.p1_idx,
                    entity.p2_idx,
                } == {
                    self.p1,
                    self.p2,
                }:
                    has_geometry = True
                    break

            if not has_geometry:
                s1 = to_screen((p1.x, p1.y))
                s2 = to_screen((p2.x, p2.y))

                _, _, dist_sq = get_line_segment_closest_point(s1, s2, sx, sy)

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
        ctx: cairo.Context,
        registry: EntityRegistry,
        to_screen: Callable[[Point], Point],
        is_selected: bool = False,
        is_hovered: bool = False,
        point_radius: float = 5.0,
    ) -> None:
        self._draw_solver_constraint(
            ctx, registry, to_screen, is_selected, is_hovered, point_radius
        )

    def to_solver_constraint(
        self, registry: EntityRegistry
    ) -> SolverConstraint | None:
        """
        Returns the equivalent raygeo solver constraint, with entity
        references resolved to concrete point ids and the dimensional
        value already resolved.
        """
        return SolverDistanceConstraint(self.p1, self.p2, self.value)
