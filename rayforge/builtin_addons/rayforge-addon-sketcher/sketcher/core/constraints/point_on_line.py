from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from gettext import gettext as _
from typing import TYPE_CHECKING, Any

import cairo
from raygeo.geo.types import Point

from ..entities import Arc, Circle, Line, TextBoxEntity
from ..types import EntityID
from .base import Constraint, ConstraintStatus

if TYPE_CHECKING:
    from ..entities import Entity
    from ..params import ParameterContext
    from ..registry import EntityRegistry
    from ..selection import SketchSelection
    from ..sketch import Sketch


class PointOnLineConstraint(Constraint):
    """Enforces a point lies on the infinite geometry of a shape."""

    def __init__(
        self, point_id: EntityID, shape_id: EntityID, user_visible: bool = True
    ):
        super().__init__(user_visible=user_visible)
        self.point_id: EntityID = point_id
        self.shape_id: EntityID = shape_id

    @classmethod
    def get_type_key(cls) -> str:
        return "point_on_line"

    @classmethod
    def can_apply_to(
        cls, selection: SketchSelection, sketch: Sketch | None = None
    ) -> bool:
        if len(selection.point_ids) != 1 or len(selection.entity_ids) != 1:
            return False
        entities = selection.resolve_entities(
            sketch.registry if sketch else None
        )
        if entities is None or not cls.applies_to_entities(entities):
            return False
        return (
            selection.point_ids[0] not in entities[0].get_junction_point_ids()
        )

    @classmethod
    def applies_to_entities(cls, entities: Sequence[Entity]) -> bool:
        """The shape must be a Line or radius-bearing entity."""
        return len(entities) == 1 and (
            isinstance(entities[0], Line) or entities[0].is_radius_entity()
        )

    @staticmethod
    def get_type_name() -> str:
        """Returns to human-readable name of this constraint type."""
        return _("Point on Line")

    def get_title(self) -> str:
        """Returns a human-readable title for this constraint."""
        return self.get_type_name()

    def get_subtitle(self, registry: EntityRegistry) -> str:
        """Returns subtitle describing constrained entities."""
        pt = registry.get_point(self.point_id)
        if pt:
            return _("Point at {}").format(self._format_coord(pt.x, pt.y))
        return ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": "PointOnLineConstraint",
            "point_id": self.point_id,
            "shape_id": self.shape_id,
            "user_visible": self.user_visible,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> PointOnLineConstraint:
        return cls(
            point_id=data["point_id"],
            shape_id=data["shape_id"],
            user_visible=data.get("user_visible", True),
        )

    def constrains_radius(
        self, registry: EntityRegistry, entity_id: EntityID
    ) -> bool:
        """
        If this constraint forces a point onto the target entity (circle/arc),
        and that point is itself constrained, then the radius of the entity
        is determined.
        """
        if self.shape_id != entity_id:
            return False

        try:
            pt = registry.get_point(self.point_id)
            return pt.constrained
        except IndexError:
            return False

    def error(self, reg: EntityRegistry, params: ParameterContext) -> float:
        pt = reg.get_point(self.point_id)
        shape = reg.get_entity(self.shape_id)

        if shape is None:
            return 0.0

        try:
            return shape.signed_distance_to(pt, reg)
        except NotImplementedError:
            return 0.0

    def gradient(
        self, reg: EntityRegistry, params: ParameterContext
    ) -> dict[EntityID, list[Point]]:
        pt = reg.get_point(self.point_id)
        shape = reg.get_entity(self.shape_id)

        if shape is None:
            return {}

        grad = {}

        def add_grad(pid, gx, gy):
            if pid not in grad:
                grad[pid] = [(0.0, 0.0)]
            cx, cy = grad[pid][0]
            grad[pid][0] = (cx + gx, cy + gy)

        if isinstance(shape, Line):
            l1 = reg.get_point(shape.p1_idx)
            l2 = reg.get_point(shape.p2_idx)
            dx = l2.x - l1.x
            dy = l2.y - l1.y
            len_sq = dx * dx + dy * dy

            if len_sq < 1e-18:
                d_pt_l1_x = pt.x - l1.x
                d_pt_l1_y = pt.y - l1.y
                dist = math.hypot(d_pt_l1_x, d_pt_l1_y)
                if dist < 1e-9:
                    return {}
                ux = d_pt_l1_x / dist
                uy = d_pt_l1_y / dist
                add_grad(self.point_id, ux, uy)
                add_grad(shape.p1_idx, -ux, -uy)
                return grad

            length = math.sqrt(len_sq)
            inv_len = 1.0 / length
            inv_len_sq = inv_len * inv_len

            # Normal vector to line
            nx = -dy * inv_len
            ny = dx * inv_len

            # Gradient for Point P is the normal vector
            add_grad(self.point_id, nx, ny)

            err = self.error(reg, params)

            # Gradient for L2
            grad_l2_x = (pt.y - l1.y) * inv_len - err * dx * inv_len_sq
            grad_l2_y = -(pt.x - l1.x) * inv_len - err * dy * inv_len_sq
            add_grad(shape.p2_idx, grad_l2_x, grad_l2_y)

            # Gradient for L1 is -(grad_p + grad_l2) due to translation
            # invariance
            grad_l1_x = -nx - grad_l2_x
            grad_l1_y = -ny - grad_l2_y
            add_grad(shape.p1_idx, grad_l1_x, grad_l1_y)
            return grad

        elif isinstance(shape, (Arc, Circle)):
            center = reg.get_point(shape.center_idx)
            dist_pc = math.hypot(pt.x - center.x, pt.y - center.y)

            ux, uy = 0.0, 0.0
            if dist_pc > 1e-9:
                ux = (pt.x - center.x) / dist_pc
                uy = (pt.y - center.y) / dist_pc

            add_grad(self.point_id, ux, uy)

            sp, rad_idx = (None, -1)
            if isinstance(shape, Arc):
                sp = reg.get_point(shape.start_idx)
                rad_idx = shape.start_idx
            else:  # Circle
                sp = reg.get_point(shape.radius_pt_idx)
                rad_idx = shape.radius_pt_idx

            radius = math.hypot(sp.x - center.x, sp.y - center.y)
            rux, ruy = 0.0, 0.0
            if radius > 1e-9:
                rux = (sp.x - center.x) / radius
                ruy = (sp.y - center.y) / radius

            add_grad(shape.center_idx, rux - ux, ruy - uy)
            add_grad(rad_idx, -rux, -ruy)

        return grad

    def is_hit(
        self,
        sx: float,
        sy: float,
        reg: EntityRegistry,
        to_screen: Callable[[Point], Point],
        element: Any,
        threshold: float,
    ) -> bool:
        pt = reg.get_point(self.point_id)
        if pt:
            s_pt = to_screen((pt.x, pt.y))
            return math.hypot(sx - s_pt[0], sy - s_pt[1]) < threshold
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
        # Hide constraint if its point is part of a text box
        text_box_point_ids = set()
        for entity in registry.entities:
            if isinstance(entity, TextBoxEntity):
                text_box_point_ids.update(
                    entity.get_all_frame_point_ids(registry)
                )
        if self.point_id in text_box_point_ids:
            return

        try:
            p = registry.get_point(self.point_id)
        except IndexError:
            return

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
        """Returns the point that lies on the line/shape."""
        return self.point_id
