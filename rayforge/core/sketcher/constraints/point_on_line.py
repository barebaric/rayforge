from __future__ import annotations
import math
import cairo
from typing import (
    Tuple,
    Dict,
    Any,
    List,
    Callable,
    TYPE_CHECKING,
)
from .base import Constraint
from ..entities import Line, Arc, Circle

if TYPE_CHECKING:
    from ..params import ParameterContext
    from ..registry import EntityRegistry


class PointOnLineConstraint(Constraint):
    """Enforces a point lies on the infinite geometry of a shape."""

    def __init__(
        self, point_id: int, shape_id: int, user_visible: bool = True
    ):
        super().__init__(user_visible=user_visible)
        self.point_id = point_id
        self.shape_id = shape_id

    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": "PointOnLineConstraint",
            "point_id": self.point_id,
            "shape_id": self.shape_id,
            "user_visible": self.user_visible,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PointOnLineConstraint":
        return cls(
            point_id=data["point_id"],
            shape_id=data["shape_id"],
            user_visible=data.get("user_visible", True),
        )

    def constrains_radius(
        self, registry: "EntityRegistry", entity_id: int
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

    def error(
        self, reg: "EntityRegistry", params: "ParameterContext"
    ) -> float:
        pt = reg.get_point(self.point_id)
        shape = reg.get_entity(self.shape_id)

        if shape is None:
            return 0.0

        if isinstance(shape, Line):
            l1 = reg.get_point(shape.p1_idx)
            l2 = reg.get_point(shape.p2_idx)
            dx = l2.x - l1.x
            dy = l2.y - l1.y
            length = math.hypot(dx, dy)
            if length < 1e-9:
                return math.hypot(pt.x - l1.x, pt.y - l1.y)

            # Cross product (signed area) divided by length = distance
            cross = (l2.x - l1.x) * (pt.y - l1.y) - (pt.x - l1.x) * (
                l2.y - l1.y
            )
            return cross / length

        elif isinstance(shape, (Arc, Circle)):
            center = reg.get_point(shape.center_idx)
            radius = 0.0
            if isinstance(shape, Arc):
                start = reg.get_point(shape.start_idx)
                radius = math.hypot(start.x - center.x, start.y - center.y)
            elif isinstance(shape, Circle):
                radius_pt = reg.get_point(shape.radius_pt_idx)
                radius = math.hypot(
                    radius_pt.x - center.x, radius_pt.y - center.y
                )

            dist_to_point = math.hypot(pt.x - center.x, pt.y - center.y)
            return dist_to_point - radius

        return 0.0

    def gradient(
        self, reg: "EntityRegistry", params: "ParameterContext"
    ) -> Dict[int, List[Tuple[float, float]]]:
        pt = reg.get_point(self.point_id)
        shape = reg.get_entity(self.shape_id)

        if shape is None:
            return {}

        grad: Dict[int, List[Tuple[float, float]]] = {}

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
        reg: "EntityRegistry",
        to_screen: Callable[[Tuple[float, float]], Tuple[float, float]],
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
        ctx: "cairo.Context",
        registry: "EntityRegistry",
        to_screen: Callable[[Tuple[float, float]], Tuple[float, float]],
        is_selected: bool = False,
        is_hovered: bool = False,
        point_radius: float = 5.0,
    ) -> None:
        from ..entities import TextBoxEntity

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

        self._set_color(ctx, is_hovered)
        ctx.stroke()
        ctx.restore()
