from __future__ import annotations
import math
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
from rayforge.core.geo.primitives import find_closest_point_on_line

if TYPE_CHECKING:
    from ..params import ParameterContext
    from ..registry import EntityRegistry


class TangentConstraint(Constraint):
    """
    Enforces tangency between a Line and an Arc/Circle.
    Logic: Distance from shape center to Line equals shape Radius.
    """

    def __init__(self, line_id: int, shape_id: int):
        self.line_id = line_id
        self.shape_id = shape_id

    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": "TangentConstraint",
            "line_id": self.line_id,
            "shape_id": self.shape_id,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TangentConstraint":
        return cls(line_id=data["line_id"], shape_id=data["shape_id"])

    def error(
        self, reg: "EntityRegistry", params: "ParameterContext"
    ) -> float:
        line = reg.get_entity(self.line_id)
        shape = reg.get_entity(self.shape_id)

        if not isinstance(line, Line) or not isinstance(shape, (Arc, Circle)):
            return 0.0

        center = reg.get_point(shape.center_idx)
        radius = 0.0
        if isinstance(shape, Arc):
            start = reg.get_point(shape.start_idx)
            radius = math.hypot(start.x - center.x, start.y - center.y)
        elif isinstance(shape, Circle):
            radius_pt = reg.get_point(shape.radius_pt_idx)
            radius = math.hypot(radius_pt.x - center.x, radius_pt.y - center.y)

        lp1 = reg.get_point(line.p1_idx)
        lp2 = reg.get_point(line.p2_idx)
        line_dx = lp2.x - lp1.x
        line_dy = lp2.y - lp1.y
        line_len = math.hypot(line_dx, line_dy)

        if line_len < 1e-9:
            dist_to_pt = math.hypot(lp1.x - center.x, lp1.y - center.y)
            return dist_to_pt - radius

        cross_product = line_dx * (center.y - lp1.y) - line_dy * (
            center.x - lp1.x
        )
        dist_val = abs(cross_product) / line_len
        return dist_val - radius

    def gradient(
        self, reg: "EntityRegistry", params: "ParameterContext"
    ) -> Dict[int, List[Tuple[float, float]]]:
        line = reg.get_entity(self.line_id)
        shape = reg.get_entity(self.shape_id)
        grad: Dict[int, List[Tuple[float, float]]] = {}

        if not isinstance(line, Line) or not isinstance(shape, (Arc, Circle)):
            return grad

        def add_grad(pid, gx, gy):
            if pid not in grad:
                grad[pid] = [(0.0, 0.0)]
            cx, cy = grad[pid][0]
            grad[pid][0] = (cx + gx, cy + gy)

        c = reg.get_point(shape.center_idx)

        # Part 1: Gradient of -radius
        p_rad, p_rad_idx = (None, -1)
        if isinstance(shape, Arc):
            p_rad = reg.get_point(shape.start_idx)
            p_rad_idx = shape.start_idx
        elif isinstance(shape, Circle):
            p_rad = reg.get_point(shape.radius_pt_idx)
            p_rad_idx = shape.radius_pt_idx
        if not (c and p_rad):
            return {}

        rad_dx, rad_dy = p_rad.x - c.x, p_rad.y - c.y
        radius = math.hypot(rad_dx, rad_dy)
        rad_ux, rad_uy = (0.0, 0.0)
        if radius > 1e-9:
            rad_ux, rad_uy = rad_dx / radius, rad_dy / radius

        add_grad(p_rad_idx, -rad_ux, -rad_uy)
        add_grad(shape.center_idx, rad_ux, rad_uy)

        # Part 2: Gradient of dist_to_line
        lp1 = reg.get_point(line.p1_idx)
        lp2 = reg.get_point(line.p2_idx)
        line_dx = lp2.x - lp1.x
        line_dy = lp2.y - lp1.y
        len_sq = line_dx * line_dx + line_dy * line_dy

        if len_sq < 1e-18:
            dist_c_l1 = math.hypot(c.x - lp1.x, c.y - lp1.y)
            if dist_c_l1 < 1e-9:
                return grad
            ux = (c.x - lp1.x) / dist_c_l1
            uy = (c.y - lp1.y) / dist_c_l1
            add_grad(shape.center_idx, ux, uy)
            add_grad(line.p1_idx, -ux, -uy)
            return grad

        length = math.sqrt(len_sq)
        inv_len = 1.0 / length
        cross = line_dx * (c.y - lp1.y) - line_dy * (c.x - lp1.x)
        sign = math.copysign(1.0, cross)

        # Start calculating gradient of signed distance from C to Line.
        # This is equivalent to PointOnLineConstraint.gradient for point C.
        nx = -line_dy * inv_len
        ny = line_dx * inv_len
        pol_err = cross * inv_len
        inv_len_sq = inv_len * inv_len

        # grad_pol w.r.t center C
        grad_pol_c_x, grad_pol_c_y = nx, ny

        # grad_pol w.r.t line point L2
        grad_pol_l2_x = (
            c.y - lp1.y
        ) * inv_len - pol_err * line_dx * inv_len_sq
        grad_pol_l2_y = (
            -(c.x - lp1.x) * inv_len - pol_err * line_dy * inv_len_sq
        )

        # grad_pol w.r.t line point L1
        grad_pol_l1_x = -grad_pol_c_x - grad_pol_l2_x
        grad_pol_l1_y = -grad_pol_c_y - grad_pol_l2_y

        # Now combine with sign factor and radius gradients
        add_grad(shape.center_idx, sign * grad_pol_c_x, sign * grad_pol_c_y)
        add_grad(line.p1_idx, sign * grad_pol_l1_x, sign * grad_pol_l1_y)
        add_grad(line.p2_idx, sign * grad_pol_l2_x, sign * grad_pol_l2_y)

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
        line = reg.get_entity(self.line_id)
        shape = reg.get_entity(self.shape_id)

        if not (
            line
            and shape
            and isinstance(line, Line)
            and isinstance(shape, (Arc, Circle))
        ):
            return False

        p1 = reg.get_point(line.p1_idx)
        p2 = reg.get_point(line.p2_idx)
        center = reg.get_point(shape.center_idx)

        if not (p1 and p2 and center):
            return False

        # Find closest point on infinite line from center
        tangent_mx, tangent_my = find_closest_point_on_line(
            (p1.x, p1.y), (p2.x, p2.y), center.x, center.y
        )

        # Convert to Screen Space first
        sx_tangent, sy_tangent = to_screen((tangent_mx, tangent_my))
        sx_center, sy_center = to_screen((center.x, center.y))

        # Calculate angle in screen space
        angle = math.atan2(sy_tangent - sy_center, sx_tangent - sx_center)

        # Apply offset in screen pixels
        offset = 15.0
        symbol_sx = sx_tangent + offset * math.cos(angle)
        symbol_sy = sy_tangent + offset * math.sin(angle)

        return math.hypot(sx - symbol_sx, sy - symbol_sy) < threshold
