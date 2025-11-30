from __future__ import annotations
import math
from typing import (
    Union,
    Tuple,
    Dict,
    Any,
    List,
    Callable,
    TYPE_CHECKING,
    cast,
)
from .base import Constraint
from ..entities import Line, Arc, Circle

if TYPE_CHECKING:
    from ..entities import EntityRegistry
    from ..params import ParameterContext


class EqualLengthConstraint(Constraint):
    """
    Enforces that all entities in a set have the same characteristic length.
    - Line: Length
    - Arc/Circle: Radius
    """

    def __init__(self, entity_ids: List[int]):
        self.entity_ids = entity_ids

    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": "EqualLengthConstraint",
            "entity_ids": self.entity_ids,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "EqualLengthConstraint":
        return cls(entity_ids=data["entity_ids"])

    def constrains_radius(
        self, registry: "EntityRegistry", entity_id: int
    ) -> bool:
        return entity_id in self.entity_ids

    def _get_length(self, entity, reg: "EntityRegistry") -> float:
        if isinstance(entity, Line):
            p1 = reg.get_point(entity.p1_idx)
            p2 = reg.get_point(entity.p2_idx)
            return math.hypot(p2.x - p1.x, p2.y - p1.y)
        elif isinstance(entity, Arc):
            c = reg.get_point(entity.center_idx)
            s = reg.get_point(entity.start_idx)
            return math.hypot(s.x - c.x, s.y - c.y)
        elif isinstance(entity, Circle):
            c = reg.get_point(entity.center_idx)
            r = reg.get_point(entity.radius_pt_idx)
            return math.hypot(r.x - c.x, r.y - c.y)
        return 0.0

    def error(
        self, reg: "EntityRegistry", params: "ParameterContext"
    ) -> List[float]:
        if len(self.entity_ids) < 2:
            return []

        entities = [reg.get_entity(eid) for eid in self.entity_ids]
        if any(e is None for e in entities):
            return []

        # All lengths should equal the length of the first entity.
        base_len = self._get_length(entities[0], reg)
        errors = []
        for i in range(1, len(entities)):
            other_len = self._get_length(entities[i], reg)
            errors.append(other_len - base_len)
        return errors

    def gradient(
        self, reg: "EntityRegistry", params: "ParameterContext"
    ) -> Dict[int, List[Tuple[float, float]]]:
        if len(self.entity_ids) < 2:
            return {}

        entities = [
            cast(Union[Line, Arc, Circle], reg.get_entity(eid))
            for eid in self.entity_ids
        ]
        grad: Dict[int, List[Tuple[float, float]]] = {}
        num_residuals = len(entities) - 1

        # Helper to get points defining length
        def get_pts(ent):
            if isinstance(ent, Line):
                return ent.p1_idx, ent.p2_idx
            elif isinstance(ent, Arc):
                return ent.center_idx, ent.start_idx
            elif isinstance(ent, Circle):
                return ent.center_idx, ent.radius_pt_idx
            return -1, -1

        # Base entity properties (index 0)
        p0a, p0b = get_pts(entities[0])
        pt0a = reg.get_point(p0a)
        pt0b = reg.get_point(p0b)
        dx0 = pt0b.x - pt0a.x
        dy0 = pt0b.y - pt0a.y
        len0 = math.hypot(dx0, dy0)

        # Precompute unit vectors for base entity
        u0x, u0y = 0.0, 0.0
        if len0 > 1e-9:
            u0x, u0y = dx0 / len0, dy0 / len0

        def add_grad(pid, r_idx, gx, gy):
            if pid not in grad:
                grad[pid] = [(0.0, 0.0)] * num_residuals
            # Copy list to modify specific row
            curr = list(grad[pid])
            ox, oy = curr[r_idx]
            curr[r_idx] = (ox + gx, oy + gy)
            grad[pid] = curr

        for row in range(num_residuals):
            # Base entity derivs (negative)
            # L_i - L_0
            # d/dP0a = -dLo/dP0a = -(-u0) = u0
            # d/dP0b = -dLo/dP0b = -(u0) = -u0
            add_grad(p0a, row, u0x, u0y)
            add_grad(p0b, row, -u0x, -u0y)

            # Current entity derivs (positive)
            ent = entities[row + 1]
            pa, pb = get_pts(ent)
            pta, ptb = reg.get_point(pa), reg.get_point(pb)
            dx, dy = ptb.x - pta.x, ptb.y - pta.y
            dist = math.hypot(dx, dy)
            if dist > 1e-9:
                ux, uy = dx / dist, dy / dist
                # dLi/dPa = -u
                # dLi/dPb = u
                add_grad(pa, row, -ux, -uy)
                add_grad(pb, row, ux, uy)

        return grad

    def _get_symbol_pos(
        self,
        entity,
        reg: "EntityRegistry",
        to_screen: Callable[[Tuple[float, float]], Tuple[float, float]],
        element: Any,
    ):
        """Calculates screen pos for an equality symbol on an entity."""
        # 1. Get anchor point (mid_x, mid_y) and normal_angle in MODEL space
        mid_x, mid_y, normal_angle = 0.0, 0.0, 0.0

        if isinstance(entity, Line):
            p1 = reg.get_point(entity.p1_idx)
            p2 = reg.get_point(entity.p2_idx)
            mid_x = (p1.x + p2.x) / 2.0
            mid_y = (p1.y + p2.y) / 2.0
            tangent_angle = math.atan2(p2.y - p1.y, p2.x - p1.x)
            normal_angle = tangent_angle - (math.pi / 2.0)
        elif isinstance(entity, (Arc, Circle)):
            midpoint = entity.get_midpoint(reg)
            if not midpoint:
                return None
            mid_x, mid_y = midpoint
            center = reg.get_point(entity.center_idx)
            normal_angle = math.atan2(mid_y - center.y, mid_x - center.x)

        scale = 1.0
        if element.canvas and hasattr(element.canvas, "get_view_scale"):
            scale, _ = element.canvas.get_view_scale()
            scale = max(scale, 1e-9)
        offset_dist_model = 15.0 / scale
        final_x = mid_x + offset_dist_model * math.cos(normal_angle)
        final_y = mid_y + offset_dist_model * math.sin(normal_angle)
        return to_screen((final_x, final_y))

    def is_hit(
        self,
        sx: float,
        sy: float,
        reg: "EntityRegistry",
        to_screen: Callable[[Tuple[float, float]], Tuple[float, float]],
        element: Any,
        threshold: float,
    ) -> bool:
        for entity_id in self.entity_ids:
            entity = reg.get_entity(entity_id)
            if not entity:
                continue
            pos = self._get_symbol_pos(entity, reg, to_screen, element)
            if pos:
                esx, esy = pos
                if math.hypot(sx - esx, sy - esy) < 15:
                    return True
        return False
