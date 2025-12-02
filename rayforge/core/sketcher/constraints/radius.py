from __future__ import annotations
import math
from typing import (
    Union,
    Tuple,
    Dict,
    Any,
    List,
    Optional,
    Callable,
    TYPE_CHECKING,
)
from ..entities import Arc, Circle
from .base import Constraint

if TYPE_CHECKING:
    from ..entities import EntityRegistry
    from ..params import ParameterContext


class RadiusConstraint(Constraint):
    """Enforces radius of an Arc or Circle."""

    def __init__(
        self,
        entity_id: int,
        value: Union[str, float],
        expression: Optional[str] = None,
    ):
        self.entity_id = entity_id

        if expression is not None:
            self.expression = expression
            self.value = float(value)
        elif isinstance(value, str):
            self.expression = value
            self.value = 0.0
        else:
            self.expression = None
            self.value = float(value)

    def to_dict(self) -> Dict[str, Any]:
        data = {
            "type": "RadiusConstraint",
            "entity_id": self.entity_id,
            "value": self.value,
        }
        if self.expression:
            data["expression"] = self.expression
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "RadiusConstraint":
        return cls(
            entity_id=data["entity_id"],
            value=data["value"],
            expression=data.get("expression"),
        )

    def constrains_radius(
        self, registry: "EntityRegistry", entity_id: int
    ) -> bool:
        return self.entity_id == entity_id

    def error(
        self, reg: "EntityRegistry", params: "ParameterContext"
    ) -> float:
        entity = reg.get_entity(self.entity_id)
        if entity is None:
            return 0.0

        target = self.value
        curr_r = 0.0

        if isinstance(entity, Arc):
            center = reg.get_point(entity.center_idx)
            start = reg.get_point(entity.start_idx)
            curr_r = math.hypot(start.x - center.x, start.y - center.y)
        elif isinstance(entity, Circle):
            center = reg.get_point(entity.center_idx)
            radius_pt = reg.get_point(entity.radius_pt_idx)
            curr_r = math.hypot(radius_pt.x - center.x, radius_pt.y - center.y)
        else:
            return 0.0

        return curr_r - target

    def gradient(
        self, reg: "EntityRegistry", params: "ParameterContext"
    ) -> Dict[int, List[Tuple[float, float]]]:
        entity = reg.get_entity(self.entity_id)

        # Type narrowing for Pylance
        if not isinstance(entity, (Arc, Circle)):
            return {}

        center_idx = entity.center_idx
        c = reg.get_point(center_idx)
        p, pt_idx = None, -1

        if isinstance(entity, Arc):
            pt_idx = entity.start_idx
            p = reg.get_point(pt_idx)
        else:  # Circle
            pt_idx = entity.radius_pt_idx
            p = reg.get_point(pt_idx)

        if c and p:
            dx, dy = p.x - c.x, p.y - c.y
            dist = math.hypot(dx, dy)

            ux, uy = 1.0, 0.0  # Default if points are coincident
            if dist > 1e-9:
                ux, uy = dx / dist, dy / dist

            return {
                pt_idx: [(ux, uy)],
                center_idx: [(-ux, -uy)],
            }
        return {}

    def get_label_pos(
        self,
        reg: "EntityRegistry",
        to_screen: Callable[[Tuple[float, float]], Tuple[float, float]],
        element: Any,
    ):
        """Calculates screen position for Radius/Diameter constraint labels."""
        entity = reg.get_entity(self.entity_id)
        if not isinstance(entity, (Arc, Circle)):
            return None

        center = reg.get_point(entity.center_idx)
        if not center:
            return None

        radius, mid_angle = 0.0, 0.0

        if isinstance(entity, Arc):
            start = reg.get_point(entity.start_idx)
            if not start:
                return None
            radius = math.hypot(start.x - center.x, start.y - center.y)
            midpoint = entity.get_midpoint(reg)
            if not midpoint:
                return None
            mid_angle = math.atan2(
                midpoint[1] - center.y, midpoint[0] - center.x
            )

        elif isinstance(entity, Circle):
            radius_pt = reg.get_point(entity.radius_pt_idx)
            if not radius_pt:
                return None
            radius = math.hypot(radius_pt.x - center.x, radius_pt.y - center.y)
            mid_angle = math.atan2(
                radius_pt.y - center.y, radius_pt.x - center.x
            )

        if radius == 0.0:
            return None

        scale = 1.0
        if element.canvas and hasattr(element.canvas, "get_view_scale"):
            scale_x, _ = element.canvas.get_view_scale()
            scale = scale_x if scale_x > 1e-9 else 1.0

        label_dist = radius + 20 / scale
        label_mx = center.x + label_dist * math.cos(mid_angle)
        label_my = center.y + label_dist * math.sin(mid_angle)
        label_sx, label_sy = to_screen((label_mx, label_my))

        # Position on the arc for the leader line
        arc_mid_mx = center.x + radius * math.cos(mid_angle)
        arc_mid_my = center.y + radius * math.sin(mid_angle)
        arc_mid_sx, arc_mid_sy = to_screen((arc_mid_mx, arc_mid_my))

        return label_sx, label_sy, arc_mid_sx, arc_mid_sy

    def is_hit(
        self,
        sx: float,
        sy: float,
        reg: "EntityRegistry",
        to_screen: Callable[[Tuple[float, float]], Tuple[float, float]],
        element: Any,
        threshold: float,
    ) -> bool:
        pos_data = self.get_label_pos(reg, to_screen, element)
        if pos_data:
            label_sx, label_sy, _, _ = pos_data
            return math.hypot(sx - label_sx, sy - label_sy) < 15
        return False
