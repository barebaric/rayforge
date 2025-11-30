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
)
from .base import Constraint
from .radius import RadiusConstraint
from ..entities import Circle

if TYPE_CHECKING:
    from ..entities import EntityRegistry
    from ..params import ParameterContext


class DiameterConstraint(Constraint):
    """Enforces the diameter of a Circle."""

    def __init__(self, circle_id: int, diameter: Union[str, float]):
        self.circle_id = circle_id
        self.value = diameter

    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": "DiameterConstraint",
            "circle_id": self.circle_id,
            "value": self.value,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DiameterConstraint":
        return cls(circle_id=data["circle_id"], diameter=data["value"])

    def constrains_radius(
        self, registry: "EntityRegistry", entity_id: int
    ) -> bool:
        return self.circle_id == entity_id

    def error(
        self, reg: "EntityRegistry", params: "ParameterContext"
    ) -> float:
        circle_entity = reg.get_entity(self.circle_id)

        if not isinstance(circle_entity, Circle):
            return 0.0

        center = reg.get_point(circle_entity.center_idx)
        radius_pt = reg.get_point(circle_entity.radius_pt_idx)
        target_diameter = params.evaluate(self.value)

        curr_r = math.hypot(radius_pt.x - center.x, radius_pt.y - center.y)
        return 2 * curr_r - target_diameter

    def gradient(
        self, reg: "EntityRegistry", params: "ParameterContext"
    ) -> Dict[int, List[Tuple[float, float]]]:
        entity = reg.get_entity(self.circle_id)
        if isinstance(entity, Circle):
            c = reg.get_point(entity.center_idx)
            p = reg.get_point(entity.radius_pt_idx)
            dx, dy = p.x - c.x, p.y - c.y
            dist = math.hypot(dx, dy)
            if dist > 1e-9:
                ux, uy = dx / dist, dy / dist
                # Error = 2*r - d. d(2r)/dp = 2 * u
                return {
                    entity.radius_pt_idx: [(2 * ux, 2 * uy)],
                    entity.center_idx: [(-2 * ux, -2 * uy)],
                }
        return {}

    def get_label_pos(
        self,
        reg: "EntityRegistry",
        to_screen: Callable[[Tuple[float, float]], Tuple[float, float]],
        element: Any,
    ):
        # Delegate to RadiusConstraint's logic as it is identical
        temp_radius_constr = RadiusConstraint(self.circle_id, 0)
        return temp_radius_constr.get_label_pos(reg, to_screen, element)

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
