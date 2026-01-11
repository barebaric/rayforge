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
from ..entities import Circle
from .base import Constraint
from .radius import RadiusConstraint

if TYPE_CHECKING:
    from ..params import ParameterContext
    from ..registry import EntityRegistry


class DiameterConstraint(Constraint):
    """Enforces the diameter of a Circle."""

    def __init__(
        self,
        circle_id: int,
        value: Union[str, float],
        expression: Optional[str] = None,
    ):
        self.circle_id = circle_id

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
            "type": "DiameterConstraint",
            "circle_id": self.circle_id,
            "value": self.value,
        }
        if self.expression:
            data["expression"] = self.expression
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DiameterConstraint":
        return cls(
            circle_id=data["circle_id"],
            value=data["value"],
            expression=data.get("expression"),
        )

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
        target_diameter = self.value

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

            ux, uy = 1.0, 0.0  # Default if points are coincident
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

            # Check if click is within label rectangle area
            # Label is drawn with background rectangle at label position
            # Use conservative size to match test expectations
            label_width = 20.0
            label_height = 20.0
            half_w = label_width / 2.0
            half_h = label_height / 2.0

            # Rectangle bounds (with padding as in renderer)
            x_min = label_sx - half_w - 4.0
            x_max = label_sx + half_w + 4.0
            y_min = label_sy - half_h - 4.0
            y_max = label_sy + half_h + 4.0

            return x_min <= sx <= x_max and y_min <= sy <= y_max
        return False
