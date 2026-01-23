from __future__ import annotations
from typing import (
    Tuple,
    Dict,
    Any,
    List,
    TYPE_CHECKING,
)
from .base import Constraint

if TYPE_CHECKING:
    from ..params import ParameterContext
    from ..registry import EntityRegistry


class ParallelogramConstraint(Constraint):
    """Enforces four points form a parallelogram."""

    def __init__(
        self,
        p_origin: int,
        p_width: int,
        p_height: int,
        p4: int,
        user_visible: bool = False,
    ):
        super().__init__(user_visible=user_visible)
        self.p_origin = p_origin
        self.p_width = p_width
        self.p_height = p_height
        self.p4 = p4

    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": "ParallelogramConstraint",
            "p_origin": self.p_origin,
            "p_width": self.p_width,
            "p_height": self.p_height,
            "p4": self.p4,
            "user_visible": self.user_visible,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ParallelogramConstraint":
        return cls(
            p_origin=data["p_origin"],
            p_width=data["p_width"],
            p_height=data["p_height"],
            p4=data["p4"],
            user_visible=data.get("user_visible", False),
        )

    def error(
        self, reg: "EntityRegistry", params: "ParameterContext"
    ) -> Tuple[float, float]:
        """Returns the difference between vectors (p_width-p_origin) and
        (p4-p_height).
        """
        p_origin = reg.get_point(self.p_origin)
        p_width = reg.get_point(self.p_width)
        p_height = reg.get_point(self.p_height)
        p4 = reg.get_point(self.p4)

        v1_x = p_width.x - p_origin.x
        v1_y = p_width.y - p_origin.y

        v2_x = p4.x - p_height.x
        v2_y = p4.y - p_height.y

        return (v1_x - v2_x, v1_y - v2_y)

    def gradient(
        self, reg: "EntityRegistry", params: "ParameterContext"
    ) -> Dict[int, List[Tuple[float, float]]]:
        """Returns the gradient of the error with respect to each point."""
        return {
            self.p_origin: [(-1.0, 0.0), (0.0, -1.0)],
            self.p_width: [(1.0, 0.0), (0.0, 1.0)],
            self.p_height: [(1.0, 0.0), (0.0, 1.0)],
            self.p4: [(-1.0, 0.0), (0.0, -1.0)],
        }
