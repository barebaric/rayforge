# constraints/drag.py

from __future__ import annotations
from typing import (
    Tuple,
    Dict,
    List,
    TYPE_CHECKING,
)
from .base import Constraint

if TYPE_CHECKING:
    from ..params import ParameterContext
    from ..registry import EntityRegistry


class DragConstraint(Constraint):
    """
    A transient constraint used only during interaction.
    It pulls a point toward a target (mouse) coordinate.
    """

    def __init__(
        self,
        point_id: int,
        target_x: float,
        target_y: float,
        weight: float = 0.1,
    ):
        self.point_id = point_id
        self.target_x = target_x
        self.target_y = target_y
        self.weight = weight

    def error(
        self, reg: "EntityRegistry", params: "ParameterContext"
    ) -> Tuple[float, float]:
        p = reg.get_point(self.point_id)
        err_x = (p.x - self.target_x) * self.weight
        err_y = (p.y - self.target_y) * self.weight
        return err_x, err_y

    def gradient(
        self, reg: "EntityRegistry", params: "ParameterContext"
    ) -> Dict[int, List[Tuple[float, float]]]:
        return {
            self.point_id: [(self.weight, 0.0), (0.0, self.weight)],
        }
