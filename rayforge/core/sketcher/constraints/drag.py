# constraints/drag.py

from __future__ import annotations
from typing import Dict, List, TYPE_CHECKING
from ...geo import Point
from ..types import EntityID
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
        point_id: EntityID,
        target_x: float,
        target_y: float,
        weight: float = 0.1,
        user_visible: bool = True,
    ):
        self.point_id = point_id
        self.target_x = target_x
        self.target_y = target_y
        self.weight = weight
        super().__init__(user_visible=user_visible)

    def error(
        self, reg: "EntityRegistry", params: "ParameterContext"
    ) -> Point:
        p = reg.get_point(self.point_id)
        err_x = (p.x - self.target_x) * self.weight
        err_y = (p.y - self.target_y) * self.weight
        return err_x, err_y

    def gradient(
        self, reg: "EntityRegistry", params: "ParameterContext"
    ) -> Dict[EntityID, List[Point]]:
        return {
            self.point_id: [(self.weight, 0.0), (0.0, self.weight)],
        }
