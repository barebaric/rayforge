import logging
from gettext import gettext as _
from typing import TYPE_CHECKING, Optional, Union

from ...core.commands import (
    AddItemsCommand,
    DistanceConstraintCommand,
)
from ...core.constraints import DistanceConstraint
from ...core.entities import Entity, Point
from .base import SketchTool

if TYPE_CHECKING:
    from ...core.constraints import Constraint

logger = logging.getLogger(__name__)


class DistanceConstraintTool(SketchTool):
    ICON = "sketch-distance-symbolic"
    LABEL = _("Distance")
    SHORTCUTS = ["kd"]

    def is_available(
        self,
        target: Optional[Union[Point, Entity, "Constraint"]],
        target_type: Optional[str],
    ) -> bool:
        return DistanceConstraint.can_apply_to(
            self.element.selection, self.element.sketch
        )

    def on_press(self, world_x: float, world_y: float, n_press: int) -> bool:
        return True

    def on_drag(self, world_dx: float, world_dy: float):
        pass

    def on_release(self, world_x: float, world_y: float):
        pass

    def on_activate(self):
        self._add_constraint()
        self.element.set_tool("select")

    def _add_constraint(self):
        editor = self.element.editor
        sketch = self.element.sketch
        sel = self.element.selection

        if not editor:
            return

        params = DistanceConstraintCommand.calculate_distance(
            sketch.registry, sel.point_ids, sel.entity_ids
        )

        if params is None:
            logger.warning("Select 2 Points or 1 Line for Distance.")
            return

        constr = DistanceConstraint(
            params.p1_id, params.p2_id, params.distance
        )
        cmd = AddItemsCommand(
            sketch, _("Add Distance Constraint"), constraints=[constr]
        )
        self.element.execute_command(cmd)
