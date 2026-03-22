import logging
from gettext import gettext as _
from typing import TYPE_CHECKING, Optional, Union

from ...core.commands import AddItemsCommand
from ...core.constraints import (
    CoincidentConstraint,
    PointOnLineConstraint,
)
from ...core.entities import Entity, Point
from .base import SketchTool

if TYPE_CHECKING:
    from ...core.constraints import Constraint

logger = logging.getLogger(__name__)


class CoincidentConstraintTool(SketchTool):
    ICON = "sketch-constrain-point-on-x-symbolic"
    LABEL = _("Coincident")
    SHORTCUTS = ["o", "c"]

    def is_available(
        self,
        target: Optional[Union[Point, Entity, "Constraint"]],
        target_type: Optional[str],
    ) -> bool:
        sel = self.element.selection
        sketch = self.element.sketch
        return CoincidentConstraint.can_apply_to(
            sel, sketch
        ) or PointOnLineConstraint.can_apply_to(sel, sketch)

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
        sel = self.element.selection
        sketch = self.element.sketch

        if CoincidentConstraint.can_apply_to(sel, sketch):
            p1_id, p2_id = sel.point_ids
            constr = CoincidentConstraint(p1_id, p2_id)
            cmd = AddItemsCommand(
                sketch,
                _("Add Coincident Constraint"),
                constraints=[constr],
            )
            self.element.execute_command(cmd)
        elif PointOnLineConstraint.can_apply_to(sel, sketch):
            sel_entity_id = sel.entity_ids[0]
            target_pid = sel.point_ids[0]
            constr = PointOnLineConstraint(target_pid, sel_entity_id)
            cmd = AddItemsCommand(
                sketch, _("Add Point On Shape"), constraints=[constr]
            )
            self.element.execute_command(cmd)
