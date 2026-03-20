import logging
from gettext import gettext as _
from typing import TYPE_CHECKING, Optional, Union

from ....core.sketcher.commands import AddItemsCommand
from ....core.sketcher.constraints import (
    CoincidentConstraint,
    PointOnLineConstraint,
)
from ....core.sketcher.entities import Entity, Point
from .base import SketchTool

if TYPE_CHECKING:
    from ....core.sketcher.constraints import Constraint

logger = logging.getLogger(__name__)


class CoincidentConstraintTool(SketchTool):
    ICON = "sketch-constrain-point-on-x-symbolic"
    LABEL = _("Coincident")
    SHORTCUT = ("o", _("Coincident"))

    def is_available(
        self,
        target: Optional[Union[Point, Entity, "Constraint"]],
        target_type: Optional[str],
    ) -> bool:
        sel = self.element.selection
        sketch = self.element.sketch
        return sketch.supports_constraint(
            "coincident", sel.point_ids, sel.entity_ids
        ) or sketch.supports_constraint(
            "point_on_line", sel.point_ids, sel.entity_ids
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
        sel = self.element.selection
        sketch = self.element.sketch

        if sketch.supports_constraint(
            "coincident", sel.point_ids, sel.entity_ids
        ):
            p1_id, p2_id = sel.point_ids
            constr = CoincidentConstraint(p1_id, p2_id)
            cmd = AddItemsCommand(
                sketch,
                _("Add Coincident Constraint"),
                constraints=[constr],
            )
            self.element.execute_command(cmd)
        elif sketch.supports_constraint(
            "point_on_line", sel.point_ids, sel.entity_ids
        ):
            sel_entity_id = sel.entity_ids[0]
            target_pid = sel.point_ids[0]
            constr = PointOnLineConstraint(target_pid, sel_entity_id)
            cmd = AddItemsCommand(
                sketch, _("Add Point On Shape"), constraints=[constr]
            )
            self.element.execute_command(cmd)
