import logging
from gettext import gettext as _
from typing import TYPE_CHECKING, Optional, Union

from ...core.commands import AddItemsCommand
from ...core.constraints import PerpendicularConstraint
from ...core.entities import Entity, Point
from .base import SketchTool

if TYPE_CHECKING:
    from ...core.constraints import Constraint

logger = logging.getLogger(__name__)


class PerpendicularConstraintTool(SketchTool):
    ICON = "sketch-constrain-perpendicular-symbolic"
    LABEL = _("Perpendicular")
    SHORTCUTS = ["n"]

    def is_available(
        self,
        target: Optional[Union[Point, Entity, "Constraint"]],
        target_type: Optional[str],
    ) -> bool:
        return PerpendicularConstraint.can_apply_to(
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
        sel = self.element.selection
        sketch = self.element.sketch
        editor = self.element.editor

        if not editor:
            return

        e1_id = sel.entity_ids[0]
        e2_id = sel.entity_ids[1]

        constr = PerpendicularConstraint(e1_id, e2_id)
        cmd = AddItemsCommand(
            sketch,
            _("Add Perpendicular Constraint"),
            constraints=[constr],
        )
        self.element.execute_command(cmd)
