import logging
from gettext import gettext as _
from typing import TYPE_CHECKING, Optional, Union

from ....core.sketcher.commands import AddItemsCommand
from ....core.sketcher.constraints import TangentConstraint
from ....core.sketcher.entities import Arc, Circle, Entity, Line, Point
from .base import SketchTool

if TYPE_CHECKING:
    from ....core.sketcher.constraints import Constraint

logger = logging.getLogger(__name__)


class TangentConstraintTool(SketchTool):
    ICON = "sketch-constrain-tangential-symbolic"
    LABEL = _("Tangent")
    SHORTCUT = ("t", _("Tangent"))

    def is_available(
        self,
        target: Optional[Union[Point, Entity, "Constraint"]],
        target_type: Optional[str],
    ) -> bool:
        sel = self.element.selection
        return self.element.sketch.supports_constraint(
            "tangent", sel.point_ids, sel.entity_ids
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

        sel_line = None
        sel_shape = None

        for eid in sel.entity_ids:
            e = sketch.registry.get_entity(eid)
            if isinstance(e, Line):
                sel_line = e
            elif isinstance(e, (Arc, Circle)):
                sel_shape = e

        if sel_line and sel_shape:
            constr = TangentConstraint(sel_line.id, sel_shape.id)
            cmd = AddItemsCommand(
                sketch, _("Add Tangent Constraint"), constraints=[constr]
            )
            self.element.execute_command(cmd)
        else:
            logger.warning("Select 1 Line and 1 Arc/Circle for Tangent.")
