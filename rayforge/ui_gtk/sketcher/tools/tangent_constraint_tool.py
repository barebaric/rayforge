import logging
from gettext import gettext as _
from typing import TYPE_CHECKING, Optional, Union

from ....core.sketcher.commands import (
    AddItemsCommand,
    TangentConstraintCommand,
)
from ....core.sketcher.constraints import TangentConstraint
from ....core.sketcher.entities import Entity, Point
from .base import SketchTool

if TYPE_CHECKING:
    from ....core.sketcher.constraints import Constraint

logger = logging.getLogger(__name__)


class TangentConstraintTool(SketchTool):
    ICON = "sketch-constrain-tangential-symbolic"
    LABEL = _("Tangent")
    SHORTCUTS = ["t"]

    def is_available(
        self,
        target: Optional[Union[Point, Entity, "Constraint"]],
        target_type: Optional[str],
    ) -> bool:
        return TangentConstraint.can_apply_to(
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

        params = TangentConstraintCommand.identify_entities(
            sketch.registry, sel.entity_ids
        )

        if params is None:
            logger.warning("Select 1 Line and 1 Arc/Circle for Tangent.")
            return

        constr = TangentConstraint(params.line_id, params.shape_id)
        cmd = AddItemsCommand(
            sketch, _("Add Tangent Constraint"), constraints=[constr]
        )
        self.element.execute_command(cmd)
