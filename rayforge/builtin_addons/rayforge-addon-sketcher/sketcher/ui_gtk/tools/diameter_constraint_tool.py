import logging
from gettext import gettext as _
from typing import TYPE_CHECKING, Optional, Union

from ...core.commands import (
    AddItemsCommand,
    CreateOrEditConstraintCommand,
)
from ...core.constraints import DiameterConstraint
from ...core.entities import Circle, Entity, Point
from .base import SketchTool

if TYPE_CHECKING:
    from ...core.constraints import Constraint

logger = logging.getLogger(__name__)


class DiameterConstraintTool(SketchTool):
    ICON = "sketch-diameter-symbolic"
    LABEL = _("Diameter")
    SHORTCUTS = ["ko"]

    def is_available(
        self,
        target: Optional[Union[Point, Entity, "Constraint"]],
        target_type: Optional[str],
    ) -> bool:
        return DiameterConstraint.can_apply_to(
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

        eid = sel.entity_ids[0]
        e = sketch.registry.get_entity(eid)

        if not isinstance(e, Circle):
            logger.warning("Selected entity is not a Circle.")
            return

        constr = CreateOrEditConstraintCommand.create_constraint_for_entity(
            sketch, e
        )

        if constr:
            cmd = AddItemsCommand(
                sketch,
                _("Add Diameter Constraint"),
                constraints=[constr],
            )
            self.element.execute_command(cmd)
