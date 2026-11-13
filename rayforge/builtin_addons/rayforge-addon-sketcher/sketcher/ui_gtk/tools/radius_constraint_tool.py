import logging
from gettext import gettext as _
from typing import TYPE_CHECKING, ClassVar, Union

from ...core.commands import (
    AddItemsCommand,
    CreateOrEditConstraintCommand,
)
from raygeo.sketch.constraints.radius import RadiusConstraint
from ...core import constraint_info
from .base import SketchTool

if TYPE_CHECKING:
    from typing import Any

logger = logging.getLogger(__name__)


class RadiusConstraintTool(SketchTool):
    ICON = "sketch-radius-symbolic"
    LABEL = _("Radius")
    SHORTCUTS: ClassVar[list[str]] = ["kr"]
    PIE_GROUP: ClassVar[str | None] = "dimension"

    def is_available(
        self,
        target: Any | None,
        target_type: str | None,
    ) -> bool:
        return constraint_info.can_apply(RadiusConstraint, 
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
        e = sketch.registry.get_entity_object(eid)

        if not isinstance(e, (Arc, Circle)):
            logger.warning("Could not add radius constraint.")
            return

        constr = CreateOrEditConstraintCommand.create_constraint_for_entity(
            sketch, e
        )

        if constr and isinstance(constr, RadiusConstraint):
            cmd = AddItemsCommand(
                sketch, _("Add Radius Constraint"), constraints=[constr]
            )
            self.element.execute_command(cmd)
        else:
            logger.warning("Could not add radius constraint.")
