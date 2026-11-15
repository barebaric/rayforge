import logging
from gettext import gettext as _
from typing import ClassVar

from raygeo.sketch.constraints.perpendicular import PerpendicularConstraint

from ...core import constraint_info
from ...core.commands import AddItemsCommand
from ...core.types import HitTarget
from .base import SketchTool

logger = logging.getLogger(__name__)


class PerpendicularConstraintTool(SketchTool):
    ICON = "sketch-constrain-perpendicular-symbolic"
    LABEL = _("Perpendicular")
    SHORTCUTS: ClassVar[list[str]] = ["n"]
    PIE_GROUP: ClassVar[str | None] = "constrain"

    def is_available(
        self,
        target: HitTarget | None,
        target_type: str | None,
    ) -> bool:
        return constraint_info.can_apply(
            PerpendicularConstraint,
            self.element.selection,
            self.element.sketch,
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
