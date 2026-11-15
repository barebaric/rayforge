import logging
from gettext import gettext as _
from typing import ClassVar

from raygeo.sketch.constraints.symmetry import SymmetryConstraint

from ...core import constraint_info
from ...core.commands import (
    AddItemsCommand,
    SymmetryConstraintCommand,
)
from ...core.types import HitTarget
from .base import SketchTool

logger = logging.getLogger(__name__)


class SymmetryConstraintTool(SketchTool):
    ICON = "sketch-constrain-symmetric-symbolic"
    LABEL = _("Symmetry")
    SHORTCUTS: ClassVar[list[str]] = ["s"]
    PIE_GROUP: ClassVar[str | None] = "constrain"

    def is_available(
        self,
        target: HitTarget | None,
        target_type: str | None,
    ) -> bool:
        return constraint_info.can_apply(
            SymmetryConstraint, self.element.selection, self.element.sketch
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

        params = SymmetryConstraintCommand.determine_constraint_params(
            sel.point_ids, sel.entity_ids
        )

        if params is None:
            return

        if params.center_id is not None:
            constr = SymmetryConstraint(
                params.p1_id, params.p2_id, center=params.center_id
            )
        else:
            constr = SymmetryConstraint(
                params.p1_id, params.p2_id, axis=params.axis_id
            )

        cmd = AddItemsCommand(
            sketch, _("Add Symmetry Constraint"), constraints=[constr]
        )
        self.element.execute_command(cmd)
