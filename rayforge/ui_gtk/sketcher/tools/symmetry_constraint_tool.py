import logging
from gettext import gettext as _
from typing import TYPE_CHECKING, Optional, Union

from ....core.sketcher.commands import (
    AddItemsCommand,
    SymmetryConstraintCommand,
)
from ....core.sketcher.constraints import SymmetryConstraint
from ....core.sketcher.entities import Entity, Point
from .base import SketchTool

if TYPE_CHECKING:
    from ....core.sketcher.constraints import Constraint

logger = logging.getLogger(__name__)


class SymmetryConstraintTool(SketchTool):
    ICON = "sketch-constrain-symmetric-symbolic"
    LABEL = _("Symmetry")
    SHORTCUTS = ["s"]

    def is_available(
        self,
        target: Optional[Union[Point, Entity, "Constraint"]],
        target_type: Optional[str],
    ) -> bool:
        sel = self.element.selection
        return self.element.sketch.supports_constraint(
            "symmetry", sel.point_ids, sel.entity_ids
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
