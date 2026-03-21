import logging
from gettext import gettext as _
from typing import TYPE_CHECKING, Optional, Union

from ....core.sketcher.commands import (
    AddItemsCommand,
    AngleConstraintCommand,
)
from ....core.sketcher.constraints import AngleConstraint
from ....core.sketcher.entities import Entity, Point
from .base import SketchTool

if TYPE_CHECKING:
    from ....core.sketcher.constraints import Constraint

logger = logging.getLogger(__name__)


class AngleConstraintTool(SketchTool):
    ICON = "sketch-constrain-angle-symbolic"
    LABEL = _("Angle")
    SHORTCUTS = ["ka"]

    def is_available(
        self,
        target: Optional[Union[Point, Entity, "Constraint"]],
        target_type: Optional[str],
    ) -> bool:
        return AngleConstraint.can_apply_to(
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

        if len(sel.entity_ids) < 2:
            logger.warning("Angle constraint requires exactly 2 lines.")
            return

        e1_id = sel.entity_ids[0]
        e2_id = sel.entity_ids[1]

        params = AngleConstraintCommand.calculate_constraint_params(
            sketch.registry, e1_id, e2_id
        )

        if params is None:
            return

        constr = AngleConstraint(
            params.anchor_id,
            params.other_id,
            params.value_deg,
            e1_far_idx=params.anchor_far_idx,
            e2_far_idx=params.other_far_idx,
        )
        cmd = AddItemsCommand(
            sketch,
            _("Add Angle Constraint"),
            constraints=[constr],
        )
        self.element.execute_command(cmd)
