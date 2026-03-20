import logging
from gettext import gettext as _
from typing import TYPE_CHECKING, Optional, Union

from ....core.sketcher.commands import AddItemsCommand
from ....core.sketcher.constraints import VerticalConstraint
from ....core.sketcher.entities import Entity, Line, Point
from .base import SketchTool

if TYPE_CHECKING:
    from ....core.sketcher.constraints import Constraint

logger = logging.getLogger(__name__)


class VerticalConstraintTool(SketchTool):
    ICON = "sketch-constrain-vertical-symbolic"
    LABEL = _("Vertical")
    SHORTCUT = ("v", _("Vertical"))

    def is_available(
        self,
        target: Optional[Union[Point, Entity, "Constraint"]],
        target_type: Optional[str],
    ) -> bool:
        sel = self.element.selection
        return self.element.sketch.supports_constraint(
            "vert", sel.point_ids, sel.entity_ids
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

        constraints_to_add = []

        if len(sel.point_ids) == 2 and not sel.entity_ids:
            p1_id, p2_id = sel.point_ids
            constraints_to_add.append(VerticalConstraint(p1_id, p2_id))
        elif len(sel.entity_ids) > 0 and not sel.point_ids:
            for eid in sel.entity_ids:
                e = sketch.registry.get_entity(eid)
                if isinstance(e, Line):
                    constraints_to_add.append(
                        VerticalConstraint(e.p1_idx, e.p2_idx)
                    )

        if constraints_to_add:
            cmd = AddItemsCommand(
                sketch,
                _("Add Vertical Constraint"),
                constraints=constraints_to_add,
            )
            self.element.execute_command(cmd)
