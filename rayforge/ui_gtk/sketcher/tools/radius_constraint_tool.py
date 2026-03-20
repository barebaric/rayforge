import logging
import math
from gettext import gettext as _
from typing import TYPE_CHECKING, Optional, Union

from ....core.sketcher.commands import AddItemsCommand
from ....core.sketcher.constraints import RadiusConstraint
from ....core.sketcher.entities import Arc, Circle, Entity, Point
from .base import SketchTool

if TYPE_CHECKING:
    from ....core.sketcher.constraints import Constraint

logger = logging.getLogger(__name__)


class RadiusConstraintTool(SketchTool):
    ICON = "sketch-radius-symbolic"
    LABEL = _("Radius")
    SHORTCUT = ("kr", _("Radius"))

    def is_available(
        self,
        target: Optional[Union[Point, Entity, "Constraint"]],
        target_type: Optional[str],
    ) -> bool:
        sel = self.element.selection
        return self.element.sketch.supports_constraint(
            "radius", sel.point_ids, sel.entity_ids
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

        radius = 0.0
        if isinstance(e, Arc):
            s = sketch.registry.get_point(e.start_idx)
            c = sketch.registry.get_point(e.center_idx)
            if s and c:
                radius = math.hypot(s.x - c.x, s.y - c.y)
        elif isinstance(e, Circle):
            r_pt = sketch.registry.get_point(e.radius_pt_idx)
            c = sketch.registry.get_point(e.center_idx)
            if r_pt and c:
                radius = math.hypot(r_pt.x - c.x, r_pt.y - c.y)

        if radius > 0 and e:
            constr = RadiusConstraint(e.id, radius)
            cmd = AddItemsCommand(
                sketch, _("Add Radius Constraint"), constraints=[constr]
            )
            self.element.execute_command(cmd)
        else:
            logger.warning("Could not add radius constraint.")
