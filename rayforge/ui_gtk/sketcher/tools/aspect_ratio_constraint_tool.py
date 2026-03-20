import logging
import math
from gettext import gettext as _
from typing import TYPE_CHECKING, Optional, Union

from ....core.sketcher.commands import AddItemsCommand
from ....core.sketcher.constraints import AspectRatioConstraint
from ....core.sketcher.entities import Entity, Line, Point
from .base import SketchTool

if TYPE_CHECKING:
    from ....core.sketcher.constraints import Constraint

logger = logging.getLogger(__name__)


class AspectRatioConstraintTool(SketchTool):
    ICON = "sketch-constrain-aspect-symbolic"
    LABEL = _("Aspect Ratio")
    SHORTCUT = ("kx", _("Aspect Ratio"))

    def is_available(
        self,
        target: Optional[Union[Point, Entity, "Constraint"]],
        target_type: Optional[str],
    ) -> bool:
        sel = self.element.selection
        return self.element.sketch.supports_constraint(
            "aspect_ratio", sel.point_ids, sel.entity_ids
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

        if len(sel.entity_ids) != 2:
            return

        e1_id = sel.entity_ids[0]
        e2_id = sel.entity_ids[1]

        e1 = sketch.registry.get_entity(e1_id)
        e2 = sketch.registry.get_entity(e2_id)

        if not isinstance(e1, Line) or not isinstance(e2, Line):
            logger.warning("Aspect ratio constraint requires 2 lines.")
            return

        p1 = sketch.registry.get_point(e1.p1_idx)
        p2 = sketch.registry.get_point(e1.p2_idx)
        p3 = sketch.registry.get_point(e2.p1_idx)
        p4 = sketch.registry.get_point(e2.p2_idx)

        if not all([p1, p2, p3, p4]):
            logger.warning("Could not resolve all points for aspect ratio.")
            return

        dist1 = math.hypot(p2.x - p1.x, p2.y - p1.y)
        dist2 = math.hypot(p4.x - p3.x, p4.y - p3.y)

        if dist2 < 1e-9:
            logger.warning("Second line has zero length.")
            return

        ratio = dist1 / dist2
        constr = AspectRatioConstraint(p1.id, p2.id, p3.id, p4.id, ratio)
        cmd = AddItemsCommand(
            sketch, _("Add Aspect Ratio Constraint"), constraints=[constr]
        )
        self.element.execute_command(cmd)
