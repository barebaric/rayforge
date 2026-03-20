import logging
import math
from gettext import gettext as _
from typing import TYPE_CHECKING, Optional, Tuple, Union

from ....core.sketcher.commands import AddItemsCommand
from ....core.sketcher.constraints import DistanceConstraint
from ....core.sketcher.entities import Entity, Line, Point
from .base import SketchTool

if TYPE_CHECKING:
    from ....core.sketcher.constraints import Constraint

logger = logging.getLogger(__name__)


class DistanceConstraintTool(SketchTool):
    ICON = "sketch-distance-symbolic"
    LABEL = _("Distance")
    SHORTCUT = ("kd", _("Distance"))

    def is_available(
        self,
        target: Optional[Union[Point, Entity, "Constraint"]],
        target_type: Optional[str],
    ) -> bool:
        sel = self.element.selection
        return self.element.sketch.supports_constraint(
            "dist", sel.point_ids, sel.entity_ids
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

    def _get_two_points_from_selection(self) -> Optional[Tuple[Point, Point]]:
        sel = self.element.selection
        sketch = self.element.sketch

        if len(sel.point_ids) == 2:
            p1 = sketch.registry.get_point(sel.point_ids[0])
            p2 = sketch.registry.get_point(sel.point_ids[1])
            if p1 and p2:
                return p1, p2

        if len(sel.entity_ids) == 1:
            eid = sel.entity_ids[0]
            e = sketch.registry.get_entity(eid)
            if isinstance(e, Line):
                p1 = sketch.registry.get_point(e.p1_idx)
                p2 = sketch.registry.get_point(e.p2_idx)
                if p1 and p2:
                    return p1, p2

        return None

    def _add_constraint(self):
        editor = self.element.editor
        sketch = self.element.sketch

        points = self._get_two_points_from_selection()
        if points and editor:
            p1, p2 = points
            dist = math.hypot(p1.x - p2.x, p1.y - p2.y)
            constr = DistanceConstraint(p1.id, p2.id, dist)
            cmd = AddItemsCommand(
                sketch, _("Add Distance Constraint"), constraints=[constr]
            )
            self.element.execute_command(cmd)
        else:
            logger.warning("Select 2 Points or 1 Line for Distance.")
