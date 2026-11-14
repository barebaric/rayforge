import logging
from gettext import gettext as _
from typing import TYPE_CHECKING, ClassVar, Union

from ...core.commands import AddItemsCommand
from raygeo.sketch.constraints.coincident import CoincidentConstraint
from raygeo.sketch.constraints.point_on_curve import PointOnCurveConstraint
from raygeo.sketch.constraints.point_on_line import (
    PointOnLineConstraint,
    PointOnTarget,
)
from ...core import constraint_info
from .base import SketchTool

if TYPE_CHECKING:
    from typing import Any

logger = logging.getLogger(__name__)


class CoincidentConstraintTool(SketchTool):
    ICON = "sketch-constrain-point-symbolic"
    LABEL = _("Coincident")
    SHORTCUTS: ClassVar[list[str]] = ["o", "c"]
    PIE_GROUP: ClassVar[str | None] = "constrain"

    def is_available(
        self,
        target: Any | None,
        target_type: str | None,
    ) -> bool:
        sel = self.element.selection
        sketch = self.element.sketch
        if constraint_info.can_apply(CoincidentConstraint, sel, sketch):
            p1_id, p2_id = sel.point_ids
            return not self._has_coincident_constraint(sketch, p1_id, p2_id)
        if constraint_info.can_apply(PointOnCurveConstraint, sel, sketch):
            point_id = sel.point_ids[0]
            entity_id = sel.entity_ids[0]
            return not self._has_point_on_curve_constraint(
                sketch, point_id, entity_id
            )
        if constraint_info.can_apply(PointOnLineConstraint, sel, sketch):
            point_id = sel.point_ids[0]
            entity_id = sel.entity_ids[0]
            return not self._has_point_on_line_constraint(
                sketch, point_id, entity_id
            )
        return False

    def _has_coincident_constraint(
        self, sketch, p1_id: int, p2_id: int
    ) -> bool:
        for c in sketch.constraints:
            if isinstance(c, CoincidentConstraint) and (
                (c.p1 == p1_id and c.p2 == p2_id)
                or (c.p1 == p2_id and c.p2 == p1_id)
            ):
                return True
        return False

    def _has_point_on_line_constraint(
        self, sketch, point_id: int, entity_id: int
    ) -> bool:
        for c in sketch.constraints:
            if (
                isinstance(c, PointOnLineConstraint)
                and c.point_id == point_id
                and c.shape_id == entity_id
            ):
                return True
        return False

    def _has_point_on_curve_constraint(
        self, sketch, point_id: int, entity_id: int
    ) -> bool:
        for c in sketch.constraints:
            if (
                isinstance(c, PointOnCurveConstraint)
                and c.point_id == point_id
                and c.shape_id == entity_id
            ):
                return True
        return False

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

        if constraint_info.can_apply(CoincidentConstraint, sel, sketch):
            p1_id, p2_id = sel.point_ids
            constr = CoincidentConstraint(p1_id, p2_id)
            cmd = AddItemsCommand(
                sketch,
                _("Add Coincident Constraint"),
                constraints=[constr],
            )
            self.element.execute_command(cmd)
            sel.point_ids = [p1_id]
            sel.changed.send(sel)
        elif constraint_info.can_apply(PointOnCurveConstraint, sel, sketch):
            target_pid = sel.point_ids[0]
            sel_entity_id = sel.entity_ids[0]
            curve = sketch.registry.get_entity_object(sel_entity_id)
            constr = PointOnCurveConstraint(
                target_pid,
                curve.start_idx,
                curve.end_idx,
                curve.cp1_offset or (0.0, 0.0),
                curve.cp2_offset or (0.0, 0.0),
                curve.is_line,
            )
            cmd = AddItemsCommand(
                sketch, _("Add Point On Curve"), constraints=[constr]
            )
            self.element.execute_command(cmd)
        elif constraint_info.can_apply(PointOnLineConstraint, sel, sketch):
            sel_entity_id = sel.entity_ids[0]
            target_pid = sel.point_ids[0]
            shape = sketch.registry.get_entity_object(sel_entity_id)
            if shape.type_name() == "line":
                target = PointOnTarget.Line(shape.p1_idx, shape.p2_idx)
            elif shape.type_name() == "circle":
                target = PointOnTarget.RadiusCurve(
                    shape.center_idx, shape.radius_pt_idx
                )
            else:
                target = PointOnTarget.RadiusCurve(
                    shape.center_idx, shape.start_idx
                )
            constr = PointOnLineConstraint(target_pid, target)
            cmd = AddItemsCommand(
                sketch, _("Add Point On Shape"), constraints=[constr]
            )
            self.element.execute_command(cmd)
