import logging
import math
from gettext import gettext as _
from typing import TYPE_CHECKING, List, Optional, Union

from ...core.commands import FilletCommand
from ...core.entities import Entity, Line, Point
from ...core.types import EntityID
from .base import SketchTool

if TYPE_CHECKING:
    from ...core.constraints import Constraint

logger = logging.getLogger(__name__)


class FilletTool(SketchTool):
    ICON = "sketch-fillet-symbolic"
    LABEL = _("Fillet")
    SHORTCUTS = ["cf"]
    DEFAULT_RADIUS_RATIO = 0.15

    def is_available(
        self,
        target: Optional[Union[Point, Entity, "Constraint"]],
        target_type: Optional[str],
    ) -> bool:
        if target_type != "junction":
            return False
        if target is None or not isinstance(target, Point):
            return False
        lines = self._get_lines_at_point(target.id)
        return len(lines) == 2

    def on_press(self, world_x: float, world_y: float, n_press: int) -> bool:
        return True

    def on_drag(self, world_dx: float, world_dy: float):
        pass

    def on_release(self, world_x: float, world_y: float):
        pass

    def on_activate(self):
        self._add_fillet()
        self.element.set_tool("select")

    def _get_lines_at_point(self, pid: EntityID) -> List[Line]:
        sketch = self.element.sketch
        return [
            e
            for e in sketch.registry.entities
            if isinstance(e, Line) and pid in (e.p1_idx, e.p2_idx)
        ]

    def _get_default_radius(self, corner_pid: EntityID) -> float:
        sketch = self.element.sketch
        corner_point = sketch.registry.get_point(corner_pid)
        if not corner_point:
            return 10.0

        lines = self._get_lines_at_point(corner_pid)
        if len(lines) != 2:
            return 10.0

        line1, line2 = lines
        other1_pid = (
            line1.p2_idx if line1.p1_idx == corner_pid else line1.p1_idx
        )
        other2_pid = (
            line2.p2_idx if line2.p1_idx == corner_pid else line2.p1_idx
        )

        other1_pt = sketch.registry.get_point(other1_pid)
        other2_pt = sketch.registry.get_point(other2_pid)

        if not other1_pt or not other2_pt:
            return 10.0

        v1 = (other1_pt.x - corner_point.x, other1_pt.y - corner_point.y)
        v2 = (other2_pt.x - corner_point.x, other2_pt.y - corner_point.y)
        len1 = math.hypot(v1[0], v1[1])
        len2 = math.hypot(v2[0], v2[1])

        min_len = min(len1, len2)
        return min_len * self.DEFAULT_RADIUS_RATIO

    def _add_fillet(self):
        sel = self.element.selection
        sketch = self.element.sketch
        editor = self.element.editor

        if not editor:
            return

        corner_pid = sel.junction_pid
        if corner_pid is None and sel.point_ids:
            corner_pid = sel.point_ids[0]

        if corner_pid is None:
            return

        lines_at_junction = self._get_lines_at_point(corner_pid)
        if len(lines_at_junction) != 2:
            return
        line1, line2 = lines_at_junction

        default_radius = self._get_default_radius(corner_pid)
        geom = FilletCommand.calculate_geometry(
            sketch.registry,
            corner_pid,
            line1.id,
            line2.id,
            default_radius,
        )
        if not geom:
            logger.warning(
                "Lines are too short or angle too acute for fillet."
            )
            return

        cmd = FilletCommand(
            sketch,
            corner_pid,
            line1.id,
            line2.id,
            sketch.params.evaluate(default_radius),
        )
        self.element.execute_command(cmd)
