import logging
import math
from gettext import gettext as _
from typing import ClassVar

from raygeo.sketch.entities.line import Line

from ...core.commands import ChamferCommand
from ...core.types import EntityID, HitTarget
from .base import SketchTool

logger = logging.getLogger(__name__)


class ChamferTool(SketchTool):
    ICON = "sketch-chamfer-symbolic"
    LABEL = _("Chamfer")
    SHORTCUTS: ClassVar[list[str]] = ["ch"]
    DEFAULT_DISTANCE_RATIO = 0.15

    def is_available(
        self,
        target: HitTarget | None,
        target_type: str | None,
    ) -> bool:
        if target_type != "junction" or not isinstance(target, int):
            return False
        lines = self._get_lines_at_point(target)
        return len(lines) == 2

    def on_press(self, world_x: float, world_y: float, n_press: int) -> bool:
        return True

    def on_drag(self, world_dx: float, world_dy: float):
        pass

    def on_release(self, world_x: float, world_y: float):
        pass

    def on_activate(self):
        self._add_chamfer()
        self.element.set_tool("select")

    def _get_lines_at_point(self, pid: EntityID) -> list[Line]:
        sketch = self.element.sketch
        return [
            e
            for e in sketch.registry.entities()
            if isinstance(e, Line) and pid in (e.p1_idx, e.p2_idx)
        ]

    def _get_default_distance(self, corner_pid: EntityID) -> float:
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

        v1 = (other1_pt[0] - corner_point[0], other1_pt[1] - corner_point[1])
        v2 = (other2_pt[0] - corner_point[0], other2_pt[1] - corner_point[1])
        len1 = math.hypot(v1[0], v1[1])
        len2 = math.hypot(v2[0], v2[1])

        min_len = min(len1, len2)
        return min_len * self.DEFAULT_DISTANCE_RATIO

    def _add_chamfer(self):
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

        default_distance = self._get_default_distance(corner_pid)
        cmd = ChamferCommand(
            sketch,
            corner_pid,
            line1.id,
            line2.id,
            sketch.params.evaluate(default_distance),
        )
        self.element.execute_command(cmd)
