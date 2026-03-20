import logging
from gettext import gettext as _
from typing import TYPE_CHECKING, List, Optional, Union

from ....core.sketcher.commands import FilletCommand
from ....core.sketcher.entities import Entity, Line, Point
from .base import SketchTool

if TYPE_CHECKING:
    from ....core.sketcher.constraints import Constraint

logger = logging.getLogger(__name__)


class FilletTool(SketchTool):
    ICON = "sketch-fillet-symbolic"
    LABEL = _("Fillet")
    SHORTCUT = ("cf", _("Fillet"))
    DEFAULT_RADIUS = 10.0

    def is_available(
        self,
        target: Optional[Union[Point, Entity, "Constraint"]],
        target_type: Optional[str],
    ) -> bool:
        if target_type != "junction":
            return False
        sel = self.element.selection
        if sel.junction_pid is None:
            return False
        lines = self._get_lines_at_point(sel.junction_pid)
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

    def _get_lines_at_point(self, pid: int) -> List[Line]:
        sketch = self.element.sketch
        return [
            e
            for e in sketch.registry.entities
            if isinstance(e, Line) and pid in (e.p1_idx, e.p2_idx)
        ]

    def _add_fillet(self):
        sel = self.element.selection
        sketch = self.element.sketch
        editor = self.element.editor

        if not editor:
            return

        corner_pid = sel.junction_pid
        if corner_pid is None:
            return

        lines_at_junction = self._get_lines_at_point(corner_pid)
        if len(lines_at_junction) != 2:
            return
        line1, line2 = lines_at_junction

        geom = FilletCommand.calculate_geometry(
            sketch.registry,
            corner_pid,
            line1.id,
            line2.id,
            self.DEFAULT_RADIUS,
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
            sketch.params.evaluate(self.DEFAULT_RADIUS),
        )
        self.element.execute_command(cmd)
