from gettext import gettext as _
from typing import TYPE_CHECKING, Optional, Union

from ....core.sketcher.entities import Bezier, Entity, Line, Point
from ....core.sketcher.entities.point import WaypointType
from ....core.sketcher.commands import SetWaypointTypeCommand
from .base import SketchTool

if TYPE_CHECKING:
    from ....core.sketcher.constraints import Constraint


class WaypointSmoothTool(SketchTool):
    ICON = "sketch-bezier-smooth-symbolic"
    LABEL = _("Smooth")

    def is_available(
        self,
        target: Optional[Union[Point, Entity, "Constraint"]],
        target_type: Optional[str],
    ) -> bool:
        pid = self._get_waypoint_pid()
        if pid is None:
            return False
        return self._is_waypoint_at_bezier(pid) or self._is_waypoint_at_line(
            pid
        )

    def on_press(self, world_x: float, world_y: float, n_press: int) -> bool:
        return True

    def on_drag(self, world_dx: float, world_dy: float):
        pass

    def on_release(self, world_x: float, world_y: float):
        pass

    def on_activate(self):
        self._set_waypoint_type(WaypointType.SMOOTH)
        self.element.set_tool("select")

    def _get_waypoint_pid(self) -> Optional[int]:
        sel = self.element.selection
        if sel.junction_pid is not None:
            return sel.junction_pid
        elif len(sel.point_ids) == 1:
            return sel.point_ids[0]
        return None

    def _is_waypoint_at_bezier(self, pid: int) -> bool:
        sketch = self.element.sketch
        for entity in sketch.registry.entities:
            if isinstance(entity, Bezier):
                if pid in (entity.start_idx, entity.end_idx):
                    return True
        return False

    def _is_waypoint_at_line(self, pid: int) -> bool:
        sketch = self.element.sketch
        for entity in sketch.registry.entities:
            if isinstance(entity, Line):
                if pid in (entity.p1_idx, entity.p2_idx):
                    return True
        return False

    def _set_waypoint_type(self, new_type: WaypointType):
        editor = self.element.editor
        sketch = self.element.sketch

        if not editor:
            return

        pid = self._get_waypoint_pid()
        if pid is None:
            return

        cmd = SetWaypointTypeCommand(sketch, pid, new_type)
        self.element.execute_command(cmd)
