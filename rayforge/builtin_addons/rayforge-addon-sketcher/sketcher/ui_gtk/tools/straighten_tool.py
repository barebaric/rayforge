from gettext import gettext as _
from typing import TYPE_CHECKING, Optional, Union

from ...core.entities import Bezier, Entity, Point
from ...core.commands import StraightenBezierCommand
from .base import SketchTool

if TYPE_CHECKING:
    from ...core.constraints import Constraint


class StraightenTool(SketchTool):
    ICON = "sketch-line-symbolic"
    LABEL = _("Straighten")

    def is_available(
        self,
        target: Optional[Union[Point, Entity, "Constraint"]],
        target_type: Optional[str],
    ) -> bool:
        if not isinstance(target, Bezier):
            return False
        return not target.is_line(self.element.sketch.registry)

    def on_press(self, world_x: float, world_y: float, n_press: int) -> bool:
        return True

    def on_drag(self, world_dx: float, world_dy: float):
        pass

    def on_release(self, world_x: float, world_y: float):
        pass

    def on_activate(self):
        sel = self.element.selection
        sketch = self.element.sketch
        editor = self.element.editor

        if not editor or len(sel.entity_ids) != 1:
            return

        entity = sketch.registry.get_entity(sel.entity_ids[0])
        if not isinstance(entity, Bezier):
            return

        cmd = StraightenBezierCommand(sketch, entity.id)
        self.element.execute_command(cmd)
        self.element.set_tool("select")
