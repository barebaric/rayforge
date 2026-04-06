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

        if not editor or not sel.entity_ids:
            return

        bezier_ids = []
        for entity_id in sel.entity_ids:
            entity = sketch.registry.get_entity(entity_id)
            if isinstance(entity, Bezier):
                bezier_ids.append(entity_id)

        if not bezier_ids:
            return

        for bezier_id in bezier_ids:
            cmd = StraightenBezierCommand(sketch, bezier_id)
            self.element.execute_command(cmd)

        self.element.set_tool("select")
