from gettext import gettext as _
from typing import Any

from raygeo.sketch.entities.bezier import Bezier

from ...core.commands import StraightenBezierCommand
from .base import SketchTool


class StraightenTool(SketchTool):
    ICON = "sketch-line-symbolic"
    LABEL = _("Straighten")

    def is_available(
        self,
        target: Any | None,
        target_type: str | None,
    ) -> bool:
        if not isinstance(target, Bezier):
            return False
        return not target.is_line

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
            entity = sketch.registry.get_entity_object(entity_id)
            if isinstance(entity, Bezier):
                bezier_ids.append(entity_id)

        if not bezier_ids:
            return

        for bezier_id in bezier_ids:
            cmd = StraightenBezierCommand(sketch, bezier_id)
            self.element.execute_command(cmd)

        self.element.set_tool("select")
