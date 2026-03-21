from gettext import gettext as _

from ....core.sketcher.commands import AddFillCommand, RemoveFillCommand
from .base import SketchTool


class FillTool(SketchTool):
    """Handles creating and removing fills from closed regions."""

    ICON = "sketch-fill-symbolic"
    LABEL = _("Fill")
    SHORTCUTS = ["gf"]

    def is_available(self, target, target_type) -> bool:
        return target is None

    def shortcut_is_active(self) -> bool:
        return True

    def on_press(self, world_x: float, world_y: float, n_press: int) -> bool:
        if n_press != 1:
            return False

        mx, my = self.element.hittester.screen_to_model(
            world_x, world_y, self.element
        )
        target_loop = self.element.sketch.get_loop_at_point(mx, my)
        if not target_loop:
            return False

        sketch = self.element.sketch
        target_loop_set = frozenset(target_loop)

        existing_fill = None
        for fill in sketch.fills:
            if frozenset(fill.boundary) == target_loop_set:
                existing_fill = fill
                break

        if existing_fill:
            cmd = RemoveFillCommand(sketch, existing_fill)
        else:
            cmd = AddFillCommand(sketch, target_loop)

        self.element.execute_command(cmd)
        self.element.mark_dirty()
        return True

    def on_drag(self, world_dx: float, world_dy: float):
        pass

    def on_release(self, world_x: float, world_y: float):
        pass
