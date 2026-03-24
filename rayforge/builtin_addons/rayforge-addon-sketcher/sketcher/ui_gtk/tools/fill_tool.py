from gettext import gettext as _

from rayforge.shared.util.colors import ColorRGBA
from ...core.commands import AddFillCommand, RemoveFillCommand
from ...core.sketch import FillStyle, DEFAULT_FILL_COLOR
from .base import SketchTool


class FillTool(SketchTool):
    """Handles creating and removing fills from closed regions."""

    ICON = "sketch-fill-symbolic"
    LABEL = _("Fill")
    SHORTCUTS = ["gf"]
    CURSOR_ICON = "sketch-fill-symbolic"

    _current_color: ColorRGBA = DEFAULT_FILL_COLOR
    _current_style: FillStyle = FillStyle.SOLID

    def __init__(self, element):
        super().__init__(element)

    @classmethod
    def get_current_color(cls) -> ColorRGBA:
        """Get the current fill color for new fills."""
        return cls._current_color

    @classmethod
    def set_current_color(cls, color: ColorRGBA):
        """Set the current fill color for new fills."""
        cls._current_color = color

    @classmethod
    def get_current_style(cls) -> FillStyle:
        """Get the current fill style for new fills."""
        return cls._current_style

    @classmethod
    def set_current_style(cls, style: FillStyle):
        """Set the current fill style for new fills."""
        cls._current_style = style

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
            cmd = AddFillCommand(
                sketch,
                target_loop,
                style=self._current_style,
                color=self._current_color,
            )

        self.element.execute_command(cmd)
        self.element.mark_dirty()
        return True

    def on_drag(self, world_dx: float, world_dy: float):
        pass

    def on_release(self, world_x: float, world_y: float):
        pass
